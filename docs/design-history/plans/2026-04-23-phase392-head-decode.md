# Phase 3.9.2 — Attention Head OV Output Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /api/sessions/{name}/decode-head` and extend `PerHeadPatchingHeatmap`'s pin card to show which vocabulary tokens an attention head most strongly promotes/suppresses via the dominant left-singular vector of its `o_proj` slice.

**Architecture:** Backend SVD-decomposes `W_O[layer][:, h*head_dim:(h+1)*head_dim]`, takes the dominant left-singular vector × singular value, decodes via `W_U @ direction`. Orients sign so promoted tokens dominate. Frontend extends the existing pin card with a conditional decode block (only when `unit` starts with `"attn.h"`; for `"ffn"` shows a caption redirecting to approx_neuron mode).

**Tech Stack:** Python 3.11 + PyTorch + FastAPI + React 18 + TypeScript. No new deps.

**Spec:** `testing/docs/superpowers/specs/2026-04-23-phase392-head-decode.md` (commit `c8ccee4`).

**Tool rules (for every subagent prompt):**
- Use Read (not cat), Edit (not Bash sed/awk/cat), Grep (not Bash grep/rg/awk), Glob (not find)
- Git: `git -C /home/ai/ai-projects/llm <cmd>`
- For tsc/pyright/pytest/vitest/playwright/git commit: pass `dangerouslyDisableSandbox: true`
- If Bash is denied TWICE on same command, STOP and report BLOCKED
- Pyright/tsc must be 0/0/0 after every task

---

## File Structure

**Backend**
- **Modify** `testing/gui/backend/routes/sessions.py`
  - Add `DecodeHeadRequest` pydantic model near `DecodeNeuronRequest`
  - Add `POST /sessions/{name}/decode-head` endpoint after `decode_neuron`

**Tests**
- **Create** `testing/tests/test_decode_head.py` — 6 unit tests + 1 TinyLlama integration

**Frontend**
- **Modify** `testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx`
  - Add decode fetch state + `useEffect` tied to `pinned.cell.unit` parsing
  - Extend pin card render with conditional top/bottom token block
- **Modify** `testing/gui/frontend/tests/e2e/smoke.spec.ts` — 17th test

---

## Task 1: Backend endpoint + unit tests + TinyLlama integration

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py`
- Create: `testing/tests/test_decode_head.py`

- [ ] **Step 1: Add pydantic model + endpoint**

In `testing/gui/backend/routes/sessions.py`, find the `decode_neuron` handler (added in Phase 3.9.1, near line ~525+). Immediately AFTER its closing `return result` + blank line, insert:

```python
class DecodeHeadRequest(BaseModel):
    layer: int
    head: int
    top_k: int = 10


@router.post("/sessions/{name}/decode-head")
async def decode_head(name: str, req: DecodeHeadRequest):
    """Return top-k and bottom-k tokens along the dominant write direction
    of attention head `head` at `layer`, computed via SVD of
    W_O[:, h*head_dim:(h+1)*head_dim].

    Sign is oriented so promoted tokens dominate. Response:
      {
        "top_tokens":    [{"token": str, "logit": float}, ...],
        "bottom_tokens": [{"token": str, "logit": float}, ...],
        "singular_value_ratio": float  # S[0]^2 / sum(S^2)
      }
    """
    import torch

    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    if info.model is None:
        raise HTTPException(500, "Session has no PyTorch model loaded")
    if info.tokenizer is None:
        raise HTTPException(500, "Session has no tokenizer loaded")

    model = info.model
    num_layers = len(model.model.layers)
    if not 0 <= req.layer < num_layers:
        raise HTTPException(
            400,
            f"layer out of range: got {req.layer}, valid [0, {num_layers})",
        )
    n_heads: int = model.config.num_attention_heads
    if not 0 <= req.head < n_heads:
        raise HTTPException(
            400,
            f"head out of range: got {req.head}, valid [0, {n_heads})",
        )

    def _compute() -> dict:
        with torch.no_grad():
            W_O = model.model.layers[req.layer].self_attn.o_proj.weight  # [hidden, hidden]
            hidden = W_O.shape[0]
            head_dim = hidden // n_heads
            start = req.head * head_dim
            end = start + head_dim
            W_O_h = W_O[:, start:end]  # [hidden, head_dim]

            # torch.linalg.svd with full_matrices=False gives U: [hidden, head_dim],
            # S: [head_dim], Vt: [head_dim, head_dim]. Cast to fp32 for numerical
            # stability on fp16 models.
            W_O_h_f32 = W_O_h.to(dtype=torch.float32)
            U, S, _Vt = torch.linalg.svd(W_O_h_f32, full_matrices=False)

            sv_energy_total = float((S ** 2).sum().item())
            sv_ratio = (
                float((S[0] ** 2).item()) / sv_energy_total
                if sv_energy_total > 0 else 0.0
            )

            write_direction = U[:, 0] * S[0]  # [hidden]
            W_U = model.lm_head.weight.to(dtype=torch.float32)  # [vocab, hidden]
            scores = W_U @ write_direction     # [vocab]

            vocab_size = int(scores.shape[0])
            top_k = max(1, min(req.top_k, 50, vocab_size))

            top_vals, top_ids = torch.topk(scores, top_k, largest=True)
            bot_vals, bot_ids = torch.topk(scores, top_k, largest=False)

            # Sign orientation: flip if "suppressed" has more magnitude than "promoted".
            if abs(float(top_vals.sum().item())) < abs(float(bot_vals.sum().item())):
                scores = -scores
                top_vals, top_ids = torch.topk(scores, top_k, largest=True)
                bot_vals, bot_ids = torch.topk(scores, top_k, largest=False)

        tok = info.tokenizer
        top_tokens = [
            {"token": tok.decode([int(i)], skip_special_tokens=False), "logit": float(v)}
            for i, v in zip(top_ids.tolist(), top_vals.tolist())
        ]
        bottom_tokens = [
            {"token": tok.decode([int(i)], skip_special_tokens=False), "logit": float(v)}
            for i, v in zip(bot_ids.tolist(), bot_vals.tolist())
        ]
        return {
            "top_tokens": top_tokens,
            "bottom_tokens": bottom_tokens,
            "singular_value_ratio": sv_ratio,
        }

    result = await asyncio.to_thread(_compute)
    return result
```

- [ ] **Step 2: Write the test file**

Create `testing/tests/test_decode_head.py`. Start with shared fixture (similar to `test_decode_neuron.py` but adjusted for the head-shaped mock):

```python
"""Unit + integration tests for POST /api/sessions/{name}/decode-head (Phase 3.9.2)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
import os

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient


class _MockTok:
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"<{i}>" for i in ids)


class _MockAttn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _MockAttn(hidden)


class _MockInner(nn.Module):
    def __init__(self, n_layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(hidden) for _ in range(n_layers)])


class _MockLMHead(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 10, hidden: int = 8, n_heads: int = 2, n_layers: int = 2) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.model = _MockInner(n_layers, hidden)
        self.lm_head = _MockLMHead(vocab, hidden)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            num_attention_heads=n_heads,
            vocab_size=vocab,
        )


@pytest.fixture
def app_with_mock_session():
    from gui.backend.app import app  # noqa: PLC0415
    from gui.backend.routes.sessions import get_manager  # noqa: PLC0415

    mgr = get_manager()
    session_name = "mock-decode-head"
    mgr._sessions[session_name] = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        name=session_name,
        model=_MockModel(),
        tokenizer=_MockTok(),
        llama=None,
        dirty=False,
        original_layer=lambda i: i,
        _layer_map=[],
    )
    try:
        yield (app, session_name)
    finally:
        mgr._sessions.pop(session_name, None)  # pyright: ignore[reportAttributeAccessIssue]


class TestDecodeHeadUnit:
    def test_returns_topk_bottomk_and_sv_ratio(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 1, "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == 3
        assert len(body["bottom_tokens"]) == 3
        top_logits = [t["logit"] for t in body["top_tokens"]]
        bot_logits = [t["logit"] for t in body["bottom_tokens"]]
        assert top_logits == sorted(top_logits, reverse=True)
        assert bot_logits == sorted(bot_logits)
        assert 0.0 <= body["singular_value_ratio"] <= 1.0

    def test_invalid_layer_returns_400(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 99, "head": 0, "top_k": 3})
        assert resp.status_code == 400
        assert "layer" in resp.json()["detail"]

    def test_invalid_head_returns_400(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 99, "top_k": 3})
        assert resp.status_code == 400
        assert "head" in resp.json()["detail"]

    def test_top_k_clamped_to_vocab(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 1000})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == min(50, 10)  # mock vocab=10

    def test_top_k_floor_at_1(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 0})
        assert resp.status_code == 200
        assert len(resp.json()["top_tokens"]) == 1

    def test_unknown_session_returns_404(self, app_with_mock_session) -> None:
        app, _ = app_with_mock_session
        client = TestClient(app)
        resp = client.post("/api/sessions/does-not-exist/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 3})
        assert resp.status_code == 404

    def test_sign_oriented_by_promoted_magnitude(self, app_with_mock_session) -> None:
        """After orientation, |sum(top logits)| >= |sum(bottom logits)|."""
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 5})
        body = resp.json()
        top_sum = sum(t["logit"] for t in body["top_tokens"])
        bot_sum = sum(t["logit"] for t in body["bottom_tokens"])
        assert abs(top_sum) >= abs(bot_sum)


# ---- TinyLlama integration ----

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestDecodeHeadTinyLlama:
    def test_top_ap_head_decode_has_signal(self) -> None:
        """On capital-of-France, pick the top positive-AP head via
        attribution_patch_per_head, then verify direct SVD+decode yields
        10 non-empty tokens and sv_ratio in [0, 1]."""
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import attribution_patch_per_head

        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        paris_id = int(tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item())
        rome_id = int(tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item())

        r = attribution_patch_per_head(
            model, tok, clean, corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=rome_id,
            direction="denoise",
        )
        # Phase 3.6 cells have unit like "attn.hN" or "ffn". Pick first
        # positive-AP attn.h* cell.
        top_head = next(
            (c for c in r.cells
             if c["ap_recovery"] > 0 and isinstance(c.get("unit"), str)
             and c["unit"].startswith("attn.h")),
            None,
        )
        assert top_head is not None, "expected at least one positive-AP attn head"

        L = int(top_head["layer"])
        h = int(str(top_head["unit"])[len("attn.h"):])

        # Direct SVD decode — mirrors endpoint logic.
        with torch.no_grad():
            W_O = model.model.layers[L].self_attn.o_proj.weight  # [hidden, hidden]
            n_heads = model.config.num_attention_heads
            head_dim = W_O.shape[0] // n_heads
            W_O_h = W_O[:, h*head_dim:(h+1)*head_dim].to(dtype=torch.float32)
            U, S, _ = torch.linalg.svd(W_O_h, full_matrices=False)
            direction = U[:, 0] * S[0]
            W_U = model.lm_head.weight.to(dtype=torch.float32)
            scores = W_U @ direction
            top_ids = torch.topk(scores, 10, largest=True).indices.tolist()
            sv_ratio = float((S[0] ** 2).item()) / float((S ** 2).sum().item())

        decoded = [tok.decode([i], skip_special_tokens=False) for i in top_ids]
        assert len(decoded) == 10
        assert any(s.strip() != "" for s in decoded)
        assert 0.0 <= sv_ratio <= 1.0
```

- [ ] **Step 3: Run unit tests**

Run:
```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_decode_head.py -v -k "Unit"
```
Expected: 7 tests pass (TestDecodeHeadUnit × 7).

- [ ] **Step 4: Run TinyLlama integration**

Run:
```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_decode_head.py::TestDecodeHeadTinyLlama -v -s
```
Expected: passes in ~1 min.

- [ ] **Step 5: Pyright clean**

Run:
```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py testing/tests/test_decode_head.py
```
Expected: 0/0/0.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/backend/routes/sessions.py testing/tests/test_decode_head.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(backend): POST /sessions/{name}/decode-head endpoint + tests

SVD-decomposes W_O[L][:, h*head_dim:(h+1)*head_dim], takes dominant
left-singular vector × singular value, decodes via W_U @ direction.
Sign is oriented so promoted tokens dominate (flip if |sum(top)| <
|sum(bot)|). Response includes singular_value_ratio = S[0]^2/Σ S^2
so the user knows how faithful the rank-1 decoding is.

Unit tests: 7 (shape, layer/head 400, top_k clamp, 404, sign
orientation). TinyLlama integration: runs Phase 3.6 per-head AP,
picks top positive-AP attn head, verifies direct SVD decode yields
10 non-empty tokens + valid sv_ratio.

Reference: Elhage et al. 2021 (OV-circuit framing).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Frontend pin-card extension + Playwright smoke

**Files:**
- Modify: `testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx`
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Add fetch state + effect to PerHeadPatchingHeatmap**

Open `PerHeadPatchingHeatmap.tsx`. Find the existing imports (line 1):

```tsx
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
```

(If `useEffect` isn't already imported, add it. Plan assumes it is since the file already uses it per grep.)

At the top of the component body, after `const [pinned, setPinned] = useState<PinnedCell | null>(null);` (line 21), add:

```tsx
  const [decodeState, setDecodeState] = useState<{
    loading: boolean;
    error: string | null;
    data: {
      top: Array<{ token: string; logit: number }>;
      bottom: Array<{ token: string; logit: number }>;
      sv_ratio: number;
    } | null;
  }>({ loading: false, error: null, data: null });

  useEffect(() => {
    if (pinned === null) {
      setDecodeState({ loading: false, error: null, data: null });
      return;
    }
    const unit = pinned.cell.unit;
    if (typeof unit !== "string" || !unit.startsWith("attn.h")) {
      setDecodeState({ loading: false, error: null, data: null });
      return;
    }
    const headIdx = parseInt(unit.slice("attn.h".length), 10);
    if (Number.isNaN(headIdx)) {
      setDecodeState({ loading: false, error: null, data: null });
      return;
    }
    const layer = pinned.cell.layer;
    if (typeof layer !== "number") return;

    const ctrl = new AbortController();
    setDecodeState({ loading: true, error: null, data: null });
    fetch(`/api/sessions/${result.sessionName}/decode-head`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ layer, head: headIdx, top_k: 10 }),
      signal: ctrl.signal,
    })
      .then((r) =>
        r.ok ? r.json() : Promise.reject(new Error(`decode-head ${r.status}`)),
      )
      .then((body: {
        top_tokens: Array<{ token: string; logit: number }>;
        bottom_tokens: Array<{ token: string; logit: number }>;
        singular_value_ratio: number;
      }) => {
        setDecodeState({
          loading: false,
          error: null,
          data: {
            top: body.top_tokens,
            bottom: body.bottom_tokens,
            sv_ratio: body.singular_value_ratio,
          },
        });
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          setDecodeState({ loading: false, error: err.message, data: null });
        }
      });
    return () => ctrl.abort();
  }, [pinned, result.sessionName]);
```

- [ ] **Step 2: Extend the pin card render**

In the same file, find the existing pin-card block (approximately lines 199-232, starting with `{pinned && (`). Replace the caption block (line 228-230) — currently:

```tsx
          <div style={{ fontSize: 10, color: "#666", marginTop: 6 }}>
            First-order approximation &mdash; run exact mode to confirm.
          </div>
```

with a conditional that renders the decode data:

```tsx
          <div style={{ fontSize: 10, color: "#666", marginTop: 6 }}>
            First-order approximation &mdash; run exact mode to confirm.
          </div>
          {typeof pinned.cell.unit === "string" && pinned.cell.unit.startsWith("attn.h") && (
            <div style={{ marginTop: 10, borderTop: "1px solid #234", paddingTop: 8 }}>
              {decodeState.loading && <div style={{ color: "#aaa" }}>loading decode…</div>}
              {decodeState.error && (
                <div style={{ color: "#c88" }}>decode error: {decodeState.error}</div>
              )}
              {decodeState.data && (
                <div>
                  <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>
                    Dominant write direction (sv energy ratio:{" "}
                    {(decodeState.data.sv_ratio * 100).toFixed(0)}%)
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontFamily: "monospace", fontSize: 11 }}>
                    <div>
                      <div style={{ color: "#8abaff", marginBottom: 2 }}>promoted</div>
                      {decodeState.data.top.slice(0, 5).map((t, i) => (
                        <div key={`ht-${i}`} style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 100 }}>
                            {t.token}
                          </span>
                          <span style={{ color: "#4caf50" }}>+{t.logit.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                    <div>
                      <div style={{ color: "#ffca8a", marginBottom: 2 }}>suppressed</div>
                      {decodeState.data.bottom.slice(0, 5).map((t, i) => (
                        <div key={`hb-${i}`} style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 100 }}>
                            {t.token}
                          </span>
                          <span style={{ color: "#c62828" }}>{t.logit.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          {pinned.cell.unit === "ffn" && (
            <div style={{ fontSize: 10, color: "#888", marginTop: 8, fontStyle: "italic" }}>
              FFN blocks are decoded per-neuron &mdash; switch to the
              approx_neuron mode for interpretation.
            </div>
          )}
```

- [ ] **Step 3: Tsc clean**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Vitest still clean**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npx vitest run
```
Expected: 19/19 tests pass (no new Vitest).

- [ ] **Step 5: Add 17th Playwright smoke test**

Append to `testing/gui/frontend/tests/e2e/smoke.spec.ts`:

```ts
test("per-head pin card shows decoded tokens for attn head", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Intercept decode-head with a stub.
  await page.route("**/api/sessions/*/decode-head", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens: [
          { token: " Paris", logit: 3.21 },
          { token: " Lyon", logit: 2.10 },
          { token: " France", logit: 1.85 },
          { token: " French", logit: 1.42 },
          { token: " Seine", logit: 1.05 },
        ],
        bottom_tokens: [
          { token: " Rome", logit: -2.78 },
          { token: " Milan", logit: -1.92 },
          { token: " Italy", logit: -1.60 },
          { token: " Italian", logit: -1.33 },
          { token: " Vatican", logit: -1.10 },
        ],
        singular_value_ratio: 0.58,
      }),
    });
  });

  const fixture = fs.readFileSync(PH_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-head.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Wait for the heatmap to render. Click the first attn-head cell
  // (the SVG heatmap renders cells as <rect> elements; clicking triggers
  // the existing Phase 3.6 pin).
  await page.locator("svg rect").first().click();

  // Pin card assertions. The card is rendered position: fixed with specific
  // styles; we target by text instead.
  await expect(page.getByText(/Dominant write direction \(sv energy ratio:/i))
    .toBeVisible({ timeout: 5000 });
  await expect(page.getByText(" Paris", { exact: true })).toBeVisible();
  await expect(page.getByText(" Rome", { exact: true })).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});
```

Note: if clicking the first `svg rect` pins an FFN row instead of an attn head, the assertions will fail because the FFN caption doesn't show decoded tokens. Check the per-head fixture's data: if the first cell (by heatmap layout) is `unit: "ffn"`, target a different cell (e.g., `page.locator("svg rect").nth(1)` or filter by selector). You may need to inspect the fixture first:

```bash
grep -o '"unit":"[^"]*"' testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-head.json | head -5
```

Pick a cell selector that lands on an `attn.h*` row. If all fixture cells are `attn.h*`, any `svg rect` click works.

- [ ] **Step 6: Run Playwright**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```
Expected: 17/17 tests pass.

- [ ] **Step 7: Final verification**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
cd /home/ai/ai-projects/llm/testing/gui/frontend && npx vitest run
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py testing/tests/test_decode_head.py
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_decode_head.py -v -k "Unit"
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```
Expected: all green, 0/0/0 pyright, 7 unit decode-head tests pass, Vitest 19/19, Playwright 17/17.

- [ ] **Step 8: Commit + update roadmap memory**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx testing/gui/frontend/tests/e2e/smoke.spec.ts
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(gui/frontend): per-head pin card decodes attn head OV output

Clicking an attn.h* cell in the per-head heatmap now fetches the
decode-head endpoint and renders the top-5 promoted + bottom-5
suppressed tokens inside the existing pin card, plus the singular-
value energy ratio (S[0]^2 / ΣS^2) so the user can see how faithful
the rank-1 decoding is. FFN cells show a caption redirecting to
approx_neuron mode.

17th Playwright smoke test exercises the click → decode flow with a
route-intercepted stub (no backend needed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Update roadmap memory at `~/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` with the Phase 3.9.2 shipped entry.

---

## Verification Matrix

| Check | Command | Expected |
|-------|---------|----------|
| Pyright | `.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py testing/tests/test_decode_head.py` | 0/0/0 |
| Tsc | `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit` | clean |
| Python unit | `.venv/bin/python -m pytest testing/tests/test_decode_head.py -v -k "Unit"` | 7 pass |
| TinyLlama | `.venv/bin/python -m pytest testing/tests/test_decode_head.py::TestDecodeHeadTinyLlama -v` | pass ~1 min |
| Regression (no GPU) | `.venv/bin/python -m pytest testing/tests/ -k "not TinyLlama"` | 451+ pass |
| Vitest | `cd testing/gui/frontend && npx vitest run` | 19/19 |
| Playwright | `cd testing/gui/frontend && npm run e2e` | 17/17 |

---

## Plan Self-Review Notes

**Spec coverage:**
- §2 G1 (endpoint) → Task 1.
- §2 G2 (pin-card extension) → Task 2.
- §2 G3 (tests) → Task 1 (unit + TinyLlama), Task 2 (Playwright).
- §4 math (SVD, sign orientation, sv_ratio) → Task 1 Step 1 code.

**Placeholder scan:** None.

**Type consistency:**
- `DecodeHeadRequest {layer, head, top_k}` matches frontend POST body exactly.
- Response keys (`top_tokens`, `bottom_tokens`, `singular_value_ratio`) consistent across backend emit + frontend fetch + Playwright stub.
- `sv_ratio` on frontend state maps from backend's `singular_value_ratio` — rename is intentional for terser UI code.

Ready for execution.
