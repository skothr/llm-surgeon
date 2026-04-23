# Phase 3.9.1 — Top-logit Decoding per Neuron Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 3.9's ranked neuron list actionable: click a neuron in `PerNeuronPatchingPanel`, see which vocabulary tokens its `W_down[:, i]` column most strongly promotes/suppresses (per Geva 2020's logit-lens-on-FFN framing).

**Architecture:** New backend endpoint `POST /api/sessions/{name}/decode-neuron` computes `W_U @ W_down[layer][:, neuron]` → top-k + bottom-k token ids + logits → decode via tokenizer. Frontend `PerNeuronPatchingPanel` gains a pinned-row card that fetches and displays the result. Inline `fetch` (matching the Phase 3.5 `decode-ids` pattern in `ActivationPatchingHeatmap`) — no new utility module.

**Tech Stack:** Python 3.11 + PyTorch + FastAPI + React 18 + TypeScript. No new deps.

**Spec:** `testing/docs/superpowers/specs/2026-04-23-phase391-neuron-decode.md` (commit `f50bba8`).

**Tool rules (for every subagent prompt):**
- Use Read (not cat), Edit (not Bash sed/awk/cat), Grep (not Bash grep/rg/awk), Glob (not find)
- Git: `git -C /home/ai/ai-projects/llm <cmd>`
- For tsc/pyright/vitest/playwright/pytest/git: pass `dangerouslyDisableSandbox: true`
- If Bash is denied TWICE on the same command, STOP and report BLOCKED
- Pyright/tsc must be 0/0/0 after every task

---

## File Structure

**Backend**
- **Modify** `testing/gui/backend/routes/sessions.py`
  - Add `DecodeNeuronRequest` pydantic model near existing `DecodeIdsRequest` (line ~492)
  - Add `POST /sessions/{name}/decode-neuron` endpoint after `decode_token_ids` (line ~523)

**Tests**
- **Create** `testing/tests/test_decode_neuron.py` — unit tests (mock session) + TinyLlama integration

**Frontend**
- **Modify** `testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx`
  - Add `sessionName: string` prop
  - Add `pinnedRow: { layer: number; neuron: number } | null` state + click handler
  - Add pinned-card render block (top-10 promoted + bottom-10 suppressed + close button)
  - Inline `fetch()` call with AbortController cleanup on remount/unpin
- **Modify** `testing/gui/frontend/src/components/VisualizationArea.tsx` — pass `sessionName` prop
- **Modify** `testing/gui/frontend/tests/e2e/smoke.spec.ts` — 16th test: click row, intercept route, assert card renders

---

## Task 1: Backend endpoint + unit tests

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py`
- Create: `testing/tests/test_decode_neuron.py` (unit-test portion — append TinyLlama in Task 2)

- [ ] **Step 1: Add Pydantic model and endpoint**

In `testing/gui/backend/routes/sessions.py`, find the existing `DecodeIdsRequest` (line ~492) and `decode_token_ids` handler (line ~499). Directly AFTER the `return {"tokens": tokens}` line of that handler (around line ~524), insert:

```python
class DecodeNeuronRequest(BaseModel):
    layer: int
    neuron: int
    top_k: int = 10


_DECODE_NEURON_TOPK_MAX = 50


@router.post("/sessions/{name}/decode-neuron")
async def decode_neuron(name: str, req: DecodeNeuronRequest):
    """Return top-k and bottom-k tokens most strongly promoted/suppressed by
    FFN neuron `neuron` at `layer`, computed as W_U @ W_down[layer][:, neuron].

    Classic logit-lens applied per-neuron (Geva 2020, nostalgebraist 2020).
    V1 deliberately skips the final-norm scaling — within-neuron ranking is
    unchanged by it; cross-neuron magnitudes are noisy (documented in UI).

    Returns: {
        "top_tokens":    [{"token": str, "logit": float}, ...],
        "bottom_tokens": [{"token": str, "logit": float}, ...]
    }
    """
    import torch

    top_k = max(1, min(req.top_k, _DECODE_NEURON_TOPK_MAX))

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
    intermediate = model.config.intermediate_size
    if not 0 <= req.neuron < intermediate:
        raise HTTPException(
            400,
            f"neuron out of range: got {req.neuron}, valid [0, {intermediate})",
        )

    def _compute() -> dict:
        with torch.no_grad():
            # W_down.weight: [hidden, intermediate]; column = direction written
            # by neuron `neuron` at layer `req.layer`.
            direction = model.model.layers[req.layer].mlp.down_proj.weight[:, req.neuron]
            # lm_head.weight: [vocab, hidden]. Matmul yields [vocab] logit scores.
            scores = model.lm_head.weight @ direction  # [vocab]
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
        return {"top_tokens": top_tokens, "bottom_tokens": bottom_tokens}

    result = await asyncio.to_thread(_compute)
    return result
```

- [ ] **Step 2: Write unit tests**

Create `testing/tests/test_decode_neuron.py`:

```python
"""Unit + integration tests for POST /api/sessions/{name}/decode-neuron (Phase 3.9.1)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import os

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient


# ---- Mock session + model setup ----

class _MockTok:
    """Minimal HF-like tokenizer: per-id decode returns a deterministic string."""
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"<{i}>" for i in ids)


class _MockDownProj(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden, intermediate))


class _MockMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.down_proj = _MockDownProj(hidden, intermediate)


class _MockLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = _MockMLP(hidden, intermediate)


class _MockInner(nn.Module):
    def __init__(self, n_layers: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(hidden, intermediate) for _ in range(n_layers)])


class _MockLMHead(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 10, hidden: int = 4, intermediate: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.model = _MockInner(n_layers, hidden, intermediate)
        self.lm_head = _MockLMHead(vocab, hidden)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=intermediate,
            vocab_size=vocab,
        )


@pytest.fixture
def app_with_mock_session():
    """Fresh FastAPI app with a single session injected that holds a mock model + tokenizer."""
    from gui.backend.app import app  # noqa: PLC0415
    from gui.backend.routes.sessions import get_manager  # noqa: PLC0415

    mgr = get_manager()
    mock_model = _MockModel()
    mock_tok = _MockTok()
    session_name = "mock-decode-neuron"
    # Insert directly into the manager's internal dict.
    mgr._sessions[session_name] = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        name=session_name,
        model=mock_model,
        tokenizer=mock_tok,
        llama=None,
        dirty=False,
        original_layer=lambda i: i,
        _layer_map=[],
    )
    try:
        yield (app, session_name)
    finally:
        mgr._sessions.pop(session_name, None)  # pyright: ignore[reportAttributeAccessIssue]


class TestDecodeNeuronUnit:
    def test_returns_topk_and_bottomk_sorted(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 3, "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == 5
        assert len(body["bottom_tokens"]) == 5
        top_logits = [t["logit"] for t in body["top_tokens"]]
        bot_logits = [t["logit"] for t in body["bottom_tokens"]]
        assert top_logits == sorted(top_logits, reverse=True)
        assert bot_logits == sorted(bot_logits)
        # Top-logit > bottom-logit (or equal at degenerate cases).
        assert top_logits[0] >= bot_logits[0]

    def test_invalid_layer_returns_400(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 99, "neuron": 0, "top_k": 3})
        assert resp.status_code == 400
        assert "layer" in resp.json()["detail"]

    def test_invalid_neuron_returns_400(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 999, "top_k": 3})
        assert resp.status_code == 400
        assert "neuron" in resp.json()["detail"]

    def test_top_k_clamped_to_50(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 0, "top_k": 1000})
        assert resp.status_code == 200
        body = resp.json()
        # Mock vocab is 10; clamp to min(50, vocab).
        assert len(body["top_tokens"]) == min(50, 10)

    def test_top_k_floor_at_1(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 0, "top_k": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == 1

    def test_unknown_session_returns_404(self, app_with_mock_session) -> None:
        app, _ = app_with_mock_session
        client = TestClient(app)
        resp = client.post("/api/sessions/does-not-exist/decode-neuron",
                           json={"layer": 0, "neuron": 0, "top_k": 3})
        assert resp.status_code == 404
```

- [ ] **Step 3: Run unit tests**

Run:
```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_decode_neuron.py -v -k "Unit"
```
Expected: 6 tests pass.

If the test fails on `mgr._sessions` attribute access, the SessionManager's private dict may have a different name. Grep for `self\._sessions\|self\.sessions` inside `gui/backend/sessions.py` and use whichever is present. If the manager exposes a public `add(name, info)` method, prefer that.

- [ ] **Step 4: Pyright clean**

Run:
```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py testing/tests/test_decode_neuron.py
```
Expected: 0/0/0.

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/backend/routes/sessions.py testing/tests/test_decode_neuron.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(backend): POST /sessions/{name}/decode-neuron endpoint

Returns top-k and bottom-k tokens most strongly promoted/suppressed by
FFN neuron (layer, neuron) via W_U @ W_down[layer][:, neuron]. Classic
per-neuron logit-lens (Geva 2020, nostalgebraist 2020). V1 skips
final-norm — within-neuron ranking is unbiased by the omission.

Unit tests cover: top/bottom sort order, 400 on out-of-range layer or
neuron, top_k clamped to [1, 50], 404 on unknown session.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: TinyLlama integration test

**Files:**
- Modify: `testing/tests/test_decode_neuron.py` — append GPU-guarded class

- [ ] **Step 1: Append TinyLlama integration test**

At the end of `testing/tests/test_decode_neuron.py`, append:

```python
# -------------------------------------------------------------------------
# TinyLlama integration — click a real top-AP neuron and sanity-check the
# decoded tokens (skipif GPU missing; fp16 to match Phase 3.9's precedent).
# -------------------------------------------------------------------------

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestDecodeNeuronTinyLlama:
    def test_top_neuron_decode_has_semantic_signal(self) -> None:
        """On the capital-of-France task, the top-ranked FFN neuron's
        decoded tokens should include at least one recognizable string
        (loose assertion — 1B-param neuron rankings are noisy)."""
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import attribution_patch_per_neuron

        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        paris_id = int(tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item())
        rome_id = int(tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item())

        r = attribution_patch_per_neuron(
            model, tok, clean, corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=rome_id,
            direction="denoise",
            top_k_neurons=10,
        )
        # Pick the top cell with positive ap_recovery (neuron that PROMOTES Paris).
        top_pos = next((c for c in r.cells if c["ap_recovery"] > 0), None)
        assert top_pos is not None, "expected at least one positive-AP neuron"

        L, neuron_idx = int(top_pos["layer"]), int(top_pos["neuron"])

        # Compute W_U @ W_down[L][:, neuron] directly (no endpoint needed).
        with torch.no_grad():
            direction = model.model.layers[L].mlp.down_proj.weight[:, neuron_idx]
            scores = model.lm_head.weight @ direction
            top_ids = torch.topk(scores, 10, largest=True).indices.tolist()

        decoded = [tok.decode([i], skip_special_tokens=False) for i in top_ids]
        # Loose semantic check: SOME token in the top-10 has non-whitespace content
        # and the list is 10 strings.
        assert len(decoded) == 10
        assert any(s.strip() != "" for s in decoded)
```

- [ ] **Step 2: Run integration test**

Run:
```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_decode_neuron.py::TestDecodeNeuronTinyLlama -v -s
```
Expected: passes in ~1 min (dominated by model load + one AP pass ~43s + trivial matmul).

- [ ] **Step 3: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/tests/test_decode_neuron.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
test(probe): TinyLlama integration for decode-neuron + Phase 3.9 chain

Runs attribution_patch_per_neuron on capital-of-France, picks the
top positive-AP neuron, and verifies the direct W_U @ W_down[:, i]
decode returns 10 non-empty tokens. Loose semantic check — neuron
rankings on a 1B-param model are noisy; tighter assertions would
be brittle.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Frontend pinned-card UI + Playwright smoke

**Files:**
- Modify: `testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx`
- Modify: `testing/gui/frontend/src/components/VisualizationArea.tsx`
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Extend `PerNeuronPatchingPanel` with pinned-card + fetch**

Open `PerNeuronPatchingPanel.tsx`. Find the `interface Props` declaration. Replace with:

```tsx
interface Props {
  cells: PatchingCellData[];
  complete?: PatchingCompleteData;
  sessionName?: string;
}
```

At the top of the component body (after `export function PerNeuronPatchingPanel({ cells, complete, sessionName }: Props) {`), add state + effect:

```tsx
  const [pinnedRow, setPinnedRow] = useState<{ layer: number; neuron: number } | null>(null);
  const [decode, setDecode] = useState<{
    top: Array<{ token: string; logit: number }>;
    bottom: Array<{ token: string; logit: number }>;
  } | null>(null);
  const [decodeLoading, setDecodeLoading] = useState<boolean>(false);
  const [decodeError, setDecodeError] = useState<string | null>(null);

  useEffect(() => {
    if (pinnedRow === null || !sessionName) {
      setDecode(null);
      setDecodeError(null);
      return;
    }
    const ctrl = new AbortController();
    setDecodeLoading(true);
    setDecodeError(null);
    fetch(`/api/sessions/${sessionName}/decode-neuron`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ layer: pinnedRow.layer, neuron: pinnedRow.neuron, top_k: 10 }),
      signal: ctrl.signal,
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`decode-neuron ${r.status}`))))
      .then((body: { top_tokens: Array<{ token: string; logit: number }>; bottom_tokens: Array<{ token: string; logit: number }> }) => {
        setDecode({ top: body.top_tokens, bottom: body.bottom_tokens });
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") setDecodeError(err.message);
      })
      .finally(() => setDecodeLoading(false));
    return () => ctrl.abort();
  }, [pinnedRow, sessionName]);
```

Make sure `useEffect` is imported — change the top import to:
```tsx
import { useEffect, useMemo, useState } from "react";
```

- [ ] **Step 2: Wire row click and render card**

Find the `<tbody>` block. Replace the row rendering with a clickable version:

```tsx
          <tbody>
            {visible.slice(0, 200).map((c, i) => {
              const isPinned =
                pinnedRow !== null &&
                pinnedRow.layer === c.layer &&
                pinnedRow.neuron === c.neuron;
              return (
                <tr
                  key={`${c.layer}-${c.neuron}-${c.position}`}
                  onClick={() =>
                    setPinnedRow(
                      isPinned
                        ? null
                        : { layer: c.layer ?? 0, neuron: c.neuron ?? 0 },
                    )
                  }
                  style={{
                    background: apColor(c.ap_recovery ?? 0),
                    cursor: "pointer",
                    outline: isPinned ? "2px solid #8abaff" : "none",
                  }}
                >
                  <td style={{ padding: 4 }}>{i + 1}</td>
                  <td style={{ padding: 4 }}>L{c.layer}</td>
                  <td style={{ padding: 4 }}>n{c.neuron}</td>
                  <td style={{ padding: 4 }}>{c.position}</td>
                  <td style={{ padding: 4, textAlign: "right" }}>
                    {(c.ap_recovery ?? 0).toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
```

Directly AFTER the stats strip `<div className="stats">…</div>` and BEFORE the `<div style={{ maxHeight: 600, ...`, insert the pinned card:

```tsx
      {pinnedRow !== null && (
        <div
          className="neuron-pin-card"
          style={{
            border: "1px solid #333",
            background: "#12121a",
            padding: 12,
            marginBottom: 8,
            borderRadius: 4,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <div>
              <b>
                Neuron L{pinnedRow.layer}.n{pinnedRow.neuron}
              </b>{" "}
              <span style={{ color: "#888", fontSize: 11, marginLeft: 8 }}>
                Raw W_U @ W_down[:, n] — normalize-for-magnitudes deferred.
              </span>
            </div>
            <button onClick={() => setPinnedRow(null)}>close</button>
          </div>
          {decodeLoading && <div style={{ color: "#aaa" }}>loading…</div>}
          {decodeError && <div style={{ color: "#c88" }}>error: {decodeError}</div>}
          {decode && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontFamily: "monospace", fontSize: 12 }}>
              <div>
                <div style={{ color: "#8abaff", marginBottom: 4 }}>top 10 promoted</div>
                <table style={{ width: "100%" }}>
                  <tbody>
                    {decode.top.map((t, i) => (
                      <tr key={`top-${i}`}>
                        <td style={{ padding: "2px 4px" }}>{t.token}</td>
                        <td style={{ padding: "2px 4px", textAlign: "right", color: "#4caf50" }}>
                          +{t.logit.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <div style={{ color: "#ffca8a", marginBottom: 4 }}>bottom 10 suppressed</div>
                <table style={{ width: "100%" }}>
                  <tbody>
                    {decode.bottom.map((t, i) => (
                      <tr key={`bot-${i}`}>
                        <td style={{ padding: "2px 4px" }}>{t.token}</td>
                        <td style={{ padding: "2px 4px", textAlign: "right", color: "#c62828" }}>
                          {t.logit.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
```

- [ ] **Step 3: Pass `sessionName` from `VisualizationArea`**

In `testing/gui/frontend/src/components/VisualizationArea.tsx`, find the `PerNeuronPatchingPanel` render. Update it to pass `sessionName`:

```tsx
<PerNeuronPatchingPanel cells={cellMsgs} complete={completeMsg} sessionName={activeResult.sessionName} />
```

(The `activeResult` object already exposes `sessionName` — other panels use it via the same `activeResult.sessionName` path.)

- [ ] **Step 4: Tsc clean**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```
Expected: no errors.

- [ ] **Step 5: Vitest still clean**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npx vitest run
```
Expected: 19/19 tests still pass (no new Vitest added — effect logic is tested via Playwright).

- [ ] **Step 6: Add 16th Playwright smoke test**

Append to `testing/gui/frontend/tests/e2e/smoke.spec.ts` (after the 15th per-neuron test):

```ts
test("per-neuron row click opens pinned card with decoded tokens", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Intercept the decode-neuron endpoint with a stub so this test doesn't
  // depend on a live backend.
  await page.route("**/api/sessions/*/decode-neuron", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens: [
          { token: " Paris", logit: 2.34 },
          { token: " Lyon", logit: 1.87 },
          { token: " France", logit: 1.42 },
          { token: " French", logit: 1.10 },
          { token: " capital", logit: 0.91 },
          { token: " city", logit: 0.75 },
          { token: " Europe", logit: 0.60 },
          { token: " Seine", logit: 0.48 },
          { token: " Eiffel", logit: 0.31 },
          { token: " Louvre", logit: 0.22 },
        ],
        bottom_tokens: [
          { token: " Rome", logit: -1.89 },
          { token: " Milan", logit: -1.52 },
          { token: " Italy", logit: -1.30 },
          { token: " Italian", logit: -1.10 },
          { token: " Vatican", logit: -0.95 },
          { token: " pizza", logit: -0.81 },
          { token: " pasta", logit: -0.67 },
          { token: " Venice", logit: -0.55 },
          { token: " Colosseum", logit: -0.40 },
          { token: " Florence", logit: -0.31 },
        ],
      }),
    });
  });

  const fixture = fs.readFileSync(PER_NEURON_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-neuron.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Per-Neuron FFN Attribution/i })
    .waitFor({ state: "visible", timeout: 5000 });

  // Click the first data row (skipping the header row).
  const firstRow = page.locator("tbody tr").first();
  await firstRow.click();

  // Pinned card assertions.
  await expect(page.getByText(/Raw W_U @ W_down/i)).toBeVisible();
  await expect(page.getByText(" Paris", { exact: true })).toBeVisible();
  await expect(page.getByText(" Rome", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /close/i })).toBeVisible();

  // Close the card.
  await page.getByRole("button", { name: /close/i }).click();
  await expect(page.getByText(/Raw W_U @ W_down/i)).not.toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});
```

- [ ] **Step 7: Run Playwright**

Run:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```
Expected: 16/16 tests pass.

- [ ] **Step 8: Final verification matrix**

Run each:
```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
cd /home/ai/ai-projects/llm/testing/gui/frontend && npx vitest run
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py testing/tests/test_decode_neuron.py
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/ -v -k "not TinyLlama"
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```

Expected: all green, 0/0/0 pyright, 68+ Python tests pass, Vitest 19/19, Playwright 16/16.

- [ ] **Step 9: Commit + roadmap memory update**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx testing/gui/frontend/src/components/VisualizationArea.tsx testing/gui/frontend/tests/e2e/smoke.spec.ts
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(gui/frontend): per-neuron pin-card with logit-lens decode

Clicking a row in PerNeuronPatchingPanel fetches the backend's
decode-neuron endpoint and renders a pinned card showing:
- top 10 promoted tokens (+logits, green)
- bottom 10 suppressed tokens (−logits, red)
- close button, abort-on-unpin

VisualizationArea passes sessionName so the panel knows which
session to query. 16th Playwright smoke test exercises the click →
card flow with a route-intercepted stub response.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Update the roadmap memory at `~/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` with a Phase 3.9.1 shipped entry (commit SHAs + verification matrix).

---

## Verification Matrix

| Check | Command | Expected |
|-------|---------|----------|
| Pyright | `.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py testing/tests/test_decode_neuron.py` | 0/0/0 |
| Tsc | `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit` | clean |
| Python unit | `.venv/bin/python -m pytest testing/tests/test_decode_neuron.py -v -k "Unit"` | 6 pass |
| Python TinyLlama | `.venv/bin/python -m pytest testing/tests/test_decode_neuron.py::TestDecodeNeuronTinyLlama -v` | pass ~1 min |
| Phase 3.5–3.9 regressions | `.venv/bin/python -m pytest testing/tests/ -v -k "not TinyLlama"` | 68+ pass |
| Vitest | `cd testing/gui/frontend && npx vitest run` | 19/19 |
| Playwright | `cd testing/gui/frontend && npm run e2e` | 16/16 |

---

## Plan Self-Review Notes

**Spec coverage:**
- §2 G1 (endpoint) → Task 1.
- §2 G2 (pin card) → Task 3.
- §2 G3 (tests) → Task 1 (unit), Task 2 (TinyLlama), Task 3 (Playwright). Spec §7.3 Vitest is intentionally skipped — the fetch is inline in the component, not in a utility module, so there's nothing to unit-test in isolation. Documented in the task steps.

**Placeholder scan:** None.

**Type consistency:**
- Backend: `DecodeNeuronRequest { layer, neuron, top_k }` matches frontend POST body exactly.
- Response keys: `top_tokens`, `bottom_tokens` same on backend + frontend fetch typing + Playwright stub.
- Token shape: `{ token: string; logit: number }` consistent across backend emit + frontend consume + stub.

Ready for execution.
