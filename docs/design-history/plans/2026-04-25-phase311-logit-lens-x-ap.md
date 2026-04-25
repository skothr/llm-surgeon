# Phase 3.11 Implementation Plan — Logit Lens × AP

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`. Steps use checkbox (`- [ ]`) syntax. **Sandbox-denial pattern is well-established** — verification + commits run by parent with `dangerouslyDisableSandbox: true`.

**Goal:** Wire a `decode-residual` backend endpoint into all five AP pin cards.

**Architecture:** Backend reuses `_capture_residual_stream` + `_project_to_logits`. Frontend extracts a shared `ResidualDecodeBlock` component used by 5 panels.

**Tech stack:** FastAPI / Pydantic / Torch (backend); React / TypeScript / Zustand (frontend); pytest + Vitest + Playwright.

---

## Task 1 — Backend endpoint + tests

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py` (append new endpoint at end of decode-* block, just after `decode_head`)
- Create: `testing/tests/test_decode_residual.py`

- [ ] **Step 1: Append the endpoint to `sessions.py`**

Add after the `decode_head` function:

```python
class DecodeResidualRequest(BaseModel):
    prompt: str
    layer: int
    sublayer: str
    position: int
    top_k: int = 10


_DECODE_RESIDUAL_TOPK_MAX = 50
_DECODE_RESIDUAL_VALID_SUBLAYERS = {"attn", "ffn"}


@router.post("/sessions/{name}/decode-residual")
async def decode_residual(name: str, req: DecodeResidualRequest):
    """Project the residual stream at (layer, sublayer, position) through
    the final norm + lm_head, return top/bottom-k tokens.

    Standard logit lens (nostalgebraist 2020) applied to a single
    residual-stream slot — used by AP pin cards to surface what the
    model "thinks" it's predicting at the patched point.
    """
    import torch
    from llm_surgeon.probe import _capture_residual_stream, _project_to_logits

    if req.sublayer not in _DECODE_RESIDUAL_VALID_SUBLAYERS:
        raise HTTPException(
            400,
            f"sublayer must be one of {sorted(_DECODE_RESIDUAL_VALID_SUBLAYERS)}; got {req.sublayer!r}",
        )

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
    tok = info.tokenizer
    num_layers = len(model.model.layers)
    if not 0 <= req.layer < num_layers:
        raise HTTPException(
            400,
            f"layer out of range: got {req.layer}, valid [0, {num_layers})",
        )

    def _compute() -> dict:
        captured, prompt_tokens = _capture_residual_stream(
            model, tok, req.prompt, sublayers=(req.sublayer,),
        )
        seq_len = len(prompt_tokens)
        if not 0 <= req.position < seq_len:
            raise HTTPException(
                400,
                f"position out of range: got {req.position}, valid [0, {seq_len})",
            )
        hidden = captured[(req.layer, req.sublayer)]
        with torch.no_grad():
            logits = _project_to_logits(model, hidden)
        scores = logits[req.position]
        vocab_size = int(scores.shape[0])
        k = max(1, min(req.top_k, _DECODE_RESIDUAL_TOPK_MAX, vocab_size))
        top_vals, top_ids = torch.topk(scores, k, largest=True)
        bot_vals, bot_ids = torch.topk(scores, k, largest=False)
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
            "prompt_tokens": prompt_tokens,
        }

    return await asyncio.to_thread(_compute)
```

- [ ] **Step 2: Create `tests/test_decode_residual.py`**

Use existing `test_decode_neuron.py` and `test_decode_head.py` as the template — same `MockModel` / `MockTokenizer` shape, same TestClient setup. Tests:

```python
"""Tests for POST /api/sessions/{name}/decode-residual.

Mirrors test_decode_neuron.py / test_decode_head.py — a tiny mock model
+ tokenizer, then routing-and-validation tests, plus one TinyLlama
parity test against probe.logit_lens at the last layer's "ffn" point.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import pytest
from fastapi.testclient import TestClient

from gui.backend.app import create_app
from gui.backend.session_manager import SessionInfo, get_manager


# ----- Mock model/tokenizer (identical shape to test_decode_neuron.py) -----

class _MockTokenizer:
    def __init__(self, vocab=("<pad>", "the", "cat", "sat", "on", "mat")):
        self.vocab = list(vocab)
        self._t2i = {t: i for i, t in enumerate(self.vocab)}

    def __call__(self, text, return_tensors="pt", **_):
        ids = [self._t2i.get(t, 0) for t in text.split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [self.vocab[i] for i in ids]

    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self.vocab[i] for i in ids)


class _MockLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.self_attn = nn.Linear(d, d)
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _MockInner(nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_MockLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)


class _MockModel(nn.Module):
    def __init__(self, vocab=6, d=8, n_layers=3):
        super().__init__()
        self.model = _MockInner(vocab, d, n_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = type("Cfg", (), {"hidden_size": d, "num_attention_heads": 1, "intermediate_size": d})()

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


# ----- Fixtures -----

@pytest.fixture
def session_with_model():
    name = "test-decode-residual"
    mgr = get_manager()
    if name in mgr._sessions:
        del mgr._sessions[name]
    info = SessionInfo(name=name, model_id="mock", mode="bf16", num_layers=3, device="cpu")
    info.model = _MockModel()
    info.tokenizer = _MockTokenizer()
    mgr._sessions[name] = info
    yield name
    del mgr._sessions[name]


@pytest.fixture
def client():
    return TestClient(create_app())


# ----- Validation tests -----

def test_404_missing_session(client):
    r = client.post("/api/sessions/missing/decode-residual", json={
        "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 0,
    })
    assert r.status_code == 404


def test_500_no_model(client, session_with_model):
    mgr = get_manager()
    mgr._sessions[session_with_model].model = None
    r = client.post(f"/api/sessions/{session_with_model}/decode-residual", json={
        "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 0,
    })
    assert r.status_code == 500


def test_400_invalid_sublayer(client, session_with_model):
    r = client.post(f"/api/sessions/{session_with_model}/decode-residual", json={
        "prompt": "the cat sat", "layer": 0, "sublayer": "embed", "position": 0,
    })
    assert r.status_code == 400
    assert "sublayer" in r.json()["detail"]


def test_400_layer_out_of_range(client, session_with_model):
    r = client.post(f"/api/sessions/{session_with_model}/decode-residual", json={
        "prompt": "the cat sat", "layer": 99, "sublayer": "ffn", "position": 0,
    })
    assert r.status_code == 400
    assert "layer" in r.json()["detail"]


def test_400_position_out_of_range(client, session_with_model):
    r = client.post(f"/api/sessions/{session_with_model}/decode-residual", json={
        "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 99,
    })
    assert r.status_code == 400
    assert "position" in r.json()["detail"]


def test_top_k_clamped(client, session_with_model):
    r = client.post(f"/api/sessions/{session_with_model}/decode-residual", json={
        "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 0, "top_k": 999,
    })
    assert r.status_code == 200
    body = r.json()
    # mock vocab is 6; clamp = min(999, 50, 6) = 6
    assert len(body["top_tokens"]) == 6
    assert len(body["bottom_tokens"]) == 6


def test_response_shape(client, session_with_model):
    r = client.post(f"/api/sessions/{session_with_model}/decode-residual", json={
        "prompt": "the cat sat", "layer": 1, "sublayer": "attn", "position": 1, "top_k": 3,
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"top_tokens", "bottom_tokens", "prompt_tokens"}
    assert len(body["top_tokens"]) == 3
    assert len(body["bottom_tokens"]) == 3
    # top descending, bottom ascending
    top_logits = [t["logit"] for t in body["top_tokens"]]
    bot_logits = [t["logit"] for t in body["bottom_tokens"]]
    assert top_logits == sorted(top_logits, reverse=True)
    assert bot_logits == sorted(bot_logits)
    assert body["prompt_tokens"] == ["the", "cat", "sat"]


# ----- TinyLlama parity test -----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_decode_residual_tinyllama_matches_logit_lens():
    """At the last layer's "ffn" point, decode-residual's top-1 token must
    equal probe.logit_lens()'s top-1 token at the same (layer, sublayer,
    position) — both go through the same final-norm + lm_head."""
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_surgeon.probe import logit_lens
    from gui.backend.session_manager import SessionInfo, get_manager
    from fastapi.testclient import TestClient
    from gui.backend.app import create_app

    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/ai-projects/llm/testing/.cache/models"))
    repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache_root)
    model = AutoModelForCausalLM.from_pretrained(
        repo, cache_dir=cache_root, torch_dtype=torch.float16
    ).cuda().eval()
    prompt = "The Eiffel Tower is in"

    mgr = get_manager()
    name = "tinyllama-decode-residual-test"
    if name in mgr._sessions:
        del mgr._sessions[name]
    info = SessionInfo(
        name=name, model_id=repo, mode="fp16",
        num_layers=len(model.model.layers), device="cuda",
    )
    info.model = model
    info.tokenizer = tok
    mgr._sessions[name] = info

    try:
        # Reference: probe.logit_lens at last layer ffn
        last_layer = len(model.model.layers) - 1
        result = logit_lens(model, tok, prompt, top_k=1)
        # Find the prediction at (last_layer, ffn, last_pos)
        seq_len = len(result.prompt_tokens)
        ref_pred = next(p for p in result.predictions
                        if p["layer"] == last_layer
                        and p["sublayer"] == "ffn"
                        and p["position"] == seq_len - 1)
        ref_top1 = ref_pred["top_k"][0]["token"]

        # Endpoint call
        client = TestClient(create_app())
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": prompt, "layer": last_layer, "sublayer": "ffn",
            "position": seq_len - 1, "top_k": 1,
        })
        assert r.status_code == 200, r.text
        endpoint_top1 = r.json()["top_tokens"][0]["token"]
        assert endpoint_top1 == ref_top1, f"endpoint {endpoint_top1!r} != logit_lens {ref_top1!r}"
    finally:
        del mgr._sessions[name]
        del model
        torch.cuda.empty_cache()
```

- [ ] **Step 3: Run unit tests (parent, sandbox-disabled)**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_decode_residual.py -v -k "not tinyllama"
```

Expected: 7/7 pass.

- [ ] **Step 4: Run pyright (parent, sandbox-disabled)**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright gui/backend/routes/sessions.py tests/test_decode_residual.py
```

Expected: 0 errors / 0 warnings / 0 informations.

- [ ] **Step 5: Run TinyLlama integration (parent, sandbox-disabled, GPU)**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_decode_residual.py -v -k "tinyllama"
```

Expected: 1/1 pass in ~30-60s.

- [ ] **Step 6: Commit**

```bash
git add testing/gui/backend/routes/sessions.py testing/tests/test_decode_residual.py
git commit -m "feat(probe): /decode-residual endpoint for AP pin-card logit lens

Standard logit lens (final-norm + lm_head) applied to a single
residual-stream point on demand. Reuses _capture_residual_stream
+ _project_to_logits. Phase 3.11 Task 1 (backend)."
```

---

## Task 2 — Shared frontend hook + ResidualDecodeBlock component

**Files:**
- Create: `testing/gui/frontend/src/utils/useResidualDecode.ts`
- Create: `testing/gui/frontend/src/components/visualizations/ResidualDecodeBlock.tsx`
- Create: `testing/gui/frontend/tests/unit/useResidualDecode.test.ts` (Vitest)

- [ ] **Step 1: Create `utils/useResidualDecode.ts`**

```typescript
import { useEffect, useState } from "react";

export type ResidualDecodeToken = { token: string; logit: number };
export type ResidualDecodeResponse = {
  top_tokens: ResidualDecodeToken[];
  bottom_tokens: ResidualDecodeToken[];
  prompt_tokens: string[];
};

export function useResidualDecode(
  sessionName: string | undefined,
  prompt: string | undefined,
  layer: number | undefined,
  sublayer: "attn" | "ffn" | undefined,
  position: number | undefined,
  topK: number = 10,
): { data: ResidualDecodeResponse | null; error: string | null; loading: boolean } {
  const [data, setData] = useState<ResidualDecodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (
      sessionName === undefined
      || prompt === undefined
      || layer === undefined
      || sublayer === undefined
      || position === undefined
    ) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const ctl = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);
    fetch(`/api/sessions/${encodeURIComponent(sessionName)}/decode-residual`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, layer, sublayer, position, top_k: topK }),
      signal: ctl.signal,
    })
      .then(async (r) => {
        if (!r.ok) {
          const text = await r.text();
          throw new Error(text || `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((j: ResidualDecodeResponse) => {
        if (!ctl.signal.aborted) {
          setData(j);
          setLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => ctl.abort();
  }, [sessionName, prompt, layer, sublayer, position, topK]);

  return { data, error, loading };
}
```

- [ ] **Step 2: Create `components/visualizations/ResidualDecodeBlock.tsx`**

```typescript
import React from "react";
import { useResidualDecode } from "../../utils/useResidualDecode";

type Props = {
  sessionName: string;
  prompt: string;
  layer: number;
  sublayer: "attn" | "ffn";
  position: number;
};

export function ResidualDecodeBlock({
  sessionName, prompt, layer, sublayer, position,
}: Props) {
  const { data, error, loading } = useResidualDecode(
    sessionName, prompt, layer, sublayer, position, 10,
  );

  return (
    <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid #2a2a3a" }}>
      <div style={{ fontSize: 12, color: "#a0a0c0", marginBottom: 6 }}>
        — Logit lens at (L{layer}, {sublayer}, pos {position})
      </div>
      {loading && (
        <div style={{ fontSize: 12, color: "#888" }}>decoding…</div>
      )}
      {error && (
        <div style={{ fontSize: 12, color: "#e88" }}>error: {error}</div>
      )}
      {data && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <div style={{ fontSize: 11, color: "#7c7", fontWeight: "bold" }}>Promoted</div>
            {data.top_tokens.map((t, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "monospace", color: "#cfc" }}>
                <span>{JSON.stringify(t.token)}</span>
                <span>+{t.logit.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#c77", fontWeight: "bold" }}>Suppressed</div>
            {data.bottom_tokens.map((t, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "monospace", color: "#fcc" }}>
                <span>{JSON.stringify(t.token)}</span>
                <span>{t.logit.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Create `tests/unit/useResidualDecode.test.ts`** (Vitest)

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useResidualDecode } from "../../src/utils/useResidualDecode";

describe("useResidualDecode", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("is inert when args are undefined", () => {
    const { result } = renderHook(() => useResidualDecode(undefined, undefined, undefined, undefined, undefined));
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("fetches and returns decoded tokens on full args", async () => {
    const mock = { top_tokens: [{ token: "x", logit: 1.0 }], bottom_tokens: [], prompt_tokens: ["hi"] };
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mock,
    });
    const { result } = renderHook(() =>
      useResidualDecode("s", "hi", 0, "ffn", 0, 5)
    );
    await waitFor(() => expect(result.current.data).toEqual(mock));
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it("aborts when args change before resolution", async () => {
    let resolveFn: (v: Response) => void = () => {};
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(() =>
      new Promise<Response>((resolve) => { resolveFn = resolve; })
    );
    const { result, rerender } = renderHook(
      ({ pos }: { pos: number }) =>
        useResidualDecode("s", "hi", 0, "ffn", pos),
      { initialProps: { pos: 0 } },
    );
    expect(result.current.loading).toBe(true);
    rerender({ pos: 1 });
    // Resolve the first fetch — should be ignored
    const mockBody = { top_tokens: [{ token: "stale", logit: 0 }], bottom_tokens: [], prompt_tokens: [] };
    resolveFn({ ok: true, json: async () => mockBody } as Response);
    await new Promise((r) => setTimeout(r, 10));
    // result should NOT contain "stale"
    expect(result.current.data === null
      || !result.current.data.top_tokens.some(t => t.token === "stale")).toBe(true);
  });
});
```

**Note:** if `@testing-library/react` is not yet installed, install via `cd testing/gui/frontend && npm install --save-dev @testing-library/react`. Check `package.json` first; it may already be a dep from prior phases.

- [ ] **Step 4: Run Vitest (parent, sandbox-disabled)**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/vitest run
```

Expected: 22/22 pass (19 prior + 3 new). If `@testing-library/react` is missing, add it first.

- [ ] **Step 5: Run tsc**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add testing/gui/frontend/src/utils/useResidualDecode.ts \
        testing/gui/frontend/src/components/visualizations/ResidualDecodeBlock.tsx \
        testing/gui/frontend/tests/unit/useResidualDecode.test.ts \
        testing/gui/frontend/package.json testing/gui/frontend/package-lock.json
git commit -m "feat(gui): ResidualDecodeBlock + useResidualDecode hook

Shared component used by all 5 AP pin cards in the next task.
Fetches POST /decode-residual with AbortController lifecycle.
Phase 3.11 Task 2 (frontend hook + component)."
```

---

## Task 3 — Wire into all 5 AP pin cards

**Files:**
- Modify: `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx`
- Modify: `testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx`
- Modify: `testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx`
- Modify: `testing/gui/frontend/src/components/visualizations/EdgeAttributionPanel.tsx`
- Modify: `testing/gui/frontend/src/components/visualizations/CircuitPanel.tsx`

**Strategy:** Each panel's pin card already shows cell info. Inject `<ResidualDecodeBlock>` (or a caption fallback for embed-writers) at the END of the pin card, after existing content but before the close button.

For each panel, locate the pin-card render block (search for `pinned`, `pinnedRow`, or the text content of pin cards already there), add the import, derive `(layer, sublayer, position)` from the cell, and render either `<ResidualDecodeBlock>` or a `<div>caption</div>`.

**Per-panel cell→args mapping** (canonical reference):

| Panel | Source field for layer | Source field for position | Sublayer derivation |
|-------|---|---|---|
| ActivationPatchingHeatmap | `cell.layer` | `cell.position` | `cell.sublayer` (already "attn"/"ffn") |
| PerHeadPatchingHeatmap | `cell.layer` | `cell.position` | `cell.unit.startsWith("attn") ? "attn" : "ffn"` |
| PerNeuronPatchingPanel | `cell.layer` | `cell.position` | constant `"ffn"` |
| EdgeAttributionPanel | `cell.writer_layer` | `cell.position` | `cell.writer_unit.startsWith("attn") ? "attn" : cell.writer_unit === "ffn" ? "ffn" : null` |
| CircuitPanel | `cell.writer_layer` | `cell.position` | same as Edge |

**For null-sublayer cases (embed writer):** render `<div style={{ marginTop: 12, color: "#888", fontSize: 12 }}>Residual decode at writer 'embed' is not supported in V1.</div>`

For `prompt`: each panel has access to the experiment's prompt via the result/baselines; use `result.prompt` (panels already use this — search to confirm).

For `sessionName`: each panel already receives this prop or has access via `result.sessionName` (verify by reading existing decode-neuron / decode-head wiring in PerNeuronPatchingPanel and PerHeadPatchingHeatmap).

- [ ] **Step 1: ActivationPatchingHeatmap** — add import, render at end of pin card.
- [ ] **Step 2: PerHeadPatchingHeatmap** — same; map unit→sublayer.
- [ ] **Step 3: PerNeuronPatchingPanel** — same; sublayer always "ffn".
- [ ] **Step 4: EdgeAttributionPanel** — same; null-sublayer fallback for embed.
- [ ] **Step 5: CircuitPanel** — same as Edge.

- [ ] **Step 6: Run tsc, vitest**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit && ./node_modules/.bin/vitest run
```

Both clean.

- [ ] **Step 7: Commit**

```bash
git add testing/gui/frontend/src/components/visualizations/{ActivationPatchingHeatmap,PerHeadPatchingHeatmap,PerNeuronPatchingPanel,EdgeAttributionPanel,CircuitPanel}.tsx
git commit -m "feat(gui): logit-lens decode block on every AP pin card

Each of the five AP visualization panels now renders ResidualDecodeBlock
when the user pins a cell. Edge/circuit embed-writers show a caption.
Phase 3.11 Task 3 (panel wiring)."
```

---

## Task 4 — Playwright smoke test

**Files:**
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Add a Playwright test that mocks `/decode-residual`**

Append to `smoke.spec.ts`:

```typescript
test("AP pin card shows residual decode (logit lens)", async ({ page }) => {
  await page.route("**/decode-residual", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens: [
          { token: "Paris", logit: 21.34 },
          { token: "France", logit: 18.0 },
        ],
        bottom_tokens: [
          { token: "Tokyo", logit: -8.12 },
        ],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
      }),
    });
  });

  // Reuse the existing exact-AP smoke fixture
  await page.goto("/");
  // ... import the activation-patching exact fixture (mirror existing AP test setup)
  // ... click an AP cell to pin it
  // ... assert the lens block renders

  await expect(page.getByText(/Logit lens at \(L/i)).toBeVisible();
  await expect(page.getByText(/Paris/)).toBeVisible();
});
```

**IMPORTANT:** the implementer must read the existing exact-AP smoke test to copy the fixture-import + cell-click pattern (do not reinvent). Use `getByText(/Logit lens at \(L/i)` for the heading match — escape `(` since it's a regex.

- [ ] **Step 2: Run Playwright (parent, sandbox-disabled)**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/playwright test
```

Expected: 19/19 pass.

- [ ] **Step 3: Commit**

```bash
git add testing/gui/frontend/tests/e2e/smoke.spec.ts
git commit -m "test(gui): Playwright smoke for AP residual lens decode

Mocks /decode-residual and asserts the Logit lens block renders
on the exact-AP pin card. Phase 3.11 Task 4."
```

---

## Self-review checklist (parent runs)

- [ ] Spec coverage: every Task 1-4 file mentioned exists and got committed.
- [ ] Pyright clean across `sessions.py` + `test_decode_residual.py`.
- [ ] Tsc clean.
- [ ] Vitest 20+/20+ pass.
- [ ] Playwright 19/19 pass.
- [ ] No probe.py changes (this phase is additive only).
- [ ] Roadmap memory updated with Phase 3.11 entry.
