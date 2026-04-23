# Phase 3.8 — ACDC-style Circuit Extraction

**Date:** 2026-04-23
**Status:** Spec (awaiting review)
**Depends on:** Phase 3.7 (edge attribution patching)
**References:**
- Conmy et al. 2023 — *Towards Automated Circuit Discovery for Mechanistic Interpretability*
- Syed et al. 2023 — *Attribution Patching Outperforms Automated Circuit Discovery* (arXiv 2310.10348)

## 1. Motivation

Phase 3.7 ships a ranked edge list: "here are the top-200 edges by |AP_edge|." That's ranked noise without a structural claim. Users still have to squint at a Sankey diagram and guess where the circuit is.

ACDC closes that gap. It turns a ranked edge list into a **single connected subgraph** — the minimal set of edges such that the metric still flows from inputs to the output through nothing but surviving edges. That subgraph is "the circuit."

This phase ships **cheap-ACDC** (Syed 2023): postprocess EAP's edge list with a threshold τ, then reverse-BFS from the logits reader to keep only edges that actually reach the metric. One forward + one backward pass total — same compute as Phase 3.7.

## 2. Goals

- **G1.** `probe.extract_circuit(...)` that returns every top-k edge annotated with `in_circuit: bool` at a user-specified τ.
- **G2.** WS route handles `mode="circuit"` on the existing `/activation-patching` endpoint.
- **G3.** `CircuitPanel.tsx` renders the circuit as a Sankey-style subgraph with a client-side threshold slider that re-runs BFS without a backend round-trip.
- **G4.** Shared `_compute_all_edges` helper so `edge_attribution_patch` (top-k) and `extract_circuit` (tau+connectivity) share compute.

## 3. Non-Goals

- **N1.** Full ACDC (iterative ablate-and-measure). Deferred to Phase 3.8.1 if demand materializes; the ranking produced by cheap-ACDC is reportedly within ~5% of full ACDC on standard circuit benchmarks (Syed 2023, Table 1).
- **N2.** Mean/zero ablation modes. Circuit is defined purely by "edge survives threshold + is reverse-reachable from logits."
- **N3.** Multi-prompt circuit averaging. Single clean/corrupted pair per run.
- **N4.** Automatic τ selection. User picks τ; a sensible default is provided (τ = 0.02 = 2% AP recovery).
- **N5.** Per-head QK/OV split within attention. Heads are atomic units, same granularity as Phase 3.7.

## 4. Math

### 4.1 Edge definition (recap from Phase 3.7)

An edge is `(writer, reader, position)`:
- Writer ∈ `{embed} ∪ {(L, "ffn") : L ∈ layers} ∪ {(L, "attn.hN") : L ∈ layers, N ∈ heads}`
- Reader ∈ `{(L, "attn_in") : L ∈ layers} ∪ {(L, "ffn_in") : L ∈ layers} ∪ {"logits"}`
- Edge valid iff writer's layer index is strictly less than reader's layer index, except `attn(L) → ffn_in(L)` which is same-layer and valid (FFN reads residual-stream-after-attn).
- Score: `AP_edge(w → r, pos) = (Δwrite_w[pos] · grad_reader_r[pos]).sum() / D`, where `D = Δ_clean − Δ_corrupted`.

### 4.2 Circuit definition

Given threshold τ ≥ 0:

1. **Filter**: retain edge e iff `|AP_edge(e)| ≥ τ`.
2. **Graph G_τ**: nodes = `(layer, unit, position)` tuples appearing as writer or reader in a retained edge. Directed edges = retained edges.
3. **Circuit**: set of edges that lie on some path from any writer to the logits reader in G_τ. Equivalently: edges whose reader is reverse-reachable from `logits` through retained edges **and** whose writer is reverse-reachable from `logits` via the same reader.
4. **Simplification**: since every retained edge goes `writer → reader`, and we want edges on paths to logits, an edge `(w, r)` is in the circuit iff `r` is reverse-reachable from `logits`. (Writer reachability then follows trivially — the edge itself reaches `r`, and `r` reaches logits, so `w → r → ... → logits`.)

**Algorithm**: reverse-BFS from `logits`, keep edges whose reader is visited.

```
visited ← {logits}
queue ← [logits]
while queue not empty:
    r ← queue.pop()
    for each retained edge (w, r') with r' = r:
        if w not in visited:
            visited.add(w)
            queue.push(w)
in_circuit(edge e=(w,r)) := r ∈ visited
```

Per-node flag `node_in_circuit(n)` = `n ∈ visited`.

### 4.3 Why reverse-BFS, not forward-BFS

Forward-BFS from embed marks every node the embed reaches, which at τ ≈ 0.02 is nearly every node — carries no signal.

Reverse-BFS from logits marks every node that causally contributes to the metric. This is the textbook circuit definition (Conmy 2023 §3.2).

## 5. API

### 5.1 New public function

```python
def extract_circuit(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    tau: float = 0.02,
    top_k_candidates: int = 2000,
    on_cell: Optional[Callable[[Dict], None]] = None,
) -> PatchingResult:
    """Cheap-ACDC circuit extraction (Syed et al. 2023).

    Runs the Phase 3.7 edge attribution pass, then annotates the top
    `top_k_candidates` edges with `in_circuit: bool` based on:
      1. |ap_recovery| >= tau  (filter)
      2. reader is reverse-reachable from 'logits' through surviving edges

    Returns PatchingResult with mode='circuit'. Cells include all top-k
    candidates (in-circuit and out). Summary fields:
      n_edges              - total pre-filter edge count
      n_edges_in_circuit   - count of cells with in_circuit=True
      n_nodes_in_circuit   - |visited| from reverse-BFS, inclusive of the
                              logits sink (so a graph with only embed->logits
                              yields n_nodes_in_circuit == 2).
      tau                  - applied threshold

    If `top_k_candidates > total valid edges`, silently caps at the actual
    edge count (matches edge_attribution_patch's top_k_edges behavior).
    """
```

Validation:
- `tau >= 0.0` (τ = 0 ⇒ every top-k edge is in-circuit, degenerate but valid)
- `top_k_candidates >= 1`
- Same prompt/direction/position validation as `edge_attribution_patch`

### 5.2 Refactor: `_compute_all_edges`

Extract the forward+backward+edge-enumeration body from `edge_attribution_patch` (lines ~1474–1649) into a new private helper:

```python
def _compute_all_edges(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str,
    measurement_position: int,
    positions: Optional[List[int]],
    layers: Optional[List[int]],
) -> Tuple[
    List[Dict],           # all_edge_scores: unsorted list of all valid edges
    float,                # D: Δ_clean - Δ_corrupted
    torch.Tensor,         # clean_baseline_logits
    torch.Tensor,         # corrupted_baseline_logits
    List[str],            # prompt_tokens_clean
    List[str],            # prompt_tokens_corrupted
    int,                  # meas_pos (normalized)
    int,                  # n_heads
]:
    """Shared forward+backward+edge-enumeration core.

    Used by both edge_attribution_patch (top-k wrapper) and
    extract_circuit (tau + connectivity wrapper).
    """
```

Callers:
- `edge_attribution_patch` — takes `all_edge_scores`, sorts by |ap_recovery| desc, slices top-k, emits via `on_cell`.
- `extract_circuit` — takes `all_edge_scores`, sorts by |ap_recovery| desc, slices top-`top_k_candidates`, runs reverse-BFS with threshold τ, annotates each kept candidate with `in_circuit: bool`, emits via `on_cell`.

Neither caller changes `PatchingResult` construction logic beyond the new summary fields.

### 5.3 `PatchingResult` extensions

```python
@dataclass
class PatchingResult:
    # ...existing fields...
    mode: str = "exact"  # now: "exact" | "approx" | "approx_head" | "edge" | "circuit"
    n_heads: Optional[int] = None
    n_edges: Optional[int] = None
    n_edges_in_circuit: Optional[int] = None   # NEW: set by extract_circuit
    n_nodes_in_circuit: Optional[int] = None   # NEW: set by extract_circuit
    tau: Optional[float] = None                # NEW: set by extract_circuit
```

All new fields are `Optional` with default `None` — preserves every existing call site.

### 5.4 Cell schema

Circuit-mode cells extend edge-mode cells with one boolean:

```python
{
    "writer_layer": int,
    "writer_unit": str,      # "embed" | "ffn" | "attn.hN"
    "reader_layer": int,     # N_L for logits reader
    "reader_unit": str,      # "attn_in" | "ffn_in" | "logits"
    "position": int,
    "ap_recovery": float,
    "in_circuit": bool,      # NEW
}
```

## 6. WS protocol

Route `/ws/sessions/{name}/activation-patching` gains a fifth mode.

**Client sends:**
```jsonc
{
  "op": "activation-patching",
  "cfg": {
    "mode": "circuit",
    "clean_prompt": "The cat sat on the",
    "corrupted_prompt": "The dog ran to the",
    "direction": "denoise",
    "measurement_position": -1,
    "correct_token_id": 5156,
    "incorrect_token_id": 6374,
    "tau": 0.02,                    // NEW
    "top_k_candidates": 2000,       // NEW
    "positions": null,
    "layers": null
  }
}
```

**Server streams**: `status` → `data` (one per cell) → `baselines` → `complete`.

Data frames: same shape as edge mode, plus `in_circuit: bool`.

Complete summary gains:
```jsonc
{
  "n_edges": 90432,
  "n_edges_in_circuit": 47,
  "n_nodes_in_circuit": 31,
  "tau": 0.02,
  "top_k_candidates": 2000
}
```

Token-ID auto-pick extends to circuit mode (same logic as edge mode).

## 7. Frontend

### 7.1 Types

`api.ts`:
- `PatchingMode` gains `"circuit"`.
- `PatchingCellData.in_circuit?: boolean`.
- `PatchingCompleteData.summary` gains optional `n_edges_in_circuit`, `n_nodes_in_circuit`, `tau`, `top_k_candidates`.

### 7.2 Controls

`PatchingControls.tsx`:
- Fifth radio labeled "Circuit (ACDC)" in the mode group.
- When `mode === "circuit"` is selected, show two numeric inputs:
  - `tau` (default 0.02, step 0.005, min 0)
  - `top_k_candidates` (default 2000, min 1)
- Inputs propagate into WS `cfg`.

`PatchingState` (in the store or component):
- Adds `tau: number`, `top_k_candidates: number` with the defaults above.

### 7.3 Visualization: `CircuitPanel.tsx`

New top-level panel (sibling of `EdgeAttributionPanel.tsx`). Routed from `VisualizationArea.tsx` when `mode === "circuit"`.

**Layout**: one position selector at top, τ slider below it, then visualization area.

**τ slider**:
- Range: `[0, max(|ap_recovery|) across received cells]`.
- Step: 0.001.
- Initial value: τ from the last received `complete.summary.tau` (the backend-applied τ).
- **On change**: client-side re-filter by new τ + re-BFS + re-render. No WS round-trip. This is the UX centerpiece.

**Visualization**: Sankey-style layout with only in-circuit edges (at current slider τ):
- Writers left column, readers right column (same as Phase 3.7 Sankey).
- Nodes colored by whether they're reverse-reachable at current τ (blue) vs. not (gray).
- Edge thickness ∝ `|ap_recovery|`, color by sign (PiYG, same as Phase 3.7).
- Node labels: `L0.h3` for attn head, `L5.ffn`, `embed`, `logits`.

**Stats strip** (below Sankey):
- "Edges in circuit: N of M" (M = received cells)
- "Nodes in circuit: K"
- "τ = X.XXX"
- Toggle: "Show only in-circuit edges" vs "dim out-of-circuit".

**Export button**:
- Copies circuit as JSON list `[{writer, reader, position, ap_recovery}, ...]` at current τ.

### 7.4 Non-pollution of `EdgeAttributionPanel`

`EdgeAttributionPanel.tsx` stays unchanged. `CircuitPanel.tsx` is a new file. If a Sankey primitive (the cubic-Bezier edge renderer) would be duplicated, extract it into `visualizations/SankeyEdges.tsx` in a follow-up, not in this phase's scope.

### 7.5 Fixture & smoke test

- `frontend/tests/e2e/fixtures/activation-patching-circuit.json` — mirrors `activation-patching-edge.json` schema (`"schema": "llm-surgeon-gui-experiment/v1"`), cells carry `in_circuit`, summary has `mode: "circuit"`, `tau`, `n_edges_in_circuit`, `n_nodes_in_circuit`.
- 14th Playwright test in `smoke.spec.ts` — imports the fixture, asserts `CircuitPanel` renders, asserts τ slider is present, asserts edge count matches `n_edges_in_circuit`.

## 8. Testing

### 8.1 Python unit tests (`test_probe_circuit.py`)

New test file. Mock-model tests (fast, no GPU):

1. **`test_extract_circuit_returns_patching_result`** — smoke, mock model.
2. **`test_circuit_matches_edge_ap_on_scores`** — `extract_circuit(tau=0)` produces the same top-k edge scores as `edge_attribution_patch(top_k=top_k_candidates)`. Verifies the refactor is behavior-preserving.
3. **`test_circuit_reverse_bfs_correctness`** — hand-constructed 4-edge graph: `embed → A`, `A → B`, `B → logits`, `C → D` (disconnected from logits). Assert `{embed→A, A→B, B→logits}` are in-circuit, `{C→D}` is not.
4. **`test_circuit_tau_filters_edges`** — two edges with known scores (0.05, 0.01). τ=0.03 ⇒ only the first in-circuit, τ=0.005 ⇒ both in-circuit.
5. **`test_circuit_tau_zero_marks_all_topk`** — τ=0 ⇒ every top-k candidate is in-circuit (provided the graph is connected). Tests the degenerate case.
6. **`test_circuit_validation`** — negative τ raises, `top_k_candidates < 1` raises, empty prompts raise, mismatched-length prompts raise.
7. **`test_n_nodes_in_circuit_counts_correctly`** — verify the `n_nodes_in_circuit` summary matches `|visited| - 1` (excluding the logits sink node itself? Decision: count all distinct `(layer, unit, position)` tuples including logits). Pick one convention and assert it.
8. **`test_in_circuit_flag_per_cell`** — each cell has `in_circuit: bool`, no cell has it unset.
9. **`test_compute_all_edges_helper_direct`** — direct call to `_compute_all_edges` returns the expected 8-tuple, assert structure.

### 8.2 TinyLlama integration test (GPU-guarded)

`test_extract_circuit_tinyllama` — skipif TinyLlama not cached, marks `@pytest.mark.slow`, uses `load_model(..., mode="fp16")` (same as Phase 3.7 for OOM-avoidance).

Assertions:
- Result has `mode == "circuit"`, `n_edges_in_circuit > 0` at τ=0.02, `n_nodes_in_circuit > 0`.
- At τ=0: `n_edges_in_circuit == len(result.cells)` (degenerate case).
- At τ=1.0 (very high): `n_edges_in_circuit == 0` (no edge clears the bar).
- `len(result.cells) == top_k_candidates` or `== n_edges` (whichever is smaller).

### 8.3 Frontend tests

- **Vitest**: `test_circuitBFS.ts` — pure BFS helper in `utils/circuitBFS.ts`. Test cases from §8.1 #3–#5 mirrored in JS.
- **Playwright**: fixture-based smoke (§7.5).

## 9. Commit plan

10 tasks, each one commit. TDD where possible (tests before implementation).

1. **Spec commit** — this file.
2. **Helper refactor** — extract `_compute_all_edges` from `edge_attribution_patch`. Existing edge tests still pass.
3. **`extract_circuit` implementation** — new function + validation.
4. **Python unit tests** — mock-model cases §8.1.
5. **TinyLlama integration test** — §8.2.
6. **Backend WS `mode="circuit"` branch** — `routes/probes.py`.
7. **Frontend types** — `api.ts`.
8. **`PatchingControls`** — fifth radio + `tau`/`top_k_candidates` inputs.
9. **`CircuitPanel.tsx`** + `utils/circuitBFS.ts` + Vitest — core viz.
10. **`VisualizationArea` routing + Playwright fixture + smoke test** — §7.5.

## 10. Verification matrix

Before declaring phase shipped:

- **pyright**: 0 errors / 0 warnings / 0 info across `probe.py`, `routes/probes.py`, all three AP test files.
- **tsc**: clean on frontend.
- **pytest**: all existing non-GPU tests pass (Phase 3.5/3.6/3.7 regressions) + new unit tests pass.
- **TinyLlama**:
  - Phase 3.5 Spearman vs exact preserved at ≥ 0.95.
  - Phase 3.6 sum-heads vs attn_out target preserved at ρ=1.0000.
  - Phase 3.7 top-k consistency still passes.
  - New circuit integration test passes at τ=0.02 with nonzero circuit.
- **Vitest**: all existing + new BFS tests pass.
- **Playwright**: 13 existing + 1 new = 14 passing.

## 11. Rollout

Single branch → master. No feature flag; `mode="circuit"` is opt-in per WS request, all existing modes unaffected.

## 12. Open questions

- ~~Should `n_nodes_in_circuit` include the logits sink?~~ **Resolved**: yes, count all reachable nodes including logits (documented in §5.1 docstring and §8.1 test #7).
- Is there a position-aggregated view where edges are summed over positions? **Deferred to Phase 3.8.1** — this phase keeps positions distinct (same as Phase 3.7).
- DOT/Graphviz export? **Deferred** — JSON copy-to-clipboard is enough for v1. Users can pipe to external tools.
