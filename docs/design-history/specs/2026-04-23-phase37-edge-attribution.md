# Phase 3.7 — Edge Attribution Patching Design

**Date:** 2026-04-23
**Roadmap:** `llm_surgeon/project_llm_surgeon_roadmap.md` (Phase 3.7)
**Paper:** [Syed et al. 2023, "Attribution Patching Outperforms Automated Circuit Discovery"](https://arxiv.org/abs/2310.10348)
**Goal:** extend gradient-based attribution patching from per-node granularity (Phase 3.5/3.6) to per-edge granularity. Given clean and corrupted prompts, produce per-`(writer, reader, position)` AP scores for every valid directed edge in the computational graph, using a single forward + backward pass.

**Why edges (not just nodes):** node-level AP tells you which layer/head matters; edge-level AP tells you *from where* that component gets its signal. This is the difference between knowing that head L3.H7 is important and knowing that it reads from embedding + L1.H2 + L2.H5. Edges are the primitive of circuit discovery (ACDC, EAP).

**Why a separate function (not a flag on existing functions):** `edge_attribution_patch` has a fundamentally different output contract (edges, not nodes), different capture structure (reader-grad hooks on layernorms in addition to writer captures), and different frontend needs (Sankey + matrix, not a single heatmap). Keeping it separate preserves Phase 3.5/3.6 invariants.

**Non-goals for Phase 3.7:** per-neuron FFN decomposition, integrated gradients, exact edge AP, automatic circuit extraction (ACDC-style thresholding), QK/OV circuit decomposition, per-head FFN decomposition.

---

## 1. Architecture Overview

### 1.1 Residual stream linearity

In HuggingFace LLaMA, the residual stream at any point is an additive accumulation:

```
h_L = embed + Σ_{L'=0}^{L-1} (attn_out_{L'} + ffn_out_{L'})
```

The input to any reader at layer L is this sum (before the reader's own layernorm). Because addition is linear, the gradient of any scalar metric `m` with respect to `h_L` equals the gradient with respect to each individual writer's output:

```
∂m/∂embed  = ∂m/∂h_L     (for any L ≥ 1)
∂m/∂attn_out_{L'} = ∂m/∂h_L    for all L' ≤ L
∂m/∂ffn_out_{L'}  = ∂m/∂h_L    for all L' < L  (for attn_in reader)
                                 for all L' ≤ L  (for ffn_in reader)
```

This is the key that makes EAP cost-equivalent to node-level AP: one backward pass populates all reader gradients, and each writer's delta is already captured.

### 1.2 Writers

A writer is any component that additively contributes to the residual stream.

| Writer type | Key format | Output |
|---|---|---|
| Token embedding | `(0, "embed")` | `embed_tokens(input_ids)`, shape `[1, seq, hidden]` |
| Attention head h at layer L | `(L, "attn.hN")` | Slice of `concat_z[L]` projected through `W_O[L]`, shape `[1, seq, hidden]` |
| FFN at layer L | `(L, "ffn")` | `ffn_out[L]`, the MLP output delta, shape `[1, seq, hidden]` |

For per-head writer decomposition, reuse the Phase 3.6 `concat_z` capture. The contribution of head h at layer L to `attn_out[L]` is:

```
head_contrib[L, h] = concat_z[L, :, h*d:(h+1)*d] @ W_O[L][h*d:(h+1)*d, :].T
```

where `d = hidden // n_heads = head_dim`.

### 1.3 Readers

A reader is a sublayer that consumes the residual stream as its input.

| Reader type | Key format | Hook target | Notes |
|---|---|---|---|
| Attention input at layer L | `(L, "attn_in")` | `model.model.layers[L].input_layernorm` (pre-hook) | LN input = residual stream before attn |
| FFN input at layer L | `(L, "ffn_in")` | `model.model.layers[L].post_attention_layernorm` (pre-hook) | LN input = residual stream after attn, before FFN |
| Final logit read | `(N_L, "logits")` | `model.model.norm` (pre-hook) | N_L = number of layers (virtual last reader) |

The pre-hook receives `args[0]` which is the layernorm's input — this is the unmodified residual stream tensor, in the autograd graph, eligible for `retain_grad()`.

### 1.4 Valid edges

An edge `(writer w, reader r)` is valid when the writer's contribution is present in the residual stream at the time the reader reads it:

- `embed → (L, "attn_in")`: valid for all L ≥ 0.
- `embed → (L, "ffn_in")`: valid for all L ≥ 0.
- `(L_w, "attn.hN") → (L_r, "attn_in")`: valid when L_w < L_r (attn writes after attn_in reads, so same-layer attn writer cannot be an ancestor of same-layer attn_in reader).
- `(L_w, "attn.hN") → (L_r, "ffn_in")`: valid when L_w ≤ L_r (attn at layer L writes before FFN at layer L reads — within the same layer, attn completes first).
- `(L_w, "ffn") → (L_r, "attn_in")`: valid when L_w < L_r.
- `(L_w, "ffn") → (L_r, "ffn_in")`: valid when L_w < L_r.
- `embed → (N_L, "logits")`: valid.
- `(L_w, "attn.hN") → (N_L, "logits")`: valid for all L_w.
- `(L_w, "ffn") → (N_L, "logits")`: valid for all L_w.

Edge count for TinyLlama (22 layers, 32 heads):
- Writers: 1 embed + 22×32 attn heads + 22 ffn = 727
- Readers: 22 attn_in + 22 ffn_in + 1 logits = 45
- Valid edge pairs: approximately 15,000 (before position expansion)
- At 6 positions: ~90,000 edge scores (dominated by the dense cross-layer head pairs)

### 1.5 Edge AP score — core math

Let:
- `grad_r[pos]` = `∂metric/∂(residual_stream_input_to_reader_r)[0, pos]`, shape `[hidden]`. Captured via forward pre-hook on the reader's layernorm.
- `Δwrite_w[pos]` = `(clean_write_w[0, pos] - corrupted_write_w[0, pos])`, shape `[hidden]`. For attn head writers, this is the per-head delta; for FFN writers, the FFN output delta; for embed, the embedding delta.

Then:

```
AP_edge(w → r, pos) = (Δwrite_w[pos] · grad_r[pos]).sum()
```

Normalized (denoise direction):

```
ap_recovery_edge(w → r, pos) = AP_edge(w → r, pos) / (d_clean - d_corrupted)
```

Noise direction:

```
ap_recovery_edge(w → r, pos) = 1 + AP_edge(w → r, pos) / (d_clean - d_corrupted)
```

### 1.6 Per-head writer computation via chain rule (reuse Phase 3.6)

For writer `(L, "attn.hN")` and reader r:

```
AP_edge((L, attn.hN) → r, pos)
  = (Δconcat_z[L, 0, pos, h*d:(h+1)*d]  ·  (grad_r[pos] @ W_O[L])[h*d:(h+1)*d]).sum()
```

Let `grad_z_r[L, pos] = grad_r[pos] @ W_O[L]`, shape `[hidden]`. Reshape to `[n_heads, head_dim]`:

```python
grad_z_r_heads = (grad_r[pos] @ W_O[L]).view(n_heads, head_dim)    # [n_heads, head_dim]
delta_z_heads  = delta_z[L, 0, pos].view(n_heads, head_dim)         # [n_heads, head_dim]
ap_heads = (delta_z_heads * grad_z_r_heads).sum(dim=-1)              # [n_heads]  — all heads at once
```

This is an O(hidden²) matmul per (layer, reader, position) triple, amortized across all 32 heads simultaneously. One matmul per (L_writer, reader) pair gives all head scores for that pair.

### 1.7 FFN writer and embed writer

```python
# FFN writer at layer L_w, reader r at position pos:
ap_ffn = (delta_ffn[L_w][0, pos] * grad_r[pos]).sum()

# Embed writer, reader r at position pos:
ap_embed = (delta_embed[0, pos] * grad_r[pos]).sum()
```

No projection needed — FFN and embed write directly to the residual stream in hidden-dim space.

### 1.8 Sum invariant

For any fixed reader r and position pos:

```
Σ_{all valid writers w} AP_edge(w → r, pos)
  ≈ (Δresidual_at_r[0, pos] · grad_r[pos]).sum()
  = (Δresidual_at_r[0, pos] · ∂metric/∂residual_at_r[0, pos]).sum()
```

This holds because `Δresidual_at_r = Σ Δwrite_w` (linearity of residual stream) and the dot product distributes. Tolerance: 1e-5 on a mock model.

### 1.9 Compute cost

One forward + one backward pass, identical to Phase 3.5/3.6. The edge loop is pure tensor operations (no additional forward passes):

- Reader grad captures: N_L pre-hooks on LN modules (forward pass cost only).
- Writer delta captures: same as Phase 3.6 (concat_z + FFN hooks).
- Edge loop: for each (L_w, reader, pos) triple — one `@ W_O` matmul per attn-writer layer + vectorized dot products. Dominated by the matmul: 22 writer layers × 45 readers × 6 positions ≈ 5940 matmuls of size `[1, hidden]` × `[hidden, hidden]` — but since grad_r is fixed per (reader, pos), we can batch: 45 readers × 6 positions matmuls upfront, then all 32 heads per writer layer read from precomputed `grad_z_r`.

Total new overhead vs Phase 3.6: ~10ms on RTX 2080 for TinyLlama.

---

## 2. Python API

### 2.1 Extended `_capture_residual_stream_with_grad`

Add `capture_reader_grads: bool = False` keyword argument. When True, registers forward pre-hooks on the reader layernorms and the final norm:

```python
def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    capture_concat_z: bool = False,
    capture_reader_grads: bool = False,          # NEW for Phase 3.7
) -> Tuple[
    Dict[Tuple[int, str], torch.Tensor],         # captured residual states
    Dict[int, torch.Tensor],                      # h_ins
    torch.Tensor,                                 # output logits
    List[str],                                    # prompt tokens
    Dict[int, torch.Tensor],                      # concat_z per layer
    Dict[Tuple, torch.Tensor],                    # reader_grad_inputs (NEW, empty if flag=False)
]: ...
```

When `capture_reader_grads=True`, additional hooks capture the pre-LN residual stream at each reader:

```python
reader_inputs: Dict[Tuple, torch.Tensor] = {}

for i in range(num_layers):
    if capture_reader_grads:
        def make_attn_in_hook(idx: int):
            def hook(_module: torch.nn.Module, args: Tuple) -> None:
                x = args[0]
                if x.requires_grad:
                    x.retain_grad()
                reader_inputs[("attn_in", idx)] = x
            return hook
        hooks.append(
            model.model.layers[i].input_layernorm.register_forward_pre_hook(
                make_attn_in_hook(i)
            )
        )

        def make_ffn_in_hook(idx: int):
            def hook(_module: torch.nn.Module, args: Tuple) -> None:
                x = args[0]
                if x.requires_grad:
                    x.retain_grad()
                reader_inputs[("ffn_in", idx)] = x
            return hook
        hooks.append(
            model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(
                make_ffn_in_hook(i)
            )
        )

if capture_reader_grads:
    n_layers = len(model.model.layers)
    def make_logits_hook(n: int):
        def hook(_module: torch.nn.Module, args: Tuple) -> None:
            x = args[0]
            if x.requires_grad:
                x.retain_grad()
            reader_inputs[("logits", n)] = x
        return hook
    hooks.append(
        model.model.norm.register_forward_pre_hook(make_logits_hook(n_layers))
    )
```

Return signature extends from 5-tuple to 6-tuple. Existing callers (`attribution_patch`, `attribution_patch_per_head`) unpack 6 values, discarding the 6th with `_`.

**HF LLaMA hook targets:**
- `attn_in` at layer L: `model.model.layers[L].input_layernorm` — this LN sits between the residual stream and the QKV projections.
- `ffn_in` at layer L: `model.model.layers[L].post_attention_layernorm` — this LN sits between the post-attn residual stream and the MLP gate/up projections.
- `logits`: `model.model.norm` — the final RMSNorm before the LM head. Its input is the last-layer residual stream.

In all cases, `args[0]` in the pre-hook is the tensor flowing into the LN — the unmodified residual stream at that point.

### 2.2 Embed delta capture

The embedding delta is not captured by existing hooks. `edge_attribution_patch` captures it directly:

```python
with torch.no_grad():
    clean_embed = model.model.embed_tokens(
        tokenizer(clean_prompt, return_tensors="pt")["input_ids"].to(device)
    ).detach()
    corr_embed = model.model.embed_tokens(
        tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"].to(device)
    ).detach()
delta_embed = clean_embed - corr_embed   # [1, seq, hidden]
```

### 2.3 FFN output delta

FFN output is the difference `ffn_out[L] = layer_output[L] - h_post_attn[L]`. In Phase 3.6, `captured[(L, "ffn")]` is the layer's full output (post-FFN residual stream), and `captured[(L, "attn")]` + `h_ins[L]` gives `h_post_attn`. So:

```python
delta_ffn[L][pos] = (
    (from_captured[(L, "ffn")][0, pos] - from_h_ins[L][0, pos] - from_captured[(L, "attn")][0, pos])
    - (base_captured[(L, "ffn")][0, pos] - base_h_ins[L][0, pos] - base_captured[(L, "attn")][0, pos])
)
```

Or equivalently, capture `ffn_out` directly by a post-hook on `model.model.layers[L].mlp` (the MLP module output before the residual add). A dedicated `capture_ffn_out: bool` flag on `_capture_residual_stream_with_grad` can be added, or derived from existing captures. The spec defers the choice to implementation — either derivation or direct capture is correct; direct capture is cleaner.

### 2.4 `PatchingResult` extension

```python
@dataclass
class PatchingResult:
    cells: List[Dict]
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str
    measurement_position: int
    mode: str = "exact"              # "exact" | "approx" | "approx_head" | "edge"
    n_heads: Optional[int] = None    # set by attribution_patch_per_head / edge_attribution_patch
    n_edges: Optional[int] = None    # NEW: total valid edge count before top-k filtering
```

### 2.5 Edge cell dict format

```python
{
    "writer_layer": int,           # writer's layer index (0 = embed writer layer is 0)
    "writer_unit": str,            # "embed" | "attn.hN" | "ffn"
    "reader_layer": int,           # reader's layer index (N_L for logits reader)
    "reader_unit": str,            # "attn_in" | "ffn_in" | "logits"
    "position": int,               # token position (absolute, 0-indexed)
    "ap_recovery": float,          # edge AP score (normalized)
}
```

Example:
```python
{"writer_layer": 3, "writer_unit": "attn.h5", "reader_layer": 11,
 "reader_unit": "attn_in", "position": 5, "ap_recovery": 0.042}
```

### 2.6 New function: `edge_attribution_patch`

```python
def edge_attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",           # "denoise" | "noise"
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    top_k_edges: int = 200,               # backend emits only top-k by |ap_recovery|
    on_cell: Optional[Callable[[Dict], None]] = None,
) -> PatchingResult: ...
```

Notes:
- `on_cell` receives a single `cell: dict` (not the `(layer, unit, position, cell)` signature of earlier modes). This is because edges have two layers (writer + reader), making the old 4-arg signature ambiguous. Callers in `probes.py` must adapt.
- `top_k_edges`: after computing all valid edge scores, sort by `|ap_recovery|` descending and emit only the top-k. The full dense set is computed internally; `result.n_edges` records the pre-filter count.
- Returns `PatchingResult(mode="edge", n_edges=total_edge_count, n_heads=n_heads)`.

### 2.7 Core algorithm

```python
# Step 1: Capture 'clean' (from-prompt in denoise) — no_grad
with torch.no_grad():
    clean_captured, clean_h_ins_raw, clean_logits, clean_tokens, clean_cz_raw, _ = \
        _capture_residual_stream_with_grad(
            model, tokenizer, clean_prompt,
            sublayers=("attn", "ffn"), layers=layers,
            capture_concat_z=True,
            capture_reader_grads=False,
        )
    clean_states   = {k: v.detach().clone() for k, v in clean_captured.items()}
    clean_h_ins    = {k: v.detach().clone() for k, v in clean_h_ins_raw.items()}
    clean_cz       = {k: v.detach().clone() for k, v in clean_cz_raw.items()}

# Step 2: Capture 'corrupted' (base in denoise) — with grad + reader grads
with torch.enable_grad():
    corr_captured, corr_h_ins, corr_logits, corr_tokens, corr_cz, reader_inputs = \
        _capture_residual_stream_with_grad(
            model, tokenizer, corrupted_prompt,
            sublayers=("attn", "ffn"), layers=layers,
            capture_concat_z=True,
            capture_reader_grads=True,
        )
    # Metric: logit_diff at measurement_position
    meas = measurement_position % corr_logits.shape[0]
    d_clean = (clean_logits[meas, correct_token_id] - clean_logits[meas, incorrect_token_id]).item()
    d_corr  = (corr_logits[meas, correct_token_id]  - corr_logits[meas, incorrect_token_id])
    if abs(d_clean - d_corr.item()) < 1e-6:
        raise ValueError("clean and corrupted baselines have identical logit_diff; AP would divide by zero")
    metric = d_corr     # scalar (denoise: base=corrupted, direction of recovery)
    metric.backward()

# Step 3: Embed delta
delta_embed = (clean_embed - corr_embed)  # [1, seq, hidden], computed separately

# Step 4: Writer deltas
# attn writer delta = per-head delta in concat_z space (Phase 3.6 pattern)
# ffn writer delta = (clean_ffn_out - corr_ffn_out) where ffn_out = layer_out - h_post_attn

# Step 5: Reader grads
# After backward(), reader_inputs[("attn_in", L)].grad, etc. are populated.
# grad_r[pos] = reader_inputs[reader_key][0, pos].grad   # [hidden]

# Step 6: Edge loop
n_heads: int = model.config.num_attention_heads
hidden: int  = model.config.hidden_size
head_dim: int = hidden // n_heads

all_edge_scores: List[Dict] = []

for reader_key, reader_tensor in reader_inputs.items():
    reader_type = reader_key[0]   # "attn_in" | "ffn_in" | "logits"
    reader_L    = reader_key[1]   # layer index or N_L for logits

    for pos in normalized_positions:
        grad_r = reader_tensor.grad[0, pos]   # [hidden]

        # Embed writer
        ap_embed = (delta_embed[0, pos] * grad_r).sum().item()
        ap_recovery = ap_embed / denominator
        all_edge_scores.append({
            "writer_layer": 0, "writer_unit": "embed",
            "reader_layer": reader_L, "reader_unit": reader_type,
            "position": pos, "ap_recovery": float(ap_recovery),
        })

        for L_w in sorted_writer_layers:
            # FFN writer
            if _is_valid_ffn_writer(L_w, reader_type, reader_L):
                ap_ffn = (delta_ffn[L_w][0, pos] * grad_r).sum().item()
                all_edge_scores.append({
                    "writer_layer": L_w, "writer_unit": "ffn",
                    "reader_layer": reader_L, "reader_unit": reader_type,
                    "position": pos, "ap_recovery": float(ap_ffn / denominator),
                })

            # Attn head writers (vectorized over all heads)
            if _is_valid_attn_writer(L_w, reader_type, reader_L):
                W_O = model.model.layers[L_w].self_attn.o_proj.weight  # [hidden, hidden]
                # grad_z_r: gradient pushed back through W_O to concat_z space
                grad_z_r = grad_r @ W_O                       # [hidden]  (row vec * mat)
                grad_z_r_heads = grad_z_r.view(n_heads, head_dim)   # [n_heads, head_dim]
                delta_z = clean_cz[L_w][0, pos] - corr_cz[L_w][0, pos]  # [hidden]
                delta_z_heads = delta_z.view(n_heads, head_dim)           # [n_heads, head_dim]
                ap_heads = (delta_z_heads * grad_z_r_heads).sum(dim=-1)   # [n_heads]

                for h in range(n_heads):
                    unit = f"attn.h{h}"
                    all_edge_scores.append({
                        "writer_layer": L_w, "writer_unit": unit,
                        "reader_layer": reader_L, "reader_unit": reader_type,
                        "position": pos,
                        "ap_recovery": float(ap_heads[h].item() / denominator),
                    })

# Step 7: Top-k selection
n_edges_total = len(all_edge_scores)
all_edge_scores.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
top_cells = all_edge_scores[:top_k_edges]

if on_cell is not None:
    for cell in top_cells:
        on_cell(cell)

return PatchingResult(
    cells=top_cells,
    ...
    mode="edge",
    n_heads=n_heads,
    n_edges=n_edges_total,
)
```

**Direction:** in denoise mode, `from_prompt = clean_prompt`, `base_prompt = corrupted_prompt`. For `noise` direction, roles swap and `ap_recovery = 1 + ap_raw / denominator`.

**Valid edge predicates:**

```python
def _is_valid_attn_writer(L_w: int, reader_type: str, reader_L: int) -> bool:
    if reader_type == "attn_in":
        return L_w < reader_L
    if reader_type == "ffn_in":
        return L_w <= reader_L   # same-layer attn → same-layer ffn_in is valid
    if reader_type == "logits":
        return True
    return False

def _is_valid_ffn_writer(L_w: int, reader_type: str, reader_L: int) -> bool:
    if reader_type in ("attn_in", "ffn_in"):
        return L_w < reader_L
    if reader_type == "logits":
        return True
    return False
```

### 2.8 Validation rules

| Condition | Error |
|---|---|
| All Phase 3.5/3.6 validation conditions | as before |
| `correct_token_id` or `incorrect_token_id` is None | `ValueError("edge_attribution_patch requires correct_token_id and incorrect_token_id")` |
| `abs(d_clean - d_corr) < 1e-6` | `ValueError("clean and corrupted baselines have identical logit_diff; AP would divide by zero")` |
| `top_k_edges < 1` | `ValueError("top_k_edges must be >= 1")` |
| Prompts tokenize to different lengths | `ValueError("prompts must tokenize to same length ...")` |

### 2.9 Memory profile

For TinyLlama (22 layers, hidden=2048, n_heads=32, head_dim=64, seq≈6):
- `reader_inputs`: 45 tensors of shape `[1, 6, 2048]` × f32 = 45 × 48 KB = 2.2 MB.
- Grad tensors (same size): 2.2 MB.
- `concat_z` (already in Phase 3.6): 1 MB.
- `delta_embed`: `[1, 6, 2048]` × f32 = 48 KB.
- Edge score list: ~90,000 dicts of 6 scalars — ~15 MB in Python objects (acceptable; freed after top-k selection).
- Total new overhead vs Phase 3.6: ~20 MB. Negligible on RTX 2080.

---

## 3. Backend WS Route

**No new route.** `/ws/sessions/{name}/activation-patching` gains a fourth `mode` value:

```python
mode = cfg.get("mode", "exact")
if mode not in ("exact", "approx", "approx_head", "edge"):
    send error; return
```

Config gains one new field:

```python
top_k_edges = int(config.get("top_k_edges", 200))
```

Auto-pick token IDs extends to `"edge"` mode:

```python
if mode in ("approx", "approx_head", "edge") and correct_token_id is None:
    # auto-pick via argmax of clean/corrupted logits (existing logic, unchanged)
```

### 3.1 `on_cell` adaptation

`edge_attribution_patch`'s `on_cell` receives a single `cell: dict` (not the 4-arg signature). The route handler must adapt:

```python
if mode == "edge":
    def on_cell(cell: dict) -> None:
        nonlocal connected
        if not connected:
            return
        msg: dict = {
            "type": "data",
            "writer_layer": cell["writer_layer"],
            "writer_unit": cell["writer_unit"],
            "reader_layer": cell["reader_layer"],
            "reader_unit": cell["reader_unit"],
            "position": cell["position"],
            "ap_recovery": cell["ap_recovery"],
        }
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
        try:
            ok = fut.result(timeout=10)
        except Exception:
            ok = False
        if not ok:
            connected = False
```

### 3.2 Dispatch

```python
elif mode == "edge":
    from llm_surgeon.probe import edge_attribution_patch
    assert correct_token_id is not None and incorrect_token_id is not None
    _cid: int = correct_token_id
    _iid: int = incorrect_token_id
    result = await loop.run_in_executor(
        None,
        lambda: edge_attribution_patch(
            info.model, info.tokenizer,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            correct_token_id=_cid,
            incorrect_token_id=_iid,
            direction=direction,
            measurement_position=measurement_position,
            positions=positions,
            layers=layers,
            top_k_edges=top_k_edges,
            on_cell=on_cell,
        ),
    )
```

### 3.3 Data frames

```json
{
  "type": "data",
  "writer_layer": 3,
  "writer_unit": "attn.h5",
  "reader_layer": 11,
  "reader_unit": "attn_in",
  "position": 5,
  "ap_recovery": 0.042
}
```

### 3.4 Complete frame

```json
{
  "type": "complete",
  "summary": {
    "num_cells": 200,
    "direction": "denoise",
    "measurement_position": 5,
    "mode": "edge",
    "n_heads": 32,
    "n_edges": 90432
  }
}
```

---

## 4. Frontend Types

### 4.1 `api.ts` additions

```typescript
export interface EdgeCellData {
  type: "data";
  writer_layer: number;
  writer_unit: string;        // "embed" | "attn.hN" | "ffn"
  reader_layer: number;
  reader_unit: string;        // "attn_in" | "ffn_in" | "logits"
  position: number;
  ap_recovery: number;
}

// Extend PatchingCompleteData summary:
export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
    mode?: "exact" | "approx" | "approx_head" | "edge";   // "edge" is new
    n_heads?: number;
    n_edges?: number;                                       // NEW: total edge count before top-k
  };
}
```

`EdgeCellData` is a separate interface (not a union extension of `PatchingCellData`) to avoid type-narrowing complexity, since the field shapes are structurally distinct.

### 4.2 `PatchingState` extension

```typescript
type PatchingMode = "exact" | "approx" | "approx_head" | "edge";
```

Add `top_k_edges: number` (default 200) to `PatchingState`.

### 4.3 `ProbeResult` union

`ProbeResult.data` items include `EdgeCellData` in the edge mode. The store may use a discriminated union or a generic `unknown[]` — ensure the viz component type-narrows correctly via `m.type === "data" && "writer_unit" in m`.

---

## 5. Frontend: `EdgeAttributionPanel.tsx`

New component. Does not modify `PerHeadPatchingHeatmap.tsx` or `ActivationPatchingHeatmap.tsx`.

### 5.1 Top-level structure

```tsx
export function EdgeAttributionPanel({ result }: { result: ProbeResult }) {
  const [view, setView] = useState<"sankey" | "matrix" | "list">("sankey");
  const [selectedPos, setSelectedPos] = useState<number>(0);
  // ...

  return (
    <div>
      <header>
        <PositionSelector ... />
        <TabBar tabs={["sankey", "matrix", "list"]} active={view} onChange={setView} />
      </header>
      {view === "sankey"  && <SankeyView  cells={posCells} ... />}
      {view === "matrix"  && <MatrixView  cells={posCells} ... />}
      {view === "list"    && <TopListView cells={posCells} ... />}
    </div>
  );
}
```

### 5.2 Sankey sub-view

- Left axis: writer-layer (grouped by layer, colored by writer type: embed/attn/ffn).
- Right axis: reader-layer (grouped by layer, colored by reader type: attn_in/ffn_in/logits).
- Bands: drawn from writer node to reader node, thickness proportional to `|ap_recovery|`, color from the Phase 3.6 color scale (`d3.interpolatePiYG`, domain `[-0.5, 1.0]`).
- Top-k only; no need to further filter.
- Hover tooltip: `writer_unit → reader_unit @ pos: AP`.
- Implementation note: use D3's `d3-sankey` plugin or implement manually with SVG `<path>` cubic Bezier curves between fixed x-coordinates for the two axis columns.

### 5.3 Matrix sub-view

```
         attn_in_L0  ffn_in_L0  attn_in_L1  ffn_in_L1  …  logits
embed         ■           ■           ■           ■       …    ■
attn.h0_L0    —           ■           ■           ■       …    ■
ffn_L0        —           —           ■           ■       …    ■
attn.h0_L1    —           —           —           ■       …    ■
…
```

- Rows: writers (embed first, then per-layer: attn heads then ffn). Strict topological order.
- Columns: readers (per-layer attn_in then ffn_in, then logits). Same order.
- Cells colored by `ap_recovery`. Fixed domain `[-0.5, 1.0]`, same scale.
- Invalid edges shown as grey (no score).
- Click to pin a detail card: `writer → reader @ pos: AP`.

### 5.4 Top-list sub-view

Ranked table of all top-k edges (for the selected position), sorted by `|ap_recovery|`:

```
#   Writer              Reader          Pos   AP
1   L3.attn.h5    →    L11.attn_in      5   0.312
2   L0.attn.h2    →    logits           5   0.289
…
```

Supports copy-to-clipboard for the full ranked list.

### 5.5 Position selector

```tsx
<select value={selectedPos} onChange={e => setSelectedPos(Number(e.target.value))}>
  {promptTokens.map((tok, i) => (
    <option key={i} value={i}>{i}: {tok}</option>
  ))}
</select>
```

Shared across all three sub-views.

### 5.6 Controls extension

`PatchingControls.tsx` gains a fourth mode radio:

```tsx
<label>
  <input type="radio" checked={state.mode === "edge"}
         onChange={() => onChange({ ...state, mode: "edge" })} />
  edge AP <span style={{ color: "#888", fontSize: 11 }}>(gradient EAP, writer→reader edges)</span>
</label>
```

Add `top_k_edges` numeric input (shown only when `mode === "edge"`, default 200).

### 5.7 Routing

`ProbePanel.tsx` routes on `completeFrame.summary.mode`:

- `"exact"` → `<ActivationPatchingHeatmap>`
- `"approx"` → `<ActivationPatchingHeatmap>`
- `"approx_head"` → `<PerHeadPatchingHeatmap>`
- `"edge"` → `<EdgeAttributionPanel>`  (new)

---

## 6. Verification Plan

### 6.1 Python unit tests (`testing/tests/test_probe_edge_ap.py`)

| Test class | Test | What it checks |
|---|---|---|
| `TestReaderGradCapture` | `test_reader_inputs_keys` | `capture_reader_grads=True` returns `reader_inputs` with keys `("attn_in", L)`, `("ffn_in", L)`, `("logits", N_L)` for all layers |
| `TestReaderGradCapture` | `test_reader_inputs_shape` | Each captured tensor has shape `[1, seq, hidden]` |
| `TestReaderGradCapture` | `test_reader_inputs_in_graph` | Base-side reader input tensors have `requires_grad=True` and non-None `grad_fn` |
| `TestReaderGradCapture` | `test_reader_grads_populated_after_backward` | After `metric.backward()`, `reader_inputs[k].grad` is non-None and non-zero for all k |
| `TestEdgeAP` | `test_edge_count_mock` | Mock (2 layers, 2 heads): total edge count matches formula (writers × readers subject to validity) |
| `TestEdgeAP` | `test_validation_errors` | `top_k_edges < 1`, identical baselines, mismatched prompt lengths all raise `ValueError` |
| `TestEdgeAP` | `test_sum_invariant_mock` | For any fixed reader r: `Σ_w AP_edge(w → r, pos)` ≈ `(Δresidual_at_r[pos] · grad_r[pos]).sum() / D` at 1e-5 |
| `TestEdgeAP` | `test_per_head_decomposability_mock` | For any fixed reader r and writer layer L: `Σ_h AP_edge((L, attn.hN) → r) == AP_edge((L, attn) → r)` at 1e-5, where the RHS is the node-level attn AP from Phase 3.5/3.6 |
| `TestEdgeAP` | `test_top_k_selection` | With `top_k_edges=3`, only 3 cells emitted; they are the 3 with largest `|ap_recovery|` from the full set |
| `TestEdgeAP` | `test_on_cell_signature` | `on_cell` receives a dict (not 4 args); keys include `writer_layer`, `writer_unit`, `reader_layer`, `reader_unit`, `position`, `ap_recovery` |
| `TestEdgeAP` | `test_embed_writer_present` | At least one cell per reader has `writer_unit == "embed"` |
| `TestEdgeAP` | `test_invalid_edges_absent` | No cell has a `ffn_in` reader with a same-layer or later-layer `ffn` writer (L_w < L_r strictly) |
| `TestTinyLlama` | `test_tinyllama_top_k_consistency` | With `top_k_edges=100`: top-100 by magnitude from `edge_attribution_patch` matches top-100 selected from the independently-computed full dense set; Spearman ρ == 1.0. Guarded by `@pytest.mark.skipif(not _tinyllama_cached() or not torch.cuda.is_available())` |

### 6.2 Return-signature regression

After extending `_capture_residual_stream_with_grad` to return 6-tuple: verify `attribution_patch` and `attribution_patch_per_head` callers are updated to unpack 6 values (discarding 6th with `_`). Run the full existing test suite to confirm no regression.

### 6.3 Backend (pyright)

`probes.py` must pass `pyright 0/0/0` after adding the `"edge"` branch and `top_k_edges` config read.

### 6.4 Frontend type check

`tsc --noEmit` clean (0 errors, 0 warnings) on all modified files after each task.

### 6.5 Playwright smoke

One new test in `smoke.spec.ts`:
- Seed fixture `activation-patching-edge.json` with `mode: "edge"` and a handful of edge-keyed data frames plus a complete frame with `n_edges`.
- Assert heading `/Edge Attribution/` (or similar) is visible.
- Assert position selector is present.
- Assert the Sankey/Matrix/Top-list tab bar is present.
- Assert no console errors beyond the `isBackendlessNoise()` filter.

Total suite: 12 existing (after Phase 3.6) + 1 new = 13 tests.

---

## 7. Data Flow Summary

```
User: mode = "edge", top_k_edges = 200 → ProbePanel.handleRun → WS cfg
  ↓
Backend: receive cfg → auto-pick token IDs (same as approx/approx_head)
  → edge_attribution_patch()
    ↓
    _capture_residual_stream_with_grad(capture_concat_z=True, capture_reader_grads=False)   [clean, no_grad]
    _capture_residual_stream_with_grad(capture_concat_z=True, capture_reader_grads=True)    [corrupted, enable_grad]
    metric.backward()                         [populates all reader_inputs[k].grad]
    for each (reader, pos):
      grad_r = reader_inputs[reader_key].grad[0, pos]
      embed edge:  ap = (delta_embed[pos] · grad_r).sum()
      ffn edges:   ap = (delta_ffn[L_w][pos] · grad_r).sum()  for valid L_w
      attn edges:  grad_z_r = grad_r @ W_O[L_w]               [one matmul per writer layer]
                   ap_heads = (delta_z_heads * grad_z_r_heads).sum(dim=-1)  [vectorized, all heads]
    sort all_edge_scores by |ap_recovery|, take top-k
    → PatchingResult(mode="edge", n_edges=total, n_heads=N)
  ↓
Backend: stream top-k data frames → complete frame with n_edges
  ↓
Frontend: PatchingCompleteData.summary.mode === "edge"
  → render <EdgeAttributionPanel>
    → position selector → tab bar [Sankey | Matrix | Top-list]
    → Sankey: writer axis left, reader axis right, bands by |ap_recovery|
    → Matrix: rows=writers (topo order), cols=readers
    → Top-list: ranked table
```

---

## 8. File Map

| File | Change |
|---|---|
| `testing/llm_surgeon/probe.py` | **~** extend `_capture_residual_stream_with_grad` with `capture_reader_grads` flag (6-tuple return); update `attribution_patch` + `attribution_patch_per_head` to unpack 6-tuple; add `n_edges` field to `PatchingResult`; add `edge_attribution_patch()` |
| `testing/tests/test_probe_edge_ap.py` | **new** — unit tests: reader grad capture, edge count, sum invariant, per-head decomposability, top-k, TinyLlama consistency |
| `testing/gui/backend/routes/probes.py` | **~** `"edge"` mode branch; `top_k_edges` config read; new `on_cell` signature for edge mode; `n_edges` in complete frame; extend mode validation |
| `testing/gui/frontend/src/types/api.ts` | **~** add `EdgeCellData` interface; extend `PatchingCompleteData.summary` with `n_edges?`; extend mode literal to `"edge"` |
| `testing/gui/frontend/src/components/PatchingControls.tsx` | **~** fourth mode radio (`"edge"`), `top_k_edges` input, extend `PatchingMode` type |
| `testing/gui/frontend/src/components/ProbePanel.tsx` | **~** route `mode === "edge"` to `<EdgeAttributionPanel>` |
| `testing/gui/frontend/src/components/visualizations/EdgeAttributionPanel.tsx` | **new** — Sankey, Matrix, Top-list sub-views; position selector; tab bar |
| `testing/gui/frontend/tests/e2e/smoke.spec.ts` | **+** one edge-mode smoke test |
| `testing/gui/frontend/tests/e2e/fixtures/activation-patching-edge.json` | **new** — fixture with `mode: "edge"` + edge-keyed data frames |

---

## 9. Explicit Non-Goals

1. **Per-neuron FFN decomposition** — `intermediate_size` ≈ 11008 for TinyLlama; edge count explosion.
2. **Integrated gradients** — requires multi-step interpolated forward passes. Deferred.
3. **Exact edge AP** — O(writers × positions) forward passes. Not this phase.
4. **Automatic circuit extraction (ACDC-style thresholding)** — EAP scores are an input to circuit extraction; automating threshold selection and subgraph construction is a separate phase.
5. **QK/OV circuit decomposition** — would require capturing Q/K/V projections individually. Deferred.
6. **GQA/MQA support** — TinyLlama uses standard MHA. GQA grouping would require adjusting the `concat_z` slice boundaries.
7. **Per-head FFN decomposition** — would multiply FFN writers by `intermediate_size`. Deferred.
8. **Side-by-side edge + node comparison** — separate runs in each mode allow manual comparison; simultaneous display is not in scope.
