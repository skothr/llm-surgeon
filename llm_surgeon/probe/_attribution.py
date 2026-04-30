"""Gradient-based attribution patching: per-cell, per-head, per-neuron, edges, circuits."""

from __future__ import annotations

from collections.abc import Callable

import torch

from llm_surgeon.probe._capture import _capture_residual_stream_with_grad
from llm_surgeon.probe._hooks import (
    _attach_reader_grad_hooks,
    _get_input_device,
)
from llm_surgeon.probe._types import PatchingResult

def _integrated_gradients_loop(
    *,
    model,
    tokenizer,
    base_prompt: str,
    base_captured: dict[tuple[int, str], torch.Tensor],
    base_h_ins: dict[int, torch.Tensor],
    from_states: dict[tuple[int, str], torch.Tensor],
    from_h_ins: dict[int, torch.Tensor],
    sublayers: tuple[str, ...],
    layers: list[int] | None,
    measurement_position: int,
    correct_token_id: int,
    incorrect_token_id: int,
    n_steps: int,
    capture_reader_grads: bool = False,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[str, int], torch.Tensor]]:
    """N forward+backward midpoint-rule Integrated Gradients over the path
    base_act → from_act at each captured (L, sub) site.

    At each step k with α_k = (k + 0.5)/N, attaches self_attn and mlp
    post-hooks that REPLACE module outputs with interpolated values
    base_component + α_k · (from_component - base_component). The native
    residual-adds then produce h_post_attn = base + α·Δh_post_attn and
    h_post_ffn = base + α·Δh_post_ffn, giving the correct IG path.
    Gradients read off the fresh leaf tensors; averaged across steps.

    Keyed components:
      - attn: base_attn_out[L] = base_captured[(L, "attn")]
              from_attn_out[L] = from_states[(L, "attn")]
      - ffn:  base_ffn_out[L]  = base_captured[(L, "ffn")] - (base_h_ins[L] + base_captured[(L, "attn")])
              (layer_output - h_post_attn = ffn_out, assuming attn row exists)
              Analogous for from_ side.

    When "attn" not in sublayers we cannot reconstruct ffn_out without h_in,
    so ffn-only IG is not supported; an empty avg_grad is returned for ffn
    rows in that case.
    """
    num_layers = len(model.model.layers)
    target_layers_set = (
        set(range(num_layers)) if layers is None else set(layers)
    )
    need_attn = "attn" in sublayers
    need_ffn = "ffn" in sublayers

    base_attn: dict[int, torch.Tensor] = {}
    base_ffn: dict[int, torch.Tensor] = {}
    from_attn: dict[int, torch.Tensor] = {}
    from_ffn: dict[int, torch.Tensor] = {}
    for L in sorted(target_layers_set):
        if need_attn:
            base_attn[L] = base_captured[(L, "attn")].detach()
            from_attn[L] = from_states[(L, "attn")]
        if need_ffn and need_attn:
            b_layer = base_captured[(L, "ffn")].detach()
            b_hpa = base_h_ins[L].detach() + base_captured[(L, "attn")].detach()
            base_ffn[L] = b_layer - b_hpa
            f_layer = from_states[(L, "ffn")]
            f_hpa = from_h_ins[L] + from_states[(L, "attn")]
            from_ffn[L] = f_layer - f_hpa

    grad_sum_attn: dict[int, torch.Tensor] = {
        L: torch.zeros_like(base_attn[L]) for L in base_attn
    }
    grad_sum_ffn: dict[int, torch.Tensor] = {
        L: torch.zeros_like(base_ffn[L]) for L in base_ffn
    }
    grad_sum_reader: dict[tuple[str, int], torch.Tensor] = {}

    device = _get_input_device(model)
    enc = tokenizer(base_prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps

        interp_attn: dict[int, torch.Tensor] = {}
        interp_ffn: dict[int, torch.Tensor] = {}
        for L in base_attn:
            t = base_attn[L] + alpha * (from_attn[L] - base_attn[L])
            t = t.detach().clone().requires_grad_(True)
            interp_attn[L] = t
        for L in base_ffn:
            t = base_ffn[L] + alpha * (from_ffn[L] - base_ffn[L])
            t = t.detach().clone().requires_grad_(True)
            interp_ffn[L] = t

        hooks: list = []
        step_readers: dict[tuple[str, int], torch.Tensor] = {}

        def make_attn_replace(L_captured: int):
            def hook(_mod, _inp, out):
                new = interp_attn[L_captured]
                if isinstance(out, tuple):
                    return (new,) + tuple(out[1:])
                return new
            return hook

        def make_mlp_replace(L_captured: int):
            def hook(_mod, _inp, _out):
                return interp_ffn[L_captured]
            return hook

        for L in interp_attn:
            hooks.append(
                model.model.layers[L].self_attn.register_forward_hook(
                    make_attn_replace(L)
                )
            )
        for L in interp_ffn:
            hooks.append(
                model.model.layers[L].mlp.register_forward_hook(
                    make_mlp_replace(L)
                )
            )

        if capture_reader_grads:
            hooks.extend(_attach_reader_grad_hooks(model, step_readers))

        try:
            out = model(input_ids)
            step_logits = out.logits[0]
            step_metric = (
                step_logits[measurement_position, correct_token_id]
                - step_logits[measurement_position, incorrect_token_id]
            )
            step_metric.backward()
        finally:
            for h in hooks:
                h.remove()

        for L, t in interp_attn.items():
            if t.grad is not None:
                grad_sum_attn[L] += t.grad.detach()
        for L, t in interp_ffn.items():
            if t.grad is not None:
                grad_sum_ffn[L] += t.grad.detach()

        if capture_reader_grads:
            for reader_key, reader_tensor in step_readers.items():
                if reader_tensor.grad is not None:
                    if reader_key not in grad_sum_reader:
                        grad_sum_reader[reader_key] = torch.zeros_like(reader_tensor.grad.detach())
                    grad_sum_reader[reader_key] += reader_tensor.grad.detach()

    avg_grad: dict[tuple[int, str], torch.Tensor] = {}
    for L in grad_sum_attn:
        avg_grad[(L, "attn")] = grad_sum_attn[L] / n_steps
    for L in grad_sum_ffn:
        avg_grad[(L, "ffn")] = grad_sum_ffn[L] / n_steps

    avg_reader_grads: dict[tuple[str, int], torch.Tensor] = {}
    if capture_reader_grads:
        for reader_key, grad_sum in grad_sum_reader.items():
            avg_reader_grads[reader_key] = grad_sum / n_steps

    return avg_grad, avg_reader_grads


def attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: list[int] | None = None,
    sublayers: tuple[str, ...] = ("attn", "ffn"),
    layers: list[int] | None = None,
    n_steps: int = 1,
    on_cell: Callable[[int, str, int, dict], None] | None = None,
) -> PatchingResult:
    """Gradient-based attribution patching.

    One forward + one backward pass produces a per-cell AP score that
    approximates exact activation_patch's logit_diff_recovery. Much cheaper
    than the O(L·S·P) exact loop.

    See: Nanda 2023 (attribution patching primer) and Kramár et al. 2024
    (Attribution Patching Outperforms Automated Circuit Discovery).
    """
    import warnings

    # --- Validation (raise before any forward pass) ---
    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not set(sublayers).issubset({"attn", "ffn"}):
        raise ValueError("sublayers must be subset of {'attn', 'ffn'}")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")

    if getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            "attribution_patch on a quantized model: gradient flow works but "
            "precision is reduced (fp16/int8 through bitsandbytes).",
            stacklevel=2,
        )

    # --- Tokenize + length check ---
    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    if positions is None:
        positions = list(range(seq_len))
    for pos in positions:
        if pos < -seq_len or pos >= seq_len:
            raise IndexError(f"position {pos} out of range for seq_len={seq_len}")
    normalized_positions: list[int] = [p if p >= 0 else seq_len + p for p in positions]
    meas_pos = measurement_position if measurement_position >= 0 else seq_len + measurement_position
    if meas_pos < 0 or meas_pos >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )

    # --- Step 1: Forward 'from' prompt in no_grad to cache activations ---
    # denoise: from=clean, base=corrupted
    # noise:   from=corrupted, base=clean
    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    with torch.no_grad():
        from_captured, from_h_ins_raw, from_logits, from_tokens, _, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt, sublayers=sublayers, layers=layers,
            )
        # Detach + clone so the 'from' tensors don't pollute the upcoming
        # base-side graph. They're used purely as values.
        from_states = {k: v.detach().clone() for k, v in from_captured.items()}
        from_h_ins = {idx: v.detach().clone() for idx, v in from_h_ins_raw.items()}

    # --- Step 2: Forward 'base' prompt WITH grad to build the graph ---
    with torch.enable_grad():
        base_captured, base_h_ins, base_logits, base_tokens, _, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt, sublayers=sublayers, layers=layers,
            )

        # base_logits already has the base-prompt logits — reuse them.
        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()

        denominator = (d_clean - d_corrupted).item()
        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        if n_steps == 1:
            # --- Step 3: Metric scalar on base-side logits, backward ---
            metric = (
                base_logits[meas_pos, correct_token_id]
                - base_logits[meas_pos, incorrect_token_id]
            )
            metric.backward()
            avg_grad: dict[tuple[int, str], torch.Tensor] | None = None  # unused on this path
        else:
            avg_grad, _ = _integrated_gradients_loop(
                model=model,
                tokenizer=tokenizer,
                base_prompt=base_prompt,
                base_captured=base_captured,
                base_h_ins=base_h_ins,
                from_states=from_states,
                from_h_ins=from_h_ins,
                sublayers=sublayers,
                layers=layers,
                measurement_position=meas_pos,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                n_steps=n_steps,
            )

    # --- Step 4: Compute AP per cell ---
    # Captured tensors have shape (1, seq_len, d_model); access batch dim [0].
    # For attn rows, reconstruct h_post_attn = h_in + attn_out to match exact
    # AP's patched quantity. Gradient of h_post_attn equals gradient of attn_out
    # (chain rule through the `+`), so base_act.grad is correct either way.
    cells: list[dict] = []
    sorted_keys = sorted(base_captured.keys(), key=lambda k: (k[0], k[1]))
    for (L, sub) in sorted_keys:
        base_act = base_captured[(L, sub)]  # (1, seq_len, d_model)
        if n_steps == 1:
            if base_act.grad is None:
                continue  # shouldn't happen after backward but guard defensively
            base_grad = base_act.grad  # (1, seq_len, d_model)
        else:
            assert avg_grad is not None
            if (L, sub) not in avg_grad:
                continue
            base_grad = avg_grad[(L, sub)]
        from_act = from_states[(L, sub)]  # (1, seq_len, d_model)
        if sub == "attn":
            # Residual-stream value = h_in + attn_out
            from_val_full = from_h_ins[L] + from_act
            base_val_full = base_h_ins[L].detach() + base_act.detach()
        else:  # ffn — captured tensor is already the residual stream post-layer
            from_val_full = from_act
            base_val_full = base_act.detach()
        for pos in normalized_positions:
            ap_raw = (
                (from_val_full[0, pos] - base_val_full[0, pos]) * base_grad[0, pos]
            ).sum().item()
            if direction == "denoise":
                ap_recovery = ap_raw / denominator
            else:  # noise
                ap_recovery = 1.0 + ap_raw / denominator
            cell: dict = {
                "layer": L,
                "sublayer": sub,
                "position": pos,
                "ap_recovery": float(ap_recovery),
            }
            cells.append(cell)
            if on_cell is not None:
                on_cell(L, sub, pos, cell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx",
        n_steps=(n_steps if n_steps > 1 else None),
    )


def attribution_patch_per_head(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    n_steps: int = 1,
    on_cell: Callable[[int, str, int, dict], None] | None = None,
) -> PatchingResult:
    """Per-attention-head gradient attribution patching.

    Decomposes attn_out's contribution to the metric into per-head scores via
    chain rule through W_O (o_proj). Produces per-(layer, head, position) AP
    values plus FFN anchor rows. One forward + one backward pass, same cost as
    the per-cell variant.

    Unit strings in on_cell / cells: "attn.h{N}" (0-indexed) for head N,
    "ffn" for FFN anchor.

    Note: sum_h AP_head(L,h,pos) == (delta_attn_out · attn_out.grad) / D, which
    equals the per-cell AP_attn(L,pos) ONLY when h_in is identical between the
    clean and corrupted prompts (trivially at L=0 for same-length tokenizations
    but not at deeper layers). Per-cell AP_attn linearizes at the full residual
    stream h_post_attn = h_in + attn_out to match exact AP's patched quantity;
    per-head AP decomposes attn_out alone, which is the right unit for
    mechanistic interpretability of individual heads.
    """
    import warnings

    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch_per_head requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")

    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")

    if getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            "attribution_patch_per_head on a quantized model: gradient flow works "
            "but precision is reduced.",
            stacklevel=2,
        )

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    normalized_positions: list[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )
    meas_pos = measurement_position % seq_len

    n_heads: int = model.config.num_attention_heads
    hidden: int = model.config.hidden_size
    head_dim: int = hidden // n_heads

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt
    sublayers: tuple[str, ...] = ("attn", "ffn")

    with torch.no_grad():
        from_captured, from_h_ins_raw, from_logits, from_tokens, from_concat_z_raw, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
            )
        from_states = {k: v.detach().clone() for k, v in from_captured.items()}
        from_h_ins = {idx: v.detach().clone() for idx, v in from_h_ins_raw.items()}
        from_concat_z = {i: v.detach().clone() for i, v in from_concat_z_raw.items()}

    with torch.enable_grad():
        base_captured, base_h_ins, base_logits, base_tokens, base_concat_z, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
            )

        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()
        denominator = (d_clean - d_corrupted).item()
        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        if n_steps == 1:
            metric.backward()
            avg_grad_head: dict[tuple[int, str], torch.Tensor] | None = None
        else:
            avg_grad_head, _ = _integrated_gradients_loop(
                model=model,
                tokenizer=tokenizer,
                base_prompt=base_prompt,
                base_captured=base_captured,
                base_h_ins=base_h_ins,
                from_states=from_states,
                from_h_ins=from_h_ins,
                sublayers=sublayers,
                layers=layers,
                measurement_position=meas_pos,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                n_steps=n_steps,
            )

    num_layers = len(model.model.layers)
    target_layers = sorted(
        set(range(num_layers)) if layers is None else set(layers)
    )

    cells: list[dict] = []

    for L in target_layers:
        # --- FFN anchor (identical math to per-cell AP) ---
        if (L, "ffn") in base_captured:
            base_ffn = base_captured[(L, "ffn")]    # [1, seq, hidden]
            ffn_grad = base_ffn.grad if n_steps == 1 else (avg_grad_head.get((L, "ffn")) if avg_grad_head is not None else None)
            from_ffn = from_states.get((L, "ffn"))
            if ffn_grad is not None and from_ffn is not None:
                for pos in normalized_positions:
                    ap_raw = (
                        (from_ffn[0, pos] - base_ffn[0, pos].detach()) * ffn_grad[0, pos]
                    ).sum().item()
                    ap_recovery = ap_raw / denominator if direction == "denoise" else 1.0 + ap_raw / denominator
                    cell: dict = {"layer": L, "unit": "ffn", "position": pos,
                                  "ap_recovery": float(ap_recovery)}
                    cells.append(cell)
                    if on_cell is not None:
                        on_cell(L, "ffn", pos, cell)

        # --- Per-head AP via chain rule through W_O ---
        if (L, "attn") in base_captured and L in base_concat_z:
            attn_out_grad = (
                base_captured[(L, "attn")].grad if n_steps == 1
                else (avg_grad_head.get((L, "attn")) if avg_grad_head is not None else None)
            )    # [1, seq, hidden]
            if attn_out_grad is None:
                continue
            W_O: torch.Tensor = model.model.layers[L].self_attn.o_proj.weight  # [hidden, hidden]
            # Chain rule: ∂metric/∂concat_z = attn_out_grad @ W_O
            # attn_out_grad[0]: [seq, hidden]; W_O: [hidden, hidden]
            concat_z_grad = attn_out_grad[0] @ W_O             # [seq, hidden]

            base_cz = base_concat_z[L]              # [1, seq, hidden], in graph
            from_cz = from_concat_z.get(L)
            if from_cz is None:
                continue

            for pos in normalized_positions:
                delta_z = (from_cz[0, pos] - base_cz[0, pos].detach())      # [hidden]
                cz_grad_pos = concat_z_grad[pos]                              # [hidden]

                dz_heads = delta_z.view(n_heads, head_dim)           # [n_heads, head_dim]
                cz_grad_heads = cz_grad_pos.view(n_heads, head_dim)  # [n_heads, head_dim]
                ap_heads_raw = (dz_heads * cz_grad_heads).sum(dim=-1)  # [n_heads]

                for h in range(n_heads):
                    ap_raw_h = ap_heads_raw[h].item()
                    ap_recovery_h = (
                        ap_raw_h / denominator
                        if direction == "denoise"
                        else 1.0 + ap_raw_h / denominator
                    )
                    unit = f"attn.h{h}"
                    hcell: dict = {"layer": L, "unit": unit, "position": pos,
                                   "ap_recovery": float(ap_recovery_h)}
                    cells.append(hcell)
                    if on_cell is not None:
                        on_cell(L, unit, pos, hcell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx_head",
        n_heads=n_heads,
        n_steps=(n_steps if n_steps > 1 else None),
    )


def attribution_patch_per_neuron(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_k_neurons: int = 200,
    n_steps: int = 1,
    on_cell: Callable[[dict], None] | None = None,
) -> PatchingResult:
    """Per-neuron FFN attribution patching.

    Decomposes Δffn_out's contribution to the metric into per-
    (layer, neuron, position) AP scores via chain rule through W_down.
    One forward + one backward pass. For each layer, each FFN output
    position, and each neuron index i in [0, intermediate_size):

        grad_act = grad_ffn_out @ W_down             # [intermediate]
        delta_act = from_act[pos] - base_act[pos]    # [intermediate]
        ap_raw[i] = delta_act[i] * grad_act[i]
        ap_recovery[i] = ap_raw[i] / D  (denoise) or 1 + ap_raw[i]/D (noise)

    Returns PatchingResult with mode='approx_neuron',
    n_neurons=intermediate_size, and `cells` containing only the top-k
    tuples by |ap_recovery|. If top_k_neurons exceeds the total
    neuron-cell count, silently caps.
    """
    if top_k_neurons < 1:
        raise ValueError("top_k_neurons must be >= 1")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompts cannot be empty")
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    meas_pos = measurement_position % seq_len

    normalized_positions: list[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    sublayers: tuple[str, ...] = ("attn", "ffn")

    # --- From pass (no_grad, capture ffn_act + ffn_out) ---
    with torch.no_grad():
        from_captured_raw, from_h_ins_raw, from_logits, from_tokens, _, _, from_ffn_acts_raw = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=False,
                capture_reader_grads=False,
                capture_ffn_out=True,
                capture_ffn_act=True,
            )
        from_ffn_acts = {k: v.detach().clone() for k, v in from_ffn_acts_raw.items()}
        from_states_neuron = {k: v.detach().clone() for k, v in from_captured_raw.items()}
        from_h_ins_neuron = {idx: v.detach().clone() for idx, v in from_h_ins_raw.items()}

    # --- Base pass (enable_grad, backward through metric) ---
    with torch.enable_grad():
        base_captured, base_h_ins_neuron, base_logits, base_tokens, _, _, base_ffn_acts = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=False,
                capture_reader_grads=False,
                capture_ffn_out=True,
                capture_ffn_act=True,
            )

        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()
        denominator = (d_clean - d_corrupted).item()

        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        if n_steps == 1:
            metric.backward()
            avg_grad_neuron: dict[tuple[int, str], torch.Tensor] | None = None
        else:
            avg_grad_neuron, _ = _integrated_gradients_loop(
                model=model,
                tokenizer=tokenizer,
                base_prompt=base_prompt,
                base_captured=base_captured,
                base_h_ins=base_h_ins_neuron,
                from_states=from_states_neuron,
                from_h_ins=from_h_ins_neuron,
                sublayers=sublayers,
                layers=layers,
                measurement_position=meas_pos,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                n_steps=n_steps,
            )

    intermediate_size: int = model.config.intermediate_size
    num_layers = len(model.model.layers)
    target_layers_set = set(range(num_layers)) if layers is None else set(layers)

    all_cells: list[dict] = []
    for L in sorted(target_layers_set):
        if n_steps == 1:
            if (L, "ffn_out") not in base_captured:
                continue
            base_ffn_out = base_captured[(L, "ffn_out")]
            if base_ffn_out.grad is None:
                continue
            grad_ffn_out_tensor: torch.Tensor = base_ffn_out.grad
        else:
            assert avg_grad_neuron is not None
            maybe_grad = avg_grad_neuron.get((L, "ffn"))
            if maybe_grad is None:
                continue
            grad_ffn_out_tensor = maybe_grad
        if L not in base_ffn_acts or L not in from_ffn_acts:
            continue
        W_down: torch.Tensor = model.model.layers[L].mlp.down_proj.weight  # [hidden, intermediate]
        from_act = from_ffn_acts[L]
        base_act_L = base_ffn_acts[L]

        for pos in normalized_positions:
            grad_ffn_out = grad_ffn_out_tensor[0, pos].detach()          # [hidden]
            grad_act = grad_ffn_out @ W_down                             # [intermediate]
            delta_act = from_act[0, pos] - base_act_L[0, pos].detach()   # [intermediate]
            ap_raw = (delta_act * grad_act)                              # [intermediate]
            if direction == "denoise":
                ap_recovery = ap_raw / denominator
            else:
                ap_recovery = 1.0 + ap_raw / denominator

            ap_recovery_cpu = ap_recovery.detach().cpu().tolist()
            for i in range(intermediate_size):
                all_cells.append({
                    "layer": L,
                    "unit": f"neuron.n{i}",
                    "neuron": i,
                    "position": pos,
                    "ap_recovery": float(ap_recovery_cpu[i]),
                })

    all_cells.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_cells[:top_k_neurons]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=top_cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx_neuron",
        n_neurons=intermediate_size,
        n_steps=(n_steps if n_steps > 1 else None),
    )


def _is_valid_attn_writer(L_w: int, reader_type: str, reader_L: int) -> bool:
    """True when attn writer at L_w can causally precede reader of type reader_type at reader_L."""
    if reader_type == "attn_in":
        return L_w < reader_L
    if reader_type == "ffn_in":
        return L_w <= reader_L  # same-layer attn → same-layer ffn_in is valid
    if reader_type == "logits":
        return True
    return False


def _is_valid_ffn_writer(L_w: int, reader_type: str, reader_L: int) -> bool:
    """True when FFN writer at L_w can causally precede reader of type reader_type at reader_L."""
    if reader_type in ("attn_in", "ffn_in"):
        return L_w < reader_L
    if reader_type == "logits":
        return True
    return False


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
    positions: list[int] | None,
    layers: list[int] | None,
    n_steps: int = 1,
) -> tuple[
    list[dict],            # all_edge_scores (unsorted)
    torch.Tensor,          # clean_baseline_logits (detached)
    torch.Tensor,          # corrupted_baseline_logits (detached)
    list[str],             # clean_tokens (ordered by direction)
    list[str],             # corrupted_tokens (ordered by direction)
    int,                   # meas_pos (normalized to [0, seq_len))
    int,                   # n_heads
]:
    """Core forward+backward+edge-enumeration shared by edge_attribution_patch
    and extract_circuit.

    Validates prompts/direction/top-k-candidate preconditions upstream
    (each caller is responsible for its own param validation).
    """
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompts cannot be empty")
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")

    device = _get_input_device(model)

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    meas_pos = measurement_position % seq_len

    normalized_positions: list[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    sublayers: tuple[str, ...] = ("attn", "ffn")

    with torch.no_grad():
        from_embed = model.model.embed_tokens(
            tokenizer(from_prompt, return_tensors="pt")["input_ids"].to(device)
        ).detach()
        base_embed = model.model.embed_tokens(
            tokenizer(base_prompt, return_tensors="pt")["input_ids"].to(device)
        ).detach()
    delta_embed = from_embed - base_embed

    with torch.no_grad():
        from_captured_raw, from_h_ins_raw, from_logits, from_tokens, from_cz_raw, _, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
                capture_reader_grads=False,
                capture_ffn_out=True,
            )
        from_states = {k: v.detach().clone() for k, v in from_captured_raw.items()}
        from_cz = {k: v.detach().clone() for k, v in from_cz_raw.items()}
        from_h_ins = {idx: v.detach().clone() for idx, v in from_h_ins_raw.items()}

    with torch.enable_grad():
        base_captured, base_h_ins, base_logits, base_tokens, base_cz, reader_inputs, _ = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
                capture_reader_grads=True,
                capture_ffn_out=True,
            )

        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()
        denominator = (d_clean - d_corrupted).item()

        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        avg_reader_grads: dict[tuple[str, int], torch.Tensor] = {}
        if n_steps == 1:
            metric = (
                base_logits[meas_pos, correct_token_id]
                - base_logits[meas_pos, incorrect_token_id]
            )
            metric.backward()
        else:
            _, avg_reader_grads = _integrated_gradients_loop(
                model=model,
                tokenizer=tokenizer,
                base_prompt=base_prompt,
                base_captured=base_captured,
                base_h_ins=base_h_ins,
                from_states=from_states,
                from_h_ins=from_h_ins,
                sublayers=sublayers,
                layers=layers,
                measurement_position=meas_pos,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                n_steps=n_steps,
                capture_reader_grads=True,
            )

    delta_ffn: dict[int, torch.Tensor] = {}
    num_layers = len(model.model.layers)
    target_layers_set = set(range(num_layers)) if layers is None else set(layers)
    for L in sorted(target_layers_set):
        from_ffn_out = from_states.get((L, "ffn_out"))
        base_ffn_out = base_captured.get((L, "ffn_out"))
        if from_ffn_out is not None and base_ffn_out is not None:
            delta_ffn[L] = from_ffn_out - base_ffn_out.detach()

    n_heads: int = model.config.num_attention_heads
    hidden: int = model.config.hidden_size
    head_dim: int = hidden // n_heads

    all_edge_scores: list[dict] = []

    for reader_key, reader_tensor in reader_inputs.items():
        reader_type = reader_key[0]
        reader_L = reader_key[1]

        maybe_grad = (
            reader_tensor.grad if n_steps == 1
            else avg_reader_grads.get(reader_key)
        )
        if maybe_grad is None:
            continue
        grad_r_full: torch.Tensor = maybe_grad

        for pos in normalized_positions:
            grad_r = grad_r_full[0, pos].detach() if n_steps == 1 else grad_r_full[0, pos]

            ap_embed_raw = (delta_embed[0, pos] * grad_r).sum().item()
            ap_recovery_embed = (
                ap_embed_raw / denominator
                if direction == "denoise"
                else 1.0 + ap_embed_raw / denominator
            )
            all_edge_scores.append({
                "writer_layer": 0,
                "writer_unit": "embed",
                "reader_layer": reader_L,
                "reader_unit": reader_type,
                "position": pos,
                "ap_recovery": float(ap_recovery_embed),
            })

            for L_w in sorted(target_layers_set):
                if _is_valid_ffn_writer(L_w, reader_type, reader_L) and L_w in delta_ffn:
                    ap_ffn_raw = (delta_ffn[L_w][0, pos] * grad_r).sum().item()
                    ap_recovery_ffn = (
                        ap_ffn_raw / denominator
                        if direction == "denoise"
                        else 1.0 + ap_ffn_raw / denominator
                    )
                    all_edge_scores.append({
                        "writer_layer": L_w,
                        "writer_unit": "ffn",
                        "reader_layer": reader_L,
                        "reader_unit": reader_type,
                        "position": pos,
                        "ap_recovery": float(ap_recovery_ffn),
                    })

                if _is_valid_attn_writer(L_w, reader_type, reader_L):
                    if (L_w, "attn") in from_states and L_w in base_cz and L_w in from_cz:
                        W_O: torch.Tensor = model.model.layers[L_w].self_attn.o_proj.weight
                        grad_z_r = grad_r @ W_O
                        grad_z_r_heads = grad_z_r.view(n_heads, head_dim)

                        delta_z = from_cz[L_w][0, pos] - base_cz[L_w][0, pos].detach()
                        delta_z_heads = delta_z.view(n_heads, head_dim)
                        ap_heads_raw = (delta_z_heads * grad_z_r_heads).sum(dim=-1)

                        for h in range(n_heads):
                            ap_h_raw = ap_heads_raw[h].item()
                            ap_recovery_h = (
                                ap_h_raw / denominator
                                if direction == "denoise"
                                else 1.0 + ap_h_raw / denominator
                            )
                            all_edge_scores.append({
                                "writer_layer": L_w,
                                "writer_unit": f"attn.h{h}",
                                "reader_layer": reader_L,
                                "reader_unit": reader_type,
                                "position": pos,
                                "ap_recovery": float(ap_recovery_h),
                            })

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return (
        all_edge_scores,
        clean_baseline.detach(),
        corrupted_baseline.detach(),
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    )


def edge_attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_k_edges: int = 200,
    n_steps: int = 1,
    on_cell: Callable[[dict], None] | None = None,
) -> PatchingResult:
    """Per-edge gradient attribution patching.

    Decomposes the residual stream's additive structure into per-(writer, reader,
    position) AP scores. One forward + one backward pass. Each edge score measures
    how much writer w's output delta (clean - corrupted) aligned with the gradient
    at reader r's input.

    Valid edges respect the residual stream's topological order:
    - embed → any reader
    - attn(L_w) → attn_in(L_r) iff L_w < L_r
    - attn(L_w) → ffn_in(L_r) iff L_w <= L_r
    - ffn(L_w) → attn_in(L_r) or ffn_in(L_r) iff L_w < L_r
    - any writer → logits reader

    Returns PatchingResult with mode="edge", n_edges=total_pre_filter_count.
    cells contains only the top-k edges by |ap_recovery|.
    """
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1")
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")

    (
        all_edge_scores,
        clean_baseline_logits,
        corrupted_baseline_logits,
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    ) = _compute_all_edges(
        model, tokenizer, clean_prompt, corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
    )

    n_edges_total = len(all_edge_scores)
    all_edge_scores.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_edge_scores[:top_k_edges]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    result = PatchingResult(
        cells=top_cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="edge",
        n_heads=n_heads,
        n_edges=n_edges_total,
    )
    result.n_steps = n_steps if n_steps > 1 else None
    return result


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
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    tau: float = 0.02,
    top_k_candidates: int = 2000,
    n_steps: int = 1,
    on_cell: Callable[[dict], None] | None = None,
) -> PatchingResult:
    """Cheap-ACDC circuit extraction (Syed et al. 2023, arXiv 2310.10348).

    Runs the edge attribution pass, then annotates the top
    `top_k_candidates` edges with `in_circuit: bool` based on:
      1. |ap_recovery| >= tau (filter)
      2. reader is reverse-reachable from 'logits' through surviving edges

    Returns PatchingResult with mode='circuit'. Cells include all top-k
    candidates (in-circuit and out). Summary fields:
      n_edges              - total pre-filter edge count
      n_edges_in_circuit   - count of cells with in_circuit=True
      n_nodes_in_circuit   - |visited| from reverse-BFS, inclusive of the
                              logits sink (a graph with only embed->logits
                              yields n_nodes_in_circuit == 2).
      tau                  - applied threshold

    If `top_k_candidates > total valid edges`, silently caps at the actual
    edge count (matches edge_attribution_patch's top_k_edges behavior).
    """
    if tau < 0.0:
        raise ValueError("tau must be >= 0.0")
    if top_k_candidates < 1:
        raise ValueError("top_k_candidates must be >= 1")
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")

    (
        all_edge_scores,
        clean_baseline_logits,
        corrupted_baseline_logits,
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    ) = _compute_all_edges(
        model, tokenizer, clean_prompt, corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
    )

    n_edges_total = len(all_edge_scores)
    all_edge_scores.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_edge_scores[:top_k_candidates]

    def node_of_writer(cell: dict) -> tuple[int, str, int]:
        return (cell["writer_layer"], cell["writer_unit"], cell["position"])

    def node_of_reader(cell: dict) -> tuple[int, str, int]:
        return (cell["reader_layer"], cell["reader_unit"], cell["position"])

    reverse_adj: dict[tuple[int, str, int], list[tuple[int, str, int]]] = {}
    for cell in top_cells:
        if abs(cell["ap_recovery"]) < tau:
            continue
        r_node = node_of_reader(cell)
        w_node = node_of_writer(cell)
        reverse_adj.setdefault(r_node, []).append(w_node)

    visited: set[tuple[int, str, int]] = set()
    queue: list[tuple[int, str, int]] = []
    for node in reverse_adj.keys():
        if node[1] == "logits":
            visited.add(node)
            queue.append(node)

    while queue:
        r = queue.pop()
        for w in reverse_adj.get(r, []):
            if w not in visited:
                visited.add(w)
                queue.append(w)

    n_edges_in_circuit = 0
    for cell in top_cells:
        if abs(cell["ap_recovery"]) >= tau and node_of_reader(cell) in visited:
            cell["in_circuit"] = True
            n_edges_in_circuit += 1
        else:
            cell["in_circuit"] = False

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    result = PatchingResult(
        cells=top_cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="circuit",
        n_heads=n_heads,
        n_edges=n_edges_total,
        n_edges_in_circuit=n_edges_in_circuit,
        n_nodes_in_circuit=len(visited),
        tau=tau,
    )
    result.n_steps = n_steps if n_steps > 1 else None
    return result

