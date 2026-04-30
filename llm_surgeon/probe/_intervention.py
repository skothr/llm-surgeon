"""Forward-pass interventions: zeroing, scaling, and activation patching."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from llm_surgeon.probe._capture import _capture_residual_stream
from llm_surgeon.probe._hooks import _get_input_device
from llm_surgeon.probe._logit_lens import _cell_metrics, _project_to_logits
from llm_surgeon.probe._types import (
    Intervention,
    InterventionResult,
    LogitLensResult,
    PatchingResult,
)

class _Op:
    """Callable wrapper with descriptive repr for experiment logging."""

    def __init__(self, fn, name: str):
        self._fn = fn
        self._name = name

    def __call__(self, hidden_state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._fn(hidden_state, layer_idx)

    def __repr__(self) -> str:
        return self._name


class _Ops:
    """Factory namespace for predefined intervention operations."""

    @staticmethod
    def scale(factor: float) -> _Op:
        return _Op(lambda h, _: h * factor, f"scale({factor})")

    @staticmethod
    def zero_dims(dims: list[int]) -> _Op:
        def fn(h, _):
            out = h.clone()
            out[:, dims] = 0
            return out
        return _Op(fn, f"zero_dims({dims})")

    @staticmethod
    def clamp(min_val: float, max_val: float) -> _Op:
        return _Op(lambda h, _: h.clamp(min=min_val, max=max_val), f"clamp({min_val}, {max_val})")

    @staticmethod
    def noise(std: float, seed: int | None = None) -> _Op:
        def fn(h, _):
            gen = torch.Generator(device=h.device)
            if seed is not None:
                gen.manual_seed(seed)
            n = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
            return h + n * std
        return _Op(fn, f"noise(std={std})")

    @staticmethod
    def replace(tensor: torch.Tensor) -> _Op:
        return _Op(lambda h, _: tensor.to(h.device), "replace(<tensor>)")

    @staticmethod
    def project_out(direction: torch.Tensor) -> _Op:
        def fn(h, _):
            d = direction.to(h.device).float()
            d = d / d.norm()
            proj = (h.float() @ d).unsqueeze(-1) * d.unsqueeze(0)
            return (h.float() - proj).to(h.dtype)
        return _Op(fn, "project_out(<direction>)")


def _apply_block_intervention(
    state: torch.Tensor,
    layer_idx: int,
    sublayer: str,
    intervention_map: dict[tuple[int, str], Callable],
    captured_states: dict[tuple[int, str], torch.Tensor] | None,
    on_layer: Callable[[int, str, dict], None] | None,
) -> tuple[torch.Tensor, bool]:
    """Apply intervention at ``(layer_idx, sublayer)`` and fire side-effects.

    Returns ``(new_state, modified)`` — the (possibly-replaced) hidden state
    cast back to the original dtype/device and a flag indicating whether the
    intervention map fired. Captures the post-intervention state into
    ``captured_states`` and invokes ``on_layer`` when supplied. The shared
    body of the per-block forward hooks; only the result-construction step
    differs between attn and ffn paths.
    """
    orig_dtype = state.dtype
    orig_device = state.device
    modified = False
    key = (layer_idx, sublayer)
    if key in intervention_map:
        state = intervention_map[key](state, layer_idx).to(
            dtype=orig_dtype, device=orig_device,
        )
        modified = True
    if captured_states is not None:
        captured_states[key] = state
    if on_layer is not None:
        on_layer(layer_idx, sublayer, {
            "hidden_state": state,
            "modified": modified,
            "top_k": None,
        })
    return state, modified


def _make_position_patch(pos: int, clean_vec: torch.Tensor) -> _Op:
    """Build an intervention op that replaces hidden_state[pos] with clean_vec,
    leaving all other positions untouched.

    Used by activation_patch() to inject a single cached activation at exactly
    one (layer, sublayer, position) triple during a corrupted (or clean) base
    forward pass. The clone avoids mutating the input tensor that intervene()'s
    hook path still references downstream.
    """
    def fn(h: torch.Tensor, _layer_idx: int) -> torch.Tensor:
        out = h.clone()
        out[pos] = clean_vec.to(device=h.device, dtype=h.dtype)
        return out
    return _Op(fn, f"patch_pos({pos})")


def intervene(
    model,
    tokenizer,
    prompt: str,
    interventions: list[Intervention],
    capture_logit_lens: bool = False,
    top_k: int = 10,
    on_layer: Callable[[int, str, dict], None] | None = None,
) -> InterventionResult:
    """Run a forward pass with hidden state modifications at specified points.

    Optionally captures logit lens data at every capture point to observe
    the downstream effect of interventions.
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)

    intervention_map: dict[tuple[int, str], Callable] = {}
    for iv in interventions:
        intervention_map[(iv.layer, iv.sublayer)] = iv.fn

    captured_states: dict[tuple[int, str], torch.Tensor] | None = (
        {} if capture_logit_lens else None
    )

    hooks = []
    layer_block_inputs: dict[int, torch.Tensor] = {}

    def make_pre(idx):
        def hook(_mod, args):
            layer_block_inputs[idx] = args[0].detach()
        return hook

    def make_attn_hook(idx):
        def hook(_mod, _inp, out):
            attn_out = out[0] if isinstance(out, tuple) else out
            state = (layer_block_inputs[idx] + attn_out.detach())[0]
            state, modified = _apply_block_intervention(
                state, idx, "attn", intervention_map, captured_states, on_layer,
            )
            if modified:
                new_attn_out = state.unsqueeze(0) - layer_block_inputs[idx]
                if isinstance(out, tuple):
                    return (new_attn_out,) + out[1:]
                return new_attn_out
        return hook

    def make_ffn_hook(idx):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            state = hidden[0].detach()
            state, modified = _apply_block_intervention(
                state, idx, "ffn", intervention_map, captured_states, on_layer,
            )
            if modified:
                new_out = state.unsqueeze(0)
                if isinstance(out, tuple):
                    return (new_out,) + out[1:]
                return new_out
        return hook

    for i in range(num_layers):
        layer = model.model.layers[i]
        hooks.append(layer.register_forward_pre_hook(make_pre(i)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(layer.register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            model_output = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    output_logits = model_output.logits[0]

    logit_lens_result = None
    if capture_logit_lens and captured_states:
        seq_len = len(prompt_tokens)
        all_positions = list(range(seq_len))
        predictions = []
        for (layer_idx, sublayer), hidden in sorted(captured_states.items()):
            with torch.no_grad():
                layer_logits = _project_to_logits(model, hidden)
            probs = F.softmax(layer_logits.float(), dim=-1)
            for pos in all_positions:
                pos_probs = probs[pos]
                metrics = _cell_metrics(pos_probs)
                topk_probs, topk_ids = pos_probs.topk(min(top_k, pos_probs.shape[0]))
                top_k_list = []
                for rank, (tid, tp) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist())):
                    top_k_list.append({
                        "token": tokenizer.decode([tid]),
                        "token_id": tid,
                        "prob": tp,
                        "rank": rank,
                    })
                predictions.append({
                    "layer": layer_idx,
                    "sublayer": sublayer,
                    "position": pos,
                    "top_k": top_k_list,
                    "metrics": metrics,
                })
        logit_lens_result = LogitLensResult(
            predictions=predictions,
            logits=None,
            prompt_tokens=prompt_tokens,
        )

    interventions_applied = [
        {"layer": iv.layer, "sublayer": iv.sublayer, "op_repr": repr(iv.fn)}
        for iv in interventions
    ]

    return InterventionResult(
        output_logits=output_logits,
        logit_lens_result=logit_lens_result,
        interventions_applied=interventions_applied,
    )


def activation_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: list[int] | None = None,
    sublayers: tuple[str, ...] = ("attn", "ffn"),
    layers: list[int] | None = None,
    on_cell: Callable[[int, str, int, dict], None] | None = None,
) -> PatchingResult:
    """Causal attribution via activation patching.

    Given two same-length prompts (clean, corrupted), computes how much each
    (layer, sublayer, position) residual-stream point causally drives the
    output delta between clean and corrupted behavior. See
    docs/superpowers/specs/2026-04-17-phase3-activation-patching-design.md.

    Args:
        direction: "denoise" (base=corrupted, patches from clean — bright cells
            are *sufficient* for clean behavior) or "noise" (base=clean, patches
            from corrupted — bright cells are *necessary* for clean behavior).
        measurement_position: absolute or negative (-1 = last) index where
            output logits are recorded. Out-of-range raises IndexError.
        positions: patch-position subset; None = all positions.
        sublayers: must be subset of {"attn", "ffn"}.
        layers: layer subset; None = all layers.
        on_cell: called with (layer, sublayer, position, cell_dict) per frame,
            before the frame is appended to the result. Used by the WS handler
            to stream cells live.

    Returns:
        PatchingResult with one cell per iterated (layer, sublayer, position).
    """
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")

    if not clean_prompt:
        raise ValueError("clean_prompt cannot be empty")
    if not corrupted_prompt:
        raise ValueError("corrupted_prompt cannot be empty")

    allowed_subs = {"attn", "ffn"}
    if not set(sublayers).issubset(allowed_subs):
        raise ValueError(f"sublayers must be subset of {allowed_subs}, got {sublayers}")

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    n_clean = clean_ids.shape[1]
    n_corr = corr_ids.shape[1]
    if n_clean != n_corr:
        raise ValueError(
            f"prompts must tokenize to same length (clean={n_clean}, corrupted={n_corr})"
        )
    seq_len = n_clean

    if measurement_position < -seq_len or measurement_position >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )
    resolved_meas = measurement_position % seq_len

    if positions is not None:
        for p in positions:
            if p < 0 or p >= seq_len:
                raise IndexError(f"position {p} out of range for seq_len={seq_len}")

    if getattr(model, "hf_quantizer", None) is not None:
        import warnings
        warnings.warn(
            "activation_patch on a quantized model: patching works but round-trips "
            "through dequant — slower and slightly less precise than fp16/fp32.",
            RuntimeWarning, stacklevel=2,
        )

    # -- Move pre-validated ids to device for the two baseline forward passes
    device = _get_input_device(model)
    clean_input_ids = clean_ids.to(device)
    corr_input_ids = corr_ids.to(device)

    # -- Capture residual streams for both prompts -------------------------
    captured_clean, prompt_tokens_clean = _capture_residual_stream(
        model, tokenizer, clean_prompt,
        sublayers=sublayers, layers=layers,
    )
    captured_corr, prompt_tokens_corrupted = _capture_residual_stream(
        model, tokenizer, corrupted_prompt,
        sublayers=sublayers, layers=layers,
    )

    # -- Baseline forward passes at measurement_position -------------------
    with torch.no_grad():
        clean_out = model(clean_input_ids)
        corr_out = model(corr_input_ids)
    clean_baseline_logits = clean_out.logits[0, resolved_meas].detach().cpu()
    corrupted_baseline_logits = corr_out.logits[0, resolved_meas].detach().cpu()

    # -- Direction selects base prompt + patch source ---------------------
    if direction == "denoise":
        base_prompt = corrupted_prompt
        patch_source = captured_clean
    else:  # "noise"
        base_prompt = clean_prompt
        patch_source = captured_corr

    # -- Resolve iteration sets -------------------------------------------
    target_positions = list(range(seq_len)) if positions is None else list(positions)

    # captured_* keys are already filtered by sublayers/layers; iterate them.
    # Sort: layer-major, attn before ffn within each layer.
    def _sort_key(k: tuple[int, str]) -> tuple[int, int]:
        return (k[0], 0 if k[1] == "attn" else 1)
    triples = sorted(patch_source.keys(), key=_sort_key)

    # -- Patching loop ----------------------------------------------------
    cells: list[dict] = []
    for (L, sub) in triples:
        patch_tensor = patch_source[(L, sub)]  # shape: (seq_len, d_model)
        for pos in target_positions:
            clean_vec = patch_tensor[pos]
            iv = Intervention(
                layer=L, sublayer=sub,
                fn=_make_position_patch(pos, clean_vec),
            )
            result = intervene(
                model, tokenizer, base_prompt,
                interventions=[iv],
                capture_logit_lens=False,
            )
            patched_logits = result.output_logits[resolved_meas].detach().cpu()
            cell: dict = {
                "layer": L,
                "sublayer": sub,
                "position": pos,
                "patched_logits": patched_logits,
            }
            if on_cell is not None:
                on_cell(L, sub, pos, cell)
            cells.append(cell)

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=prompt_tokens_clean,
        prompt_tokens_corrupted=prompt_tokens_corrupted,
        direction=direction,
        measurement_position=resolved_meas,
    )



ops = _Ops()
