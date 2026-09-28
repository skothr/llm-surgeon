"""Gradient-based attribution patching: per-cell, per-head, per-neuron, edges, circuits."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field

import torch

from llm_surgeon.probe._capture import _capture_residual_stream_with_grad
from llm_surgeon.probe._hooks import (
    _attach_reader_grad_hooks,
    _get_input_device,
    _unwrap_hook_output,
)
from llm_surgeon.probe._types import PatchingResult


def _grads_of(
    metric: torch.Tensor,
    tensors: Mapping[Hashable, torch.Tensor],
) -> dict[Hashable, torch.Tensor]:
    """Return d(metric)/d(tensor) for each tensor that is in the graph.

    Uses ``torch.autograd.grad`` rather than ``metric.backward()`` so no
    gradient is ever accumulated into the model's parameter ``.grad``
    fields: those would leak across calls and cost a full parameter-sized
    gradient set of memory.
    """
    keys = [k for k, t in tensors.items() if t.requires_grad]
    if not keys:
        return {}
    grads = torch.autograd.grad(
        metric,
        [tensors[k] for k in keys],
        allow_unused=True,
    )
    return {k: g.detach() for k, g in zip(keys, grads) if g is not None}


def _make_path_hook(value: torch.Tensor):
    """Forward hook forcing a module's output to ``value`` while keeping its Jacobian.

    The returned output is ``value + (out - out.detach())``: numerically
    ``value``, but gradients still flow from it back through the module to
    the module's input. A gradient read off ``value`` is therefore the total
    derivative of the metric, including every path through downstream
    attention and MLP sublayers, not only the residual skip.
    """

    def hook(_mod, _inp, out):
        o = _unwrap_hook_output(out)
        new = value + (o - o.detach())
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new

    return hook


def _integrated_gradients_loop(
    *,
    model,
    input_ids: torch.Tensor,
    base_components: dict[tuple[int, str], torch.Tensor],
    from_components: dict[tuple[int, str], torch.Tensor],
    measurement_position: int,
    correct_token_id: int,
    incorrect_token_id: int,
    n_steps: int,
    reader_layers: list[int] | None = None,
    capture_reader_grads: bool = False,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[str, int], torch.Tensor]]:
    """Midpoint-rule Integrated Gradients over the sublayer outputs.

    ``base_components`` / ``from_components`` hold the sublayer outputs of
    the base and from prompts, keyed ``(L, "attn")`` (self_attn output) or
    ``(L, "ffn")`` (mlp output). At step k, with α_k = (k + 0.5)/N, a
    forward of the base prompt runs with every keyed sublayer's output
    forced to ``base + α_k · (from - base)``. Embeddings and sublayers not
    in the dicts are computed from the base prompt as usual.

    The forcing keeps each sublayer's local Jacobian (see
    ``_make_path_hook``), so the gradient at a site includes its effect
    through every downstream attention and MLP sublayer. The gradient of a
    sublayer output equals the gradient of the residual stream just after
    that sublayer's residual add, so the averages serve both the sublayer
    and the residual-stream rows.

    With one keyed sublayer this is the exact IG path of patching that
    sublayer: Σ Δ·avg_grad converges to the activation-patching effect.
    With several keyed sublayers the forced values lie on the joint
    straight line, and each gradient uses the downstream sublayers' local
    Jacobians at those values. With from == base every average equals the
    plain gradient at the base forward.

    Returns ``(avg_grad, avg_reader_grads)``; the reader grads are keyed
    like ``_attach_reader_grad_hooks`` and are empty unless
    ``capture_reader_grads``.
    """
    grad_sum: dict[tuple[int, str], torch.Tensor] = {
        key: torch.zeros_like(t) for key, t in base_components.items()
    }
    reader_sum: dict[tuple[str, int], torch.Tensor] = {}

    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        interp: dict[tuple[int, str], torch.Tensor] = {}
        for key, base_t in base_components.items():
            t = base_t + alpha * (from_components[key] - base_t)
            interp[key] = t.detach().clone().requires_grad_(True)

        step_readers: dict[tuple, torch.Tensor] = {}
        hooks: list = []
        try:
            for (L, sub), leaf in interp.items():
                layer = model.model.layers[L]
                module = layer.self_attn if sub == "attn" else layer.mlp
                hooks.append(module.register_forward_hook(_make_path_hook(leaf)))
            if capture_reader_grads:
                hooks.extend(
                    _attach_reader_grad_hooks(model, step_readers, layers=reader_layers)
                )
            with torch.enable_grad():
                step_logits = model(input_ids).logits[0]
                step_metric = (
                    step_logits[measurement_position, correct_token_id]
                    - step_logits[measurement_position, incorrect_token_id]
                )
                targets: dict[Hashable, torch.Tensor] = {
                    ("site", key): t for key, t in interp.items()
                }
                targets.update({("reader", key): t for key, t in step_readers.items()})
                step_grads = _grads_of(step_metric, targets)
        finally:
            for h in hooks:
                h.remove()

        for (kind, key), g in step_grads.items():  # pyright: ignore[reportGeneralTypeIssues]
            if kind == "site":
                grad_sum[key] += g
            elif key in reader_sum:
                reader_sum[key] += g
            else:
                reader_sum[key] = g.clone()

    avg_grad = {key: g / n_steps for key, g in grad_sum.items()}
    avg_reader_grads = {key: g / n_steps for key, g in reader_sum.items()}
    return avg_grad, avg_reader_grads


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


@dataclass
class _APSetup:
    """Everything the attribution entry points share after the two forwards.

    Captured tensors are detached values. ``grads`` holds the metric
    gradient (plain at the base point when n_steps == 1, IG-averaged
    otherwise) keyed ``(L, "attn")`` (self_attn output, equivalently the
    residual stream after the attention add) and ``(L, "ffn")`` (layer
    output, equivalently the mlp output). ``reader_grads`` is keyed like
    ``_attach_reader_grad_hooks``.
    """

    direction: str
    denominator: float
    meas_pos: int
    positions: list[int]
    layers: list[int]
    clean_baseline: torch.Tensor
    corrupted_baseline: torch.Tensor
    clean_tokens: list[str]
    corrupted_tokens: list[str]
    from_states: dict[tuple[int, str], torch.Tensor]
    from_h_ins: dict[int, torch.Tensor]
    from_cz: dict[int, torch.Tensor]
    from_ffn_acts: dict[int, torch.Tensor]
    base_states: dict[tuple[int, str], torch.Tensor]
    base_h_ins: dict[int, torch.Tensor]
    base_cz: dict[int, torch.Tensor]
    base_ffn_acts: dict[int, torch.Tensor]
    base_input_ids: torch.Tensor
    from_input_ids: torch.Tensor
    grads: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    reader_grads: dict[tuple, torch.Tensor] = field(default_factory=dict)

    def effect(self, ap_raw: float) -> tuple[float, float]:
        """Return ``(ap_effect, ap_recovery)`` for a raw first-order score.

        ``ap_effect = ap_raw / D`` is the signed effect size used for
        ranking and thresholding. ``ap_recovery`` is the display value:
        equal to the effect for denoise, ``1 + effect`` for noise.
        """
        eff = ap_raw / self.denominator
        return eff, (eff if self.direction == "denoise" else 1.0 + eff)


def _check_plain_weights(model, fn_name: str) -> None:
    """Reject models whose o_proj/down_proj weights cannot be multiplied directly.

    The per-head, per-neuron and edge variants multiply gradients by the
    raw ``.weight`` tensors. Packed 4-bit, int8, meta-device (offloaded)
    and multi-device weights make that matmul crash or mix devices.
    """
    if getattr(model, "hf_quantizer", None) is not None:
        raise ValueError(
            f"{fn_name} does not support quantized models: it multiplies by "
            "o_proj/down_proj .weight directly; load the model unquantized"
        )
    devices = {p.device for p in model.parameters()}
    if any(d.type == "meta" for d in devices):
        raise ValueError(
            f"{fn_name} does not support offloaded models (weights on the meta device)"
        )
    if len(devices) > 1:
        raise ValueError(
            f"{fn_name} requires all weights on one device, got {sorted(map(str, devices))}"
        )


def _normalize_layers(model, layers: list[int] | None) -> list[int]:
    num_layers = len(model.model.layers)
    if layers is None:
        return list(range(num_layers))
    out: set[int] = set()
    for L in layers:
        if L < -num_layers or L >= num_layers:
            raise IndexError(f"layer {L} out of range for {num_layers} layers")
        out.add(L % num_layers)
    return sorted(out)


def _resolve_head_dim(model) -> tuple[int, int]:
    """Return ``(n_heads, head_dim)``; ``config.head_dim`` wins when set."""
    cfg = model.config
    n_heads: int = cfg.num_attention_heads
    head_dim: int = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return n_heads, head_dim


def _o_proj_weight(model, L: int, n_heads: int, head_dim: int) -> torch.Tensor:
    W_O: torch.Tensor = model.model.layers[
        L
    ].self_attn.o_proj.weight  # [hidden, n_heads*head_dim]
    if W_O.shape[1] != n_heads * head_dim:
        raise ValueError(
            f"layer {L} o_proj input width {W_O.shape[1]} != "
            f"num_attention_heads*head_dim = {n_heads}*{head_dim}"
        )
    return W_O


def _ap_setup(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    fn_name: str,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str,
    measurement_position: int,
    positions: list[int] | None,
    layers: list[int] | None,
    n_steps: int,
    sublayers: tuple[str, ...] = ("attn", "ffn"),
    capture_concat_z: bool = False,
    capture_ffn_act: bool = False,
    capture_reader_grads: bool = False,
    capture_ffn_out: bool = False,
    plain_weights_required: bool = True,
) -> _APSetup:
    """Validate, run the from/base forwards and compute the metric gradients.

    denoise: from=clean, base=corrupted. noise: from=corrupted, base=clean.
    Two forwards (from, base) plus one backward when n_steps == 1; with
    n_steps > 1 the base forward runs without a graph and the IG loop adds
    n_steps forward+backward passes.
    """
    # --- Validation (raise before any forward pass) ---
    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(f"{fn_name} requires correct_token_id and incorrect_token_id")
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not sublayers or not set(sublayers).issubset({"attn", "ffn"}):
        raise ValueError("sublayers must be a non-empty subset of {'attn', 'ffn'}")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompts cannot be empty")
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")
    if plain_weights_required:
        _check_plain_weights(model, fn_name)
    elif getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            f"{fn_name} on a quantized model: gradient flow works but "
            "precision is reduced (fp16/int8 through bitsandbytes).",
            stacklevel=3,
        )
    target_layers = _normalize_layers(model, layers)

    # --- Tokenize + range checks ---
    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    raw_positions = list(range(seq_len)) if positions is None else positions
    for pos in raw_positions:
        if pos < -seq_len or pos >= seq_len:
            raise IndexError(f"position {pos} out of range for seq_len={seq_len}")
    normalized_positions = [p % seq_len for p in raw_positions]
    if measurement_position < -seq_len or measurement_position >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )
    meas_pos = measurement_position % seq_len

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt
    device = _get_input_device(model)
    from_input_ids = (clean_ids if direction == "denoise" else corr_ids).to(device)
    base_input_ids = (corr_ids if direction == "denoise" else clean_ids).to(device)

    capture_kwargs = dict(
        sublayers=sublayers,
        layers=target_layers,
        capture_concat_z=capture_concat_z and "attn" in sublayers,
        # The IG path forces mlp outputs, so it needs them captured.
        capture_ffn_out=capture_ffn_out or (n_steps > 1 and "ffn" in sublayers),
        capture_ffn_act=capture_ffn_act,
    )

    def _detached(d: dict) -> dict:
        return {k: v.detach() for k, v in d.items()}

    with torch.no_grad():
        from_captured, from_h_ins, from_logits, from_tokens, from_cz, _, from_acts = (
            _capture_residual_stream_with_grad(
                model,
                tokenizer,
                from_prompt,
                **capture_kwargs,  # pyright: ignore[reportArgumentType]
            )
        )

    # The base forward keeps its graph only when the plain gradient is read
    # off it; the IG path needs base values alone.
    with torch.set_grad_enabled(n_steps == 1):
        (
            base_captured,
            base_h_ins,
            base_logits,
            base_tokens,
            base_cz,
            reader_inputs,
            base_acts,
        ) = _capture_residual_stream_with_grad(
            model,
            tokenizer,
            base_prompt,
            capture_reader_grads=capture_reader_grads and n_steps == 1,
            **capture_kwargs,  # pyright: ignore[reportArgumentType]
        )

        clean_baseline = (
            from_logits if direction == "denoise" else base_logits
        ).detach()
        corrupted_baseline = (
            base_logits if direction == "denoise" else from_logits
        ).detach()
        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        )
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        )
        denominator = (d_clean - d_corrupted).item()
        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        grads: dict[tuple[int, str], torch.Tensor] = {}
        reader_grads: dict[tuple, torch.Tensor] = {}
        if n_steps == 1:
            metric = (
                base_logits[meas_pos, correct_token_id]
                - base_logits[meas_pos, incorrect_token_id]
            )
            targets: dict[Hashable, torch.Tensor] = {
                ("site", key): base_captured[key]
                for key in ((L, sub) for L in target_layers for sub in sublayers)
                if key in base_captured
            }
            targets.update({("reader", key): t for key, t in reader_inputs.items()})
            for (kind, key), g in _grads_of(metric, targets).items():  # pyright: ignore[reportGeneralTypeIssues]
                (grads if kind == "site" else reader_grads)[key] = g
            del metric, targets

    if n_steps > 1:
        base_components: dict[tuple[int, str], torch.Tensor] = {}
        from_components: dict[tuple[int, str], torch.Tensor] = {}
        for L in target_layers:
            for sub, cap_key in (("attn", (L, "attn")), ("ffn", (L, "ffn_out"))):
                if (
                    sub in sublayers
                    and cap_key in base_captured
                    and cap_key in from_captured
                ):
                    base_components[(L, sub)] = base_captured[cap_key].detach()
                    from_components[(L, sub)] = from_captured[cap_key].detach()
        grads, reader_grads_ig = _integrated_gradients_loop(
            model=model,
            input_ids=base_input_ids,
            base_components=base_components,
            from_components=from_components,
            measurement_position=meas_pos,
            correct_token_id=correct_token_id,
            incorrect_token_id=incorrect_token_id,
            n_steps=n_steps,
            reader_layers=target_layers,
            capture_reader_grads=capture_reader_grads,
        )
        reader_grads = dict(reader_grads_ig)

    return _APSetup(
        direction=direction,
        denominator=denominator,
        meas_pos=meas_pos,
        positions=normalized_positions,
        layers=target_layers,
        clean_baseline=clean_baseline,
        corrupted_baseline=corrupted_baseline,
        clean_tokens=from_tokens if direction == "denoise" else base_tokens,
        corrupted_tokens=base_tokens if direction == "denoise" else from_tokens,
        from_states=_detached(from_captured),
        from_h_ins=_detached(from_h_ins),
        from_cz=_detached(from_cz),
        from_ffn_acts=_detached(from_acts),
        base_states=_detached(base_captured),
        base_h_ins=_detached(base_h_ins),
        base_cz=_detached(base_cz),
        base_ffn_acts=_detached(base_acts),
        base_input_ids=base_input_ids,
        from_input_ids=from_input_ids,
        grads=grads,
        reader_grads=reader_grads,
    )


def _result(
    setup: _APSetup, cells: list[dict], mode: str, n_steps: int, **extra
) -> PatchingResult:
    return PatchingResult(
        cells=cells,
        clean_baseline_logits=setup.clean_baseline,
        corrupted_baseline_logits=setup.corrupted_baseline,
        prompt_tokens_clean=setup.clean_tokens,
        prompt_tokens_corrupted=setup.corrupted_tokens,
        direction=setup.direction,
        measurement_position=setup.meas_pos,
        mode=mode,
        n_steps=(n_steps if n_steps > 1 else None),
        **extra,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


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

    Two forwards (clean and corrupted) plus one backward produce a per-cell
    AP score that approximates exact activation_patch's
    logit_diff_recovery; with n_steps > 1, Integrated Gradients adds
    n_steps forward+backward passes. Much cheaper than the O(L·S·P) exact
    loop.

    Cells carry ``ap_recovery`` (display value, ``1 + effect`` for noise)
    and ``ap_effect`` (the signed effect ``ap_raw / D``).

    See: Nanda 2023 (attribution patching primer) and Kramár et al. 2024
    (Attribution Patching Outperforms Automated Circuit Discovery).
    """
    setup = _ap_setup(
        model,
        tokenizer,
        clean_prompt,
        corrupted_prompt,
        fn_name="attribution_patch",
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
        sublayers=sublayers,
        plain_weights_required=False,
    )

    # Attn rows use the residual-stream value h_post_attn = h_in + attn_out
    # to match exact AP's patched quantity; its gradient equals attn_out's
    # (chain rule through the `+`). Ffn rows use the layer output.
    cells: list[dict] = []
    for L in setup.layers:
        for sub in sorted(sublayers):
            grad = setup.grads.get((L, sub))
            if grad is None:
                continue
            if sub == "attn":
                from_val = setup.from_h_ins[L] + setup.from_states[(L, "attn")]
                base_val = setup.base_h_ins[L] + setup.base_states[(L, "attn")]
            else:
                from_val = setup.from_states[(L, "ffn")]
                base_val = setup.base_states[(L, "ffn")]
            for pos in setup.positions:
                ap_raw = (
                    ((from_val[0, pos] - base_val[0, pos]) * grad[0, pos]).sum().item()
                )
                ap_effect, ap_recovery = setup.effect(ap_raw)
                cell: dict = {
                    "layer": L,
                    "sublayer": sub,
                    "position": pos,
                    "ap_recovery": float(ap_recovery),
                    "ap_effect": float(ap_effect),
                }
                cells.append(cell)
                if on_cell is not None:
                    on_cell(L, sub, pos, cell)

    return _result(setup, cells, "approx", n_steps)


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
    values plus FFN anchor rows. Same cost as the per-cell variant: two
    forwards plus one backward (n_steps more of each with IG).

    Unit strings in on_cell / cells: "attn.h{N}" (0-indexed) for head N,
    "ffn" for FFN anchor. The head width is ``config.head_dim`` when the
    config sets it, else ``hidden_size // num_attention_heads``.

    Note: sum_h AP_head(L,h,pos) == (delta_attn_out · attn_out.grad) / D, which
    equals the per-cell AP_attn(L,pos) ONLY when h_in is identical between the
    clean and corrupted prompts (trivially at L=0 for same-length tokenizations
    but not at deeper layers). Per-cell AP_attn linearizes at the full residual
    stream h_post_attn = h_in + attn_out to match exact AP's patched quantity;
    per-head AP decomposes attn_out alone, which is the right unit for
    mechanistic interpretability of individual heads.
    """
    setup = _ap_setup(
        model,
        tokenizer,
        clean_prompt,
        corrupted_prompt,
        fn_name="attribution_patch_per_head",
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
        capture_concat_z=True,
    )
    n_heads, head_dim = _resolve_head_dim(model)

    cells: list[dict] = []
    for L in setup.layers:
        # --- FFN anchor (identical math to per-cell AP) ---
        ffn_grad = setup.grads.get((L, "ffn"))
        if ffn_grad is not None:
            from_ffn = setup.from_states[(L, "ffn")]
            base_ffn = setup.base_states[(L, "ffn")]
            for pos in setup.positions:
                ap_raw = (
                    ((from_ffn[0, pos] - base_ffn[0, pos]) * ffn_grad[0, pos])
                    .sum()
                    .item()
                )
                ap_effect, ap_recovery = setup.effect(ap_raw)
                cell: dict = {
                    "layer": L,
                    "unit": "ffn",
                    "position": pos,
                    "ap_recovery": float(ap_recovery),
                    "ap_effect": float(ap_effect),
                }
                cells.append(cell)
                if on_cell is not None:
                    on_cell(L, "ffn", pos, cell)

        # --- Per-head AP via chain rule through W_O ---
        attn_grad = setup.grads.get((L, "attn"))
        if attn_grad is None or L not in setup.base_cz or L not in setup.from_cz:
            continue
        W_O = _o_proj_weight(model, L, n_heads, head_dim)
        # ∂metric/∂concat_z = attn_out_grad @ W_O: [seq, n_heads*head_dim]
        concat_z_grad = attn_grad[0] @ W_O
        delta_z = setup.from_cz[L][0] - setup.base_cz[L][0]
        for pos in setup.positions:
            dz_heads = delta_z[pos].view(n_heads, head_dim)
            cz_grad_heads = concat_z_grad[pos].view(n_heads, head_dim)
            ap_heads_raw = (dz_heads * cz_grad_heads).sum(dim=-1).tolist()
            for h, ap_raw_h in enumerate(ap_heads_raw):
                ap_effect, ap_recovery = setup.effect(ap_raw_h)
                unit = f"attn.h{h}"
                hcell: dict = {
                    "layer": L,
                    "unit": unit,
                    "position": pos,
                    "ap_recovery": float(ap_recovery),
                    "ap_effect": float(ap_effect),
                }
                cells.append(hcell)
                if on_cell is not None:
                    on_cell(L, unit, pos, hcell)

    return _result(setup, cells, "approx_head", n_steps, n_heads=n_heads)


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
    Two forwards plus one backward (n_steps more of each with IG). For
    each layer, each FFN output position, and each neuron index i in
    [0, intermediate_size):

        grad_act = grad_ffn_out @ W_down             # [intermediate]
        delta_act = from_act[pos] - base_act[pos]    # [intermediate]
        ap_effect[i] = delta_act[i] * grad_act[i] / D
        ap_recovery[i] = ap_effect[i] (denoise) or 1 + ap_effect[i] (noise)

    Returns PatchingResult with mode='approx_neuron',
    n_neurons=intermediate_size, and `cells` containing only the top-k
    tuples by |ap_effect|. If top_k_neurons exceeds the total
    neuron-cell count, silently caps.
    """
    if top_k_neurons < 1:
        raise ValueError("top_k_neurons must be >= 1")
    setup = _ap_setup(
        model,
        tokenizer,
        clean_prompt,
        corrupted_prompt,
        fn_name="attribution_patch_per_neuron",
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
        capture_ffn_act=True,
    )

    intermediate_size: int = model.config.intermediate_size
    effect_rows: list[torch.Tensor] = []  # each [n_pos, intermediate]
    row_layers: list[int] = []
    for L in setup.layers:
        grad = setup.grads.get((L, "ffn"))
        if grad is None or L not in setup.base_ffn_acts or L not in setup.from_ffn_acts:
            continue
        W_down: torch.Tensor = model.model.layers[
            L
        ].mlp.down_proj.weight  # [hidden, intermediate]
        grad_act = grad[0, setup.positions] @ W_down  # [n_pos, intermediate]
        delta_act = (
            setup.from_ffn_acts[L][0, setup.positions]
            - setup.base_ffn_acts[L][0, setup.positions]
        )
        effect_rows.append((delta_act * grad_act).float() / setup.denominator)
        row_layers.append(L)

    top_cells: list[dict] = []
    if effect_rows:
        effects = torch.stack(effect_rows)  # [n_layers, n_pos, intermediate]
        flat = effects.flatten()
        k = min(top_k_neurons, flat.numel())
        top = torch.topk(flat.abs(), k)
        n_pos, n_inter = effects.shape[1], effects.shape[2]
        for flat_idx, eff in zip(top.indices.tolist(), flat[top.indices].tolist()):
            li, rem = divmod(flat_idx, n_pos * n_inter)
            pi, i = divmod(rem, n_inter)
            top_cells.append(
                {
                    "layer": row_layers[li],
                    "unit": f"neuron.n{i}",
                    "neuron": i,
                    "position": setup.positions[pi],
                    "ap_recovery": float(eff if direction == "denoise" else 1.0 + eff),
                    "ap_effect": float(eff),
                }
            )

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    return _result(
        setup, top_cells, "approx_neuron", n_steps, n_neurons=intermediate_size
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
    list[dict],  # all_edge_scores (unsorted)
    torch.Tensor,  # clean_baseline_logits (detached)
    torch.Tensor,  # corrupted_baseline_logits (detached)
    list[str],  # clean_tokens (ordered by direction)
    list[str],  # corrupted_tokens (ordered by direction)
    int,  # meas_pos (normalized to [0, seq_len))
    int,  # n_heads
]:
    """Core forward+backward+edge-enumeration shared by edge_attribution_patch
    and extract_circuit.

    Each caller validates its own top-k / tau parameters; everything else
    is validated in the shared setup.
    """
    setup = _ap_setup(
        model,
        tokenizer,
        clean_prompt,
        corrupted_prompt,
        fn_name="edge_attribution_patch",
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
        capture_concat_z=True,
        capture_reader_grads=True,
        capture_ffn_out=True,
    )
    n_heads, head_dim = _resolve_head_dim(model)

    with torch.no_grad():
        delta_embed = model.model.embed_tokens(
            setup.from_input_ids
        ) - model.model.embed_tokens(setup.base_input_ids)

    delta_ffn: dict[int, torch.Tensor] = {}
    delta_z: dict[int, torch.Tensor] = {}
    W_O: dict[int, torch.Tensor] = {}
    for L in setup.layers:
        if (L, "ffn_out") in setup.from_states and (L, "ffn_out") in setup.base_states:
            delta_ffn[L] = (
                setup.from_states[(L, "ffn_out")] - setup.base_states[(L, "ffn_out")]
            )
        if L in setup.from_cz and L in setup.base_cz:
            delta_z[L] = setup.from_cz[L] - setup.base_cz[L]
            W_O[L] = _o_proj_weight(model, L, n_heads, head_dim)

    all_edge_scores: list[dict] = []

    def _add(
        writer_layer: int,
        writer_unit: str,
        reader_L: int,
        reader_type: str,
        pos: int,
        ap_raw: float,
    ) -> None:
        ap_effect, ap_recovery = setup.effect(ap_raw)
        all_edge_scores.append(
            {
                "writer_layer": writer_layer,
                "writer_unit": writer_unit,
                "reader_layer": reader_L,
                "reader_unit": reader_type,
                "position": pos,
                "ap_recovery": float(ap_recovery),
                "ap_effect": float(ap_effect),
            }
        )

    for reader_key, grad_r_full in setup.reader_grads.items():
        reader_type, reader_L = reader_key[0], reader_key[1]
        for pos in setup.positions:
            grad_r = grad_r_full[0, pos]
            _add(
                0,
                "embed",
                reader_L,
                reader_type,
                pos,
                (delta_embed[0, pos] * grad_r).sum().item(),
            )

            for L_w in setup.layers:
                if (
                    _is_valid_ffn_writer(L_w, reader_type, reader_L)
                    and L_w in delta_ffn
                ):
                    _add(
                        L_w,
                        "ffn",
                        reader_L,
                        reader_type,
                        pos,
                        (delta_ffn[L_w][0, pos] * grad_r).sum().item(),
                    )

                if _is_valid_attn_writer(L_w, reader_type, reader_L) and L_w in delta_z:
                    grad_z_heads = (grad_r @ W_O[L_w]).view(n_heads, head_dim)
                    dz_heads = delta_z[L_w][0, pos].view(n_heads, head_dim)
                    for h, ap_h_raw in enumerate(
                        (dz_heads * grad_z_heads).sum(dim=-1).tolist()
                    ):
                        _add(L_w, f"attn.h{h}", reader_L, reader_type, pos, ap_h_raw)

    return (
        all_edge_scores,
        setup.clean_baseline,
        setup.corrupted_baseline,
        setup.clean_tokens,
        setup.corrupted_tokens,
        setup.meas_pos,
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
    position) AP scores. Two forwards plus one backward (n_steps more of each
    with IG). Each edge score measures how much writer w's output delta
    (clean - corrupted) aligned with the gradient at reader r's input.

    Valid edges respect the residual stream's topological order:
    - embed → any reader
    - attn(L_w) → attn_in(L_r) iff L_w < L_r
    - attn(L_w) → ffn_in(L_r) iff L_w <= L_r
    - ffn(L_w) → attn_in(L_r) or ffn_in(L_r) iff L_w < L_r
    - any writer → logits reader

    Returns PatchingResult with mode="edge", n_edges=total_pre_filter_count.
    cells contains only the top-k edges by |ap_effect|.
    """
    if top_k_edges < 1:
        raise ValueError("top_k_edges must be >= 1")

    (
        all_edge_scores,
        clean_baseline_logits,
        corrupted_baseline_logits,
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    ) = _compute_all_edges(
        model,
        tokenizer,
        clean_prompt,
        corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
    )

    n_edges_total = len(all_edge_scores)
    all_edge_scores.sort(key=lambda c: abs(c["ap_effect"]), reverse=True)
    top_cells = all_edge_scores[:top_k_edges]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    return PatchingResult(
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
        n_steps=(n_steps if n_steps > 1 else None),
    )


_Node = tuple[int, str, int]


def _readers_feeding(
    writer: tuple[int, str, int],
    positions: set[int],
) -> list[tuple[int, str, int]]:
    """Reader nodes whose value the writer node's output is computed from.

    An FFN at (L, pos) reads ffn_in(L) at pos. An attention head at
    (L, pos) reads attn_in(L) at every position q <= pos (causal
    attention). The embedding reads nothing.
    """
    L, unit, pos = writer
    if unit == "ffn":
        return [(L, "ffn_in", pos)]
    if unit.startswith("attn."):
        return [(L, "attn_in", q) for q in sorted(positions) if q <= pos]
    return []


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
    `top_k_candidates` edges (by |ap_effect|) with `in_circuit: bool`:
      1. |ap_effect| >= tau (filter)
      2. the edge's reader is reverse-reachable from 'logits' through
         surviving edges. The walk goes reader -> writer along a surviving
         edge, then writer -> the reader nodes that writer computes from
         (ffn(L) reads ffn_in(L); an attention head at L reads attn_in(L)
         at the same and earlier positions), so it follows multi-hop
         paths such as embed -> attn_in(1) -> attn.hN(1) -> logits.

    Returns PatchingResult with mode='circuit'. Cells include all top-k
    candidates (in-circuit and out). Summary fields:
      n_edges              - total pre-filter edge count
      n_edges_in_circuit   - count of cells with in_circuit=True
      n_nodes_in_circuit   - visited writer and reader nodes, inclusive of
                              the logits sink (a graph with only
                              embed->logits yields n_nodes_in_circuit == 2).
      tau                  - applied threshold

    If `top_k_candidates > total valid edges`, silently caps at the actual
    edge count (matches edge_attribution_patch's top_k_edges behavior).
    """
    if tau < 0.0:
        raise ValueError("tau must be >= 0.0")
    if top_k_candidates < 1:
        raise ValueError("top_k_candidates must be >= 1")

    (
        all_edge_scores,
        clean_baseline_logits,
        corrupted_baseline_logits,
        clean_tokens,
        corrupted_tokens,
        meas_pos,
        n_heads,
    ) = _compute_all_edges(
        model,
        tokenizer,
        clean_prompt,
        corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        n_steps=n_steps,
    )

    n_edges_total = len(all_edge_scores)
    all_edge_scores.sort(key=lambda c: abs(c["ap_effect"]), reverse=True)
    top_cells = all_edge_scores[:top_k_candidates]

    def node_of_writer(cell: dict) -> _Node:
        return (cell["writer_layer"], cell["writer_unit"], cell["position"])

    def node_of_reader(cell: dict) -> _Node:
        return (cell["reader_layer"], cell["reader_unit"], cell["position"])

    def survives(cell: dict) -> bool:
        return abs(cell["ap_effect"]) >= tau

    reverse_adj: dict[_Node, list[_Node]] = {}
    for cell in top_cells:
        if survives(cell):
            reverse_adj.setdefault(node_of_reader(cell), []).append(
                node_of_writer(cell)
            )
    edge_positions = {cell["position"] for cell in top_cells}

    visited: set[_Node] = set()
    queue: list[_Node] = []
    for node in reverse_adj:
        if node[1] == "logits":
            visited.add(node)
            queue.append(node)

    while queue:
        r = queue.pop()
        for w in reverse_adj.get(r, []):
            if w in visited:
                continue
            visited.add(w)
            for r_up in _readers_feeding(w, edge_positions):
                if r_up in reverse_adj and r_up not in visited:
                    visited.add(r_up)
                    queue.append(r_up)

    n_edges_in_circuit = 0
    for cell in top_cells:
        cell["in_circuit"] = survives(cell) and node_of_reader(cell) in visited
        n_edges_in_circuit += cell["in_circuit"]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    return PatchingResult(
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
        n_steps=(n_steps if n_steps > 1 else None),
    )
