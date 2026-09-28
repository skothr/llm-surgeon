"""Model loading and layer surgery operations."""

import copy
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_surgeon._paths import model_cache_dir

logger = logging.getLogger("llm_surgeon.surgery")

# Dedicated cache for clean HF model downloads: $LLM_SURGEON_CACHE_DIR, else
# <llm-surgeon home>/models (see llm_surgeon._paths). Resolved at import time.
MODEL_CACHE_DIR = model_cache_dir()

# Threshold above which duplicate_layer issues a memory warning. Sized for a
# 32 GB host with headroom for activations + Python overhead.
_FP16_GB_WARN_THRESHOLD = 28.0


# Any one of these, next to config.json, means the weights were fetched.
_WEIGHT_FILES = (
    "model.safetensors", "model.safetensors.index.json",
    "pytorch_model.bin", "pytorch_model.bin.index.json",
)


def _is_cached(model_id: str, cache_dir: str | None = None, revision: str | None = None) -> bool:
    """True if the local HF cache holds config.json and a weights file for
    ``model_id`` at ``revision`` (default: ``main``).

    load_model passes ``local_files_only`` from this probe, so a config-only
    cache (e.g. left by an ``AutoConfig`` call or an interrupted download)
    or a different revision must read as not cached, or the download would
    never be attempted.
    """
    from huggingface_hub import try_to_load_from_cache

    def _hit(filename: str) -> bool:
        path = try_to_load_from_cache(
            model_id, filename=filename,
            cache_dir=cache_dir or MODEL_CACHE_DIR, revision=revision,
        )
        # None = not cached; a non-str sentinel = cached as known-missing.
        return isinstance(path, str)

    return _hit("config.json") and any(_hit(f) for f in _WEIGHT_FILES)


@dataclass
class SurgeryOp:
    """A single surgery operation record."""
    operation: str
    description: str
    layer_count_before: int
    layer_count_after: int

    def __str__(self) -> str:
        return (
            f"{self.operation}: {self.description} "
            f"({self.layer_count_before} -> {self.layer_count_after} layers)"
        )


@dataclass
class SurgeryLog:
    """Log of surgery operations performed on a model."""
    ops: list[SurgeryOp] = field(default_factory=list)

    def add(self, operation: str, description: str, before: int, after: int) -> None:
        self.ops.append(SurgeryOp(operation, description, before, after))

    @classmethod
    def of(cls, operation: str, description: str, before: int, after: int) -> "SurgeryLog":
        """Build a single-op log. Use for structural ops where before != after."""
        log = cls()
        log.add(operation, description, before, after)
        return log

    @classmethod
    def inplace(cls, model, operation: str, description: str) -> "SurgeryLog":
        """Build a single-op log for in-place ops that don't change layer count."""
        n = len(model.model.layers)
        return cls.of(operation, description, n, n)

    def __str__(self) -> str:
        if not self.ops:
            return "SurgeryLog: (empty)"
        lines = ["SurgeryLog:"]
        for op in self.ops:
            lines.append(f"  {op}")
        return "\n".join(lines)


def _renumber_layers(model) -> None:
    """Renumber self_attn.layer_idx on every layer to match its current position.

    The KV-cache is indexed by layer_idx; after any structural surgery the
    surviving layers must use contiguous indices or the cache will go out of
    range on the next forward pass.
    """
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = i


# Attribute set on each decoder layer module recording its index before any
# structural surgery. Module identity survives remove/keep/reorder/swap, and
# duplicate_layer's deepcopy carries it to the copy, so calibrate() can match
# a surviving layer to the baseline stats of the layer it came from.
_ORIGIN_ATTR = "_llm_surgeon_origin"


def _layer_origins(model) -> list[int]:
    """Return each current layer's original index, tagging untagged layers.

    A layer without a tag is tagged with its current position, so the first
    call on a freshly loaded model yields ``[0, 1, ..., n-1]``. Tags live on
    the module objects and are not saved by ``save_pretrained``; a reloaded
    model starts numbering afresh.
    """
    origins = []
    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer, _ORIGIN_ATTR):
            setattr(layer, _ORIGIN_ATTR, i)
        origins.append(getattr(layer, _ORIGIN_ATTR))
    return origins


def get_layer_info(model) -> dict[str, Any]:
    """Print and return summary of model layer structure."""
    layers = model.model.layers
    total_params = sum(p.numel() for p in model.parameters())
    est_memory_gb = total_params * 2 / 1e9

    layer_params = []
    for i, layer in enumerate(layers):
        lp = sum(p.numel() for p in layer.parameters())
        layer_params.append(lp)
        print(f"  Layer {i:2d}: {lp:,} params")

    print(f"\nModel: {model.config.model_type}")
    print(f"Layers: {len(layers)}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory (fp16): {est_memory_gb:.2f} GB")

    return {
        "num_layers": len(layers),
        "hidden_size": model.config.hidden_size,
        "total_params": total_params,
        "estimated_memory_gb": est_memory_gb,
        "layer_params": layer_params,
    }


def remove_layers(model, layer_indices: list[int]) -> SurgeryLog:
    """Remove layers at the specified indices. Indices are current positions."""
    layers = model.model.layers
    num_before = len(layers)
    _layer_origins(model)  # tag layers before their positions change

    # Reject duplicates explicitly — without this, sorted(reverse=True) would
    # pop the same index twice and silently remove a neighbouring layer.
    if len(set(layer_indices)) != len(layer_indices):
        dupes = sorted({i for i in layer_indices if layer_indices.count(i) > 1})
        raise ValueError(f"Duplicate layer indices in remove_layers: {dupes}")

    for idx in layer_indices:
        if idx < 0 or idx >= num_before:
            raise IndexError(f"Layer index {idx} out of range [0, {num_before - 1}]")

    for idx in sorted(layer_indices, reverse=True):
        del layers[idx]

    model.config.num_hidden_layers = len(layers)
    _renumber_layers(model)

    return SurgeryLog.of(
        "remove_layers", f"Removed layers {layer_indices}", num_before, len(layers)
    )


def keep_layers(model, layer_indices: list[int]) -> SurgeryLog:
    """Keep only the layers at the specified indices, remove all others.

    Indices may be reordered but not repeated: a repeated index would put one
    module object at two positions sharing one KV-cache slot and one set of
    weights. Use :func:`duplicate_layer` to repeat a layer.
    """
    layers = model.model.layers
    num_before = len(layers)
    _layer_origins(model)  # tag layers before their positions change

    if len(set(layer_indices)) != len(layer_indices):
        dupes = sorted({i for i in layer_indices if layer_indices.count(i) > 1})
        raise ValueError(
            f"Duplicate layer indices in keep_layers: {dupes} (use duplicate_layer to repeat a layer)"
        )

    for idx in layer_indices:
        if idx < 0 or idx >= num_before:
            raise IndexError(f"Layer index {idx} out of range [0, {num_before - 1}]")

    new_layers = nn.ModuleList([layers[i] for i in layer_indices])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    _renumber_layers(model)

    return SurgeryLog.of(
        "keep_layers", f"Kept layers {layer_indices}", num_before, len(new_layers)
    )


def reorder_layers(model, new_order: list[int]) -> SurgeryLog:
    """Rearrange layers to the specified order. new_order must be a permutation."""
    layers = model.model.layers
    num_before = len(layers)
    _layer_origins(model)  # tag layers before their positions change

    if len(new_order) != num_before:
        raise ValueError(f"new_order length ({len(new_order)}) must match layer count ({num_before})")
    if set(new_order) != set(range(num_before)):
        raise ValueError(f"new_order must be a permutation of [0, {num_before - 1}]")

    new_layers = nn.ModuleList([layers[i] for i in new_order])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    _renumber_layers(model)

    return SurgeryLog.of(
        "reorder_layers", f"Reordered to {new_order}", num_before, len(new_layers)
    )


def swap_layers(model, i: int, j: int) -> SurgeryLog:
    """Swap two layers' positions."""
    layers = model.model.layers
    num_before = len(layers)
    _layer_origins(model)  # tag layers before their positions change

    for idx in (i, j):
        if idx < 0 or idx >= num_before:
            raise IndexError(f"Layer index {idx} out of range [0, {num_before - 1}]")

    layers[i], layers[j] = layers[j], layers[i]
    _renumber_layers(model)

    return SurgeryLog.of(
        "swap_layers", f"Swapped layers {i} and {j}", num_before, len(layers)
    )


def duplicate_layer(model, src: int, dst: int) -> SurgeryLog:
    """Deep-copy a layer and insert it at the destination position."""
    layers = model.model.layers
    num_before = len(layers)
    _layer_origins(model)  # tag layers before their positions change

    if src < 0 or src >= num_before:
        raise IndexError(f"Source index {src} out of range [0, {num_before - 1}]")
    if dst < 0 or dst > num_before:
        raise IndexError(f"Destination index {dst} out of range [0, {num_before}]")

    total_params = sum(p.numel() for p in model.parameters())
    est_gb = total_params * 2 / 1e9
    if est_gb > _FP16_GB_WARN_THRESHOLD:
        warnings.warn(
            f"Model is ~{est_gb:.1f} GB in fp16, approaching 32 GB RAM limit. "
            f"Duplicating a layer will increase this.",
            ResourceWarning,
        )

    new_layer = copy.deepcopy(layers[src])
    layers.insert(dst, new_layer)
    model.config.num_hidden_layers = len(layers)
    _renumber_layers(model)

    return SurgeryLog.of(
        "duplicate_layer", f"Duplicated layer {src} -> position {dst}", num_before, len(layers)
    )


# Attention head surgery

def _validate_head_args(model, layer: int, heads: list) -> None:
    """Validate layer index and head indices."""
    num_layers = len(model.model.layers)
    if layer < 0 or layer >= num_layers:
        raise IndexError(f"Layer index {layer} out of range [0, {num_layers - 1}]")
    num_heads = model.config.num_attention_heads
    for h in heads:
        if h < 0 or h >= num_heads:
            raise IndexError(f"Head index {h} out of range [0, {num_heads - 1}]")
    if len(set(heads)) != len(heads):
        dupes = sorted({h for h in heads if heads.count(h) > 1})
        raise ValueError(f"Duplicate head indices: {dupes}")


def _head_dim(model) -> int:
    """Per-head dimension, honouring an explicit ``config.head_dim``.

    Matches HF ``LlamaAttention``: some LLaMA-family configs (e.g.
    Mistral-Nemo) set ``head_dim`` to something other than
    ``hidden_size // num_attention_heads``.
    """
    cfg = model.config
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def _require_dense_weight(module: nn.Module, op: str) -> torch.Tensor:
    """Return ``module.weight.data`` if it is a plain floating-point tensor.

    bitsandbytes layers (the ``nf4``/``int8`` load modes) keep packed 4-bit
    codes or int8 codes in ``weight.data``. Zeroing, scaling or slicing head
    columns there corrupts the weights instead of editing them, so raise.
    """
    weight = module.weight
    if hasattr(weight, "quant_state") or hasattr(weight, "SCB") or not weight.is_floating_point():
        raise TypeError(
            f"{op}: {type(module).__name__} holds a quantized weight "
            f"({type(weight).__name__}, dtype={weight.dtype}); weight-level surgery "
            "needs a dense model. Load it with mode 'bf16', 'fp16', 'fp32' or "
            "'fp32-cpu' ('export') instead of 'nf4'/'int8'."
        )
    return weight.data


def zero_heads(model, layer: int, heads: list[int]) -> SurgeryLog:
    """Zero out specific attention heads by zeroing their o_proj columns.

    The head still exists structurally but contributes nothing to the
    residual stream. This is the standard ablation approach.
    """
    _validate_head_args(model, layer, heads)
    hd = _head_dim(model)
    o = _require_dense_weight(model.model.layers[layer].self_attn.o_proj, "zero_heads")
    with torch.no_grad():
        for h in heads:
            o[:, h * hd : (h + 1) * hd] = 0

    return SurgeryLog.inplace(model, "zero_heads", f"Zeroed heads {heads} in layer {layer}")


def scale_heads(model, layer: int, heads: list[int], factor: float) -> SurgeryLog:
    """Scale specific heads' contribution by multiplying their o_proj columns."""
    _validate_head_args(model, layer, heads)
    hd = _head_dim(model)
    o = _require_dense_weight(model.model.layers[layer].self_attn.o_proj, "scale_heads")
    with torch.no_grad():
        for h in heads:
            o[:, h * hd : (h + 1) * hd] *= factor

    return SurgeryLog.inplace(
        model, "scale_heads", f"Scaled heads {heads} in layer {layer} by {factor}"
    )


def _swap_rows(t: torch.Tensor, a: int, b: int, size: int) -> None:
    """Exchange rows ``[a*size, (a+1)*size)`` and ``[b*size, (b+1)*size)`` of ``t``."""
    t[a * size : (a + 1) * size], t[b * size : (b + 1) * size] = (
        t[b * size : (b + 1) * size].clone(),
        t[a * size : (a + 1) * size].clone(),
    )


def swap_heads(model, layer: int, h1: int, h2: int) -> SurgeryLog:
    """Permute two attention heads: exchange their q/k/v rows and o_proj columns.

    This is a relabelling, not an ablation: the model computes the same
    function afterwards (logits are unchanged up to float rounding). It is
    useful for testing position-dependent tooling, not for changing behaviour.

    Under GQA (``num_key_value_heads < num_attention_heads``) query heads in
    one KV group share a K/V head, so only heads in the same group can be
    swapped (their q rows and o columns move; the shared K/V stays). A
    cross-group swap cannot be expressed as a slice exchange — moving a K/V
    head would rewire every other query head in both groups — and raises
    ``ValueError``.
    """
    _validate_head_args(model, layer, sorted({h1, h2}))
    hd = _head_dim(model)
    attn = model.model.layers[layer].self_attn

    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", None) or num_q_heads
    kv_group_size = num_q_heads // num_kv_heads
    kv1 = h1 // kv_group_size
    kv2 = h2 // kv_group_size
    if kv1 != kv2 and kv_group_size > 1:
        raise ValueError(
            f"swap_heads: heads {h1} and {h2} use different KV heads ({kv1} and {kv2}; "
            f"{kv_group_size} query heads share each KV head). Only heads in the same "
            "KV group can be swapped under GQA."
        )

    q = _require_dense_weight(attn.q_proj, "swap_heads")
    o = _require_dense_weight(attn.o_proj, "swap_heads")
    kv = [_require_dense_weight(p, "swap_heads") for p in (attn.k_proj, attn.v_proj)]
    with torch.no_grad():
        # q_proj rows (each head's query projection), plus bias if present.
        _swap_rows(q, h1, h2, hd)
        if attn.q_proj.bias is not None:
            _swap_rows(attn.q_proj.bias.data, h1, h2, hd)

        # MHA: each query head owns its K/V head, so those rows move too.
        if kv1 != kv2:
            for proj, w in zip((attn.k_proj, attn.v_proj), kv):
                _swap_rows(w, kv1, kv2, hd)
                if proj.bias is not None:
                    _swap_rows(proj.bias.data, kv1, kv2, hd)

        # o_proj columns (each head's output contribution).
        _swap_rows(o.T, h1, h2, hd)

    return SurgeryLog.inplace(model, "swap_heads", f"Swapped heads {h1} and {h2} in layer {layer}")


def zero_mlp(model, layer: int) -> SurgeryLog:
    """Zero out a layer's MLP by zeroing down_proj's weight and bias (if any).

    The MLP still exists structurally but contributes nothing to the
    residual stream (the residual connection passes through unchanged).
    """
    num_layers = len(model.model.layers)
    if layer < 0 or layer >= num_layers:
        raise IndexError(f"Layer index {layer} out of range [0, {num_layers - 1}]")
    down_proj = model.model.layers[layer].mlp.down_proj
    down = _require_dense_weight(down_proj, "zero_mlp")
    with torch.no_grad():
        down.zero_()
        if down_proj.bias is not None:
            down_proj.bias.data.zero_()
    return SurgeryLog.inplace(model, "zero_mlp", f"Zeroed MLP in layer {layer}")


def zero_attention(model, layer: int) -> SurgeryLog:
    """Zero out a layer's entire attention by zeroing o_proj's weight and bias (if any).

    The attention module still exists structurally but contributes nothing
    to the residual stream.
    """
    num_layers = len(model.model.layers)
    if layer < 0 or layer >= num_layers:
        raise IndexError(f"Layer index {layer} out of range [0, {num_layers - 1}]")
    o_proj = model.model.layers[layer].self_attn.o_proj
    o = _require_dense_weight(o_proj, "zero_attention")
    with torch.no_grad():
        o.zero_()
        if o_proj.bias is not None:
            o_proj.bias.data.zero_()
    return SurgeryLog.inplace(model, "zero_attention", f"Zeroed attention in layer {layer}")


@dataclass
class CalibrationStats:
    """Per-layer per-channel mean-square of RMSNorm outputs.

    Produced by :func:`capture_calibration_stats` on a reference (pre-surgery)
    model. Each tensor has shape ``(hidden_size,)`` and represents
    ``mean_pos(y_i^2)`` over all token positions in the calibration text,
    where ``y`` is the output of the corresponding RMSNorm layer.

    Attributes:
        input_norm: Mean-square of each layer's ``input_layernorm`` output.
        post_attn_norm: Same for ``post_attention_layernorm``.
        layer_origins: Original (pre-surgery) index of each captured layer,
            used by :func:`calibrate` to pair surviving layers with their
            baseline. ``None`` means stats index ``i`` is original layer ``i``.
    """
    input_norm: list[torch.Tensor]
    post_attn_norm: list[torch.Tensor]
    layer_origins: list[int] | None = None

    @property
    def num_layers(self) -> int:
        return len(self.input_norm)

    def __len__(self) -> int:
        return self.num_layers


@dataclass
class CalibrationReport:
    """Summary of a :func:`calibrate` run.

    Attributes:
        layers_calibrated: Count of layer/norm pairs whose weight was rescaled
            (a norm whose every channel was skipped is not counted).
        channels_clipped: Total channels whose scale hit the clip bounds
            (indicates severe variance mismatch — often a dead channel).
        channels_skipped: Channels skipped due to sub-threshold variance in
            either baseline or current stats.
        per_layer_scale_mean: Mean of the applied scale vector for each
            layer/norm pair visited, in order (``input_layernorm`` then
            ``post_attention_layernorm`` per layer), so its length is up to
            twice the layer count (diagnostic — values far from 1.0 indicate
            large drift).
        layers_fully_skipped: Current-model layer indices where every channel
            of at least one norm was skipped — i.e. the layer is
            mathematically untouched. Usually indicates a hook that never
            fired during stats capture (baseline or current) and is a
            silent correctness hazard.
    """
    layers_calibrated: int = 0
    channels_clipped: int = 0
    channels_skipped: int = 0
    per_layer_scale_mean: list[float] = field(default_factory=list)
    layers_fully_skipped: list[int] = field(default_factory=list)


def capture_calibration_stats(
    model,
    tokenizer,
    text: str | None = None,
    dataset: str | None = None,
    num_samples: int = 128,
) -> CalibrationStats:
    """Capture per-channel post-norm mean-square for each RMSNorm layer.

    Run this BEFORE surgery on the original model. The returned stats are
    passed to :func:`calibrate` after surgery to rescale each layer's
    RMSNorm gain per-channel so the post-norm output distribution
    downstream layers see matches what they were trained on.

    The stats record each layer's original index (see
    :attr:`CalibrationStats.layer_origins`), so capturing on a model that
    has already had structural surgery is also supported.

    Returns:
        :class:`CalibrationStats` with ``input_norm[i]`` and
        ``post_attn_norm[i]`` each a ``(hidden_size,)`` tensor on CPU.
    """
    return _capture_norm_outputs(
        model, tokenizer, text=text, dataset=dataset, num_samples=num_samples
    )


def calibrate(
    model,
    tokenizer,
    baseline_stats: CalibrationStats | None = None,
    *,
    layer_map: list[int] | None = None,
    scale_clip: float = 5.0,
    min_variance: float = 1e-6,
    text: str | None = None,
    dataset: str | None = None,
    num_samples: int = 128,
) -> CalibrationReport:
    """Rescale RMSNorm gains per-channel to match pre-surgery post-norm variance.

    For each surviving layer's ``input_layernorm`` and
    ``post_attention_layernorm``, captures the current per-channel
    mean-square of the norm's output, and multiplies the gain vector
    element-wise by ``sqrt(baseline_mean_sq / current_mean_sq)``.

    This is a first-order correction. The current stats are captured in one
    forward pass before any gain changes, and rescaling a norm changes the
    residual stream every later norm sees. The per-channel scale is exact
    only for the first rescaled norm; for later norms it is computed from
    stale inputs.

    This does NOT correct directional drift in the residual stream — layer
    removal changes the direction of activations, and no amount of gain
    scaling fixes that. The goal is more modest: keep each channel's
    post-norm output in the magnitude regime the downstream layer was
    trained on, so non-linearities and attention softmaxes don't saturate.

    Args:
        model: The surgically-modified model to calibrate.
        tokenizer: Tokenizer matching the model.
        baseline_stats: :class:`CalibrationStats` from the pre-surgery model
            (required; ``None`` raises ``ValueError``).
        layer_map: Maps current-model layer index → baseline layer index.
            If ``None``, each current layer is paired with the baseline entry
            for the layer it originally was: structural ops in this module
            (``remove_layers``, ``keep_layers``, ``reorder_layers``,
            ``swap_layers``, ``duplicate_layer``) record each layer's
            original index. A duplicated layer is paired with its source.
            Layers with no baseline counterpart are left unchanged, with a
            warning.
        scale_clip: Per-channel scale factor is clipped to
            ``[1/scale_clip, scale_clip]`` so dead or near-dead channels
            don't produce astronomical gains.
        min_variance: Channels with mean-square below this in either the
            baseline or the current stats are left untouched (prevents
            division by noise).
        text, dataset, num_samples: Calibration corpus. Same text should be
            used for baseline capture and this call for a fair comparison.

    Returns:
        :class:`CalibrationReport` summarising what was changed.
    """
    if baseline_stats is None:
        raise ValueError(
            "calibrate() needs baseline_stats. Call capture_calibration_stats() "
            "on the original model before surgery and pass the result here."
        )
    report = CalibrationReport()

    current_layers = model.model.layers
    mapping: list[int | None]
    if layer_map is None:
        mapping = _default_layer_map(model, baseline_stats)
        unmatched = [i for i, b in enumerate(mapping) if b is None]
        if unmatched:
            warnings.warn(
                f"calibrate(): current layers {unmatched} have no counterpart in "
                f"baseline_stats and are left uncalibrated.",
                UserWarning,
                stacklevel=2,
            )
    else:
        bad = [b for b in layer_map if not 0 <= b < baseline_stats.num_layers]
        if bad:
            raise ValueError(
                f"layer_map entries {bad} out of range [0, {baseline_stats.num_layers - 1}]"
            )
        mapping = list(layer_map)

    current_stats = _capture_norm_outputs(
        model, tokenizer, text=text, dataset=dataset, num_samples=num_samples
    )

    for cur_idx, base_idx in enumerate(mapping):
        if cur_idx >= len(current_layers):
            break
        if base_idx is None:
            continue

        layer = current_layers[cur_idx]
        layer_has_full_skip = False
        for attr, base_list, cur_list in (
            ("input_layernorm", baseline_stats.input_norm, current_stats.input_norm),
            ("post_attention_layernorm", baseline_stats.post_attn_norm, current_stats.post_attn_norm),
        ):
            if not hasattr(layer, attr):
                continue
            norm = getattr(layer, attr)
            if norm.weight is None:
                continue

            base_ms = base_list[base_idx].to(norm.weight.device, dtype=torch.float32)
            cur_ms = cur_list[cur_idx].to(norm.weight.device, dtype=torch.float32)

            # Per-channel scale: g_new = g * sqrt(baseline / current). Channels
            # below min_variance in either side are skipped (scale = 1).
            valid = (base_ms > min_variance) & (cur_ms > min_variance)
            raw_scale = torch.where(valid, torch.sqrt(base_ms / cur_ms.clamp_min(min_variance)),
                                    torch.ones_like(base_ms))
            lo = 1.0 / scale_clip
            hi = scale_clip
            clipped = raw_scale.clamp(min=lo, max=hi)
            report.channels_clipped += int((raw_scale != clipped).sum().item())
            skipped_here = int((~valid).sum().item())
            report.channels_skipped += skipped_here
            report.per_layer_scale_mean.append(float(clipped.mean().item()))

            # A norm whose every channel was skipped contributes nothing — the
            # applied scale is identity. This usually means stats capture
            # produced an all-zero tensor for this layer/norm (hook never fired).
            if skipped_here == base_ms.numel():
                layer_has_full_skip = True

            with torch.no_grad():
                norm.weight.data.mul_(clipped.to(norm.weight.dtype))
            if skipped_here < base_ms.numel():
                report.layers_calibrated += 1

        if layer_has_full_skip:
            report.layers_fully_skipped.append(cur_idx)

    if report.layers_fully_skipped:
        warnings.warn(
            f"calibrate(): layers {report.layers_fully_skipped} were fully skipped "
            f"(every channel sub-threshold in baseline or current stats). "
            f"These layers are mathematically unchanged — likely a missed hook "
            f"during stats capture. Surgery may not be calibrated correctly.",
            UserWarning,
            stacklevel=2,
        )

    return report


def _default_layer_map(model, baseline_stats: CalibrationStats) -> list[int | None]:
    """Pair each current layer with the baseline index of the layer it came from."""
    if baseline_stats.layer_origins is None:
        # Hand-built stats: entry i is original layer i.
        position = {i: i for i in range(baseline_stats.num_layers)}
    else:
        position = {}
        for i, origin in enumerate(baseline_stats.layer_origins):
            position.setdefault(origin, i)
    return [position.get(origin) for origin in _layer_origins(model)]


# Tokens per calibration sequence (capped by the model's context length).
_CALIB_SEQ_LEN = 512
_CALIB_DATASETS = ("wikitext2",)


def _calibration_sequences(
    model, tokenizer, text: str | None, dataset: str | None, num_samples: int,
) -> list[torch.Tensor]:
    """Tokenize the calibration corpus into at most ``num_samples`` sequences.

    ``text`` wins over ``dataset``; with neither, wikitext-2 (train split,
    blank lines dropped) is used. The corpus is tokenized once and cut into
    consecutive chunks of up to ``_CALIB_SEQ_LEN`` tokens.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    source = "text"
    if text is None:
        dataset = dataset or "wikitext2"
        if dataset not in _CALIB_DATASETS:
            raise ValueError(
                f"Unsupported calibration dataset {dataset!r}. Supported: "
                f"{list(_CALIB_DATASETS)}; or pass text=..."
            )
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "".join(line for line in ds["text"] if line.strip())
        source = dataset

    seq_len = min(_CALIB_SEQ_LEN, getattr(model.config, "max_position_embeddings", None) or _CALIB_SEQ_LEN)
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    chunks = [c.unsqueeze(0) for c in ids.split(seq_len)][:num_samples]
    if not chunks:
        raise ValueError("Calibration corpus tokenized to zero tokens.")
    if source != "text" and len(chunks) < num_samples:
        warnings.warn(
            f"Calibration corpus {source!r} yields {len(chunks)} sequences of "
            f"{seq_len} tokens; {num_samples} were requested.",
            UserWarning,
            stacklevel=3,
        )
    return chunks


def _capture_norm_outputs(
    model,
    tokenizer,
    text: str | None = None,
    dataset: str | None = None,
    num_samples: int = 128,
) -> CalibrationStats:
    """Run the calibration corpus through ``model`` and capture per-channel
    mean-square of each layer's RMSNorm outputs.

    The corpus is cut into up to ``num_samples`` sequences (see
    :func:`_calibration_sequences`); the mean is taken over every token of
    every sequence. Uses forward hooks on ``input_layernorm`` and
    ``post_attention_layernorm`` so the captured tensors are genuine
    post-norm activations (not the pre-norm input). All tensors are moved to
    CPU before returning so callers can keep them around without pinning
    GPU memory. The model's train/eval mode is restored afterwards.
    """
    sequences = _calibration_sequences(model, tokenizer, text, dataset, num_samples)
    device = model.get_input_embeddings().weight.device

    num_layers = len(model.model.layers)
    input_ms: list[torch.Tensor | None] = [None] * num_layers
    post_ms: list[torch.Tensor | None] = [None] * num_layers
    input_tokens = [0] * num_layers
    post_tokens = [0] * num_layers
    hooks = []

    def _make_hook(idx: int, target: list[torch.Tensor | None], counts: list[int]):
        def hook(_module, _inp, out):
            # out: (batch, seq, hidden). Accumulate the per-channel sum of
            # squares; divided by the token count after all sequences.
            y = out.detach().float()
            sq = y.pow(2).sum(dim=(0, 1)).cpu()
            prev = target[idx]
            target[idx] = sq if prev is None else prev + sq
            counts[idx] += y.shape[0] * y.shape[1]
        return hook

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "input_layernorm"):
            hooks.append(layer.input_layernorm.register_forward_hook(_make_hook(i, input_ms, input_tokens)))
        if hasattr(layer, "post_attention_layernorm"):
            hooks.append(layer.post_attention_layernorm.register_forward_hook(_make_hook(i, post_ms, post_tokens)))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for seq in sequences:
                model(seq.to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    for target, counts in ((input_ms, input_tokens), (post_ms, post_tokens)):
        for i, total in enumerate(target):
            if total is not None:
                target[i] = total / counts[i]

    hidden = model.config.hidden_size

    missing = []
    for i in range(num_layers):
        if input_ms[i] is None:
            missing.append((i, "input_layernorm"))
        if post_ms[i] is None:
            missing.append((i, "post_attention_layernorm"))
    if missing:
        warnings.warn(
            f"_capture_norm_outputs: forward hook never fired for "
            f"{len(missing)} layer/norm pairs: {missing}. Their stats will "
            f"be zero, which calibrate() will flag as fully-skipped layers.",
            UserWarning,
            stacklevel=2,
        )

    return CalibrationStats(
        input_norm=[v if v is not None else torch.zeros(hidden) for v in input_ms],
        post_attn_norm=[v if v is not None else torch.zeros(hidden) for v in post_ms],
        layer_origins=_layer_origins(model),
    )


def _is_ollama_id(model_id: str) -> bool:
    """Check if model_id looks like an Ollama model (name:tag, no '/')."""
    return "/" not in model_id and not os.path.isdir(model_id)


def _require_bitsandbytes(mode: str):
    """Import bitsandbytes for the nf4/int8 modes, or raise an actionable error."""
    try:
        import bitsandbytes
    except ImportError as e:
        raise ImportError(
            f"load_model(mode={mode!r}) needs bitsandbytes, which is an optional "
            "dependency. Install it with `pip install 'llm-surgeon[quant]'`, or use "
            "a non-quantized mode (bf16, fp16, fp32, fp32-cpu)."
        ) from e
    return bitsandbytes


def _quantize_in_place(model, bnb_config, device: str | int | torch.device = "cuda:0"):
    """Quantize an in-memory model's Linear layers with BitsAndBytes.

    Wraps each nn.Linear weight as a BnB Params4bit/Int8Params, then moves
    the model to ``device`` (which triggers quantization). No disk
    round-trip needed. ``lm_head`` stays in full precision, matching HF's
    bitsandbytes loader, so GGUF- and HF-sourced nf4 models are comparable.
    """
    is_4bit = getattr(bnb_config, "load_in_4bit", False)
    bnb = _require_bitsandbytes("nf4" if is_4bit else "int8")
    quant_type = getattr(bnb_config, "bnb_4bit_quant_type", "nf4")
    compute_dtype = getattr(bnb_config, "bnb_4bit_compute_dtype", torch.float16)

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model

        w = module.weight.data
        bias_data = module.bias.data if module.bias is not None else None

        if is_4bit:
            new_mod = bnb.nn.Linear4bit(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                compute_dtype=compute_dtype, quant_type=quant_type,
            )
            new_mod.weight = bnb.nn.Params4bit(
                w, requires_grad=False,
                quant_type=quant_type,  # pyright: ignore[reportCallIssue]
                compress_statistics=True,  # pyright: ignore[reportCallIssue]
            )
        else:
            new_mod = bnb.nn.Linear8bitLt(
                module.in_features, module.out_features,
                bias=module.bias is not None, has_fp16_weights=False,
            )
            new_mod.weight = bnb.nn.Int8Params(w, requires_grad=False)

        if bias_data is not None:
            new_mod.bias = nn.Parameter(bias_data)
        setattr(parent, attr, new_mod)

    model = model.to(device)
    model.eval()
    return model


def _gguf_quant_device(
    device_map: str | dict[str, int | str] | None,
    max_memory: dict[int | str, str] | None,
) -> str | int:
    """Device for quantizing a GGUF-loaded model in place (nf4/int8).

    The in-place path moves the whole model to one device, so only a
    single-device ``device_map`` (a device string, or ``{"": device}``) is
    honoured; a per-module map or ``max_memory`` raises instead of being
    silently ignored.
    """
    if max_memory is not None:
        raise ValueError(
            "max_memory is not supported when quantizing an Ollama/GGUF model; "
            "pass device_map='cuda:0' (or {'': 0}) instead."
        )
    if device_map is None or device_map == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Quantizing an Ollama/GGUF model needs a device; CUDA is not "
                "available. Pass device_map='cpu' or use a dense mode."
            )
        return "cuda:0"
    if isinstance(device_map, str):
        return device_map
    if set(device_map) == {""}:
        return device_map[""]
    raise ValueError(
        f"Ollama/GGUF quantization supports only a single-device device_map, got {device_map!r}"
    )


def _is_missing_safetensors(e: OSError) -> bool:
    """True if ``e`` is transformers reporting that no safetensors file exists.

    Other OSErrors that merely mention safetensors (e.g. a corrupt shard)
    must not trigger the pickle ``.bin`` fallback.
    """
    msg = str(e).lower()
    return "safetensors" in msg and (
        "does not appear to have a file named" in msg or "no file named" in msg
    )


_MODE_ALIASES = {"inspect": "nf4", "eval": "fp16", "export": "fp32-cpu"}
VALID_MODES = {"nf4", "int8", "bf16", "fp16", "fp32", "fp32-cpu"}


def load_model(
    model_id: str,
    mode: str = "nf4",
    *,
    revision: str | None = None,
    max_memory: dict[int | str, str] | None = None,
    device_map: str | dict[str, int | str] | None = None,
) -> tuple:
    """Load a model and tokenizer.

    Modes (aliases: inspect=nf4, eval=fp16, export=fp32-cpu):
        nf4:      4-bit NormalFloat, device_map="auto" (smallest; for
                  inspection and structural surgery — weight-level ops such
                  as zero_heads need a dense mode)
        int8:     8-bit LLM.int8(), device_map="auto" (balanced quality/memory)
        bf16:     bfloat16; loads on CPU unless ``device_map`` is given
        fp16:     half precision; loads on CPU unless ``device_map`` is given
        fp32:     full precision; loads on CPU unless ``device_map`` is given
        fp32-cpu: full precision forced to CPU (for export)

    Supports HuggingFace Hub IDs, local paths, and Ollama model IDs
    (e.g. 'tinyllama:latest'). Ollama models are loaded from GGUF and
    dequantized into standard HuggingFace models.

    Args:
        revision: Optional HF Hub commit SHA / branch / tag. Pass to pin an
            experiment to an exact model snapshot. Ignored for local paths
            and Ollama IDs.
        max_memory: Optional accelerate-style budget for the device map
            (e.g. ``{0: "5.5GiB", "cpu": "20GiB"}``). Use to force a
            near-full-fit on a small GPU when the auto-mapper would otherwise
            dispatch layers to CPU (bnb 4-bit can't span CPU+GPU without
            ``llm_int8_enable_fp32_cpu_offload``). Needs a device map: it
            raises ``ValueError`` for fp32-cpu, and for bf16/fp16/fp32
            without ``device_map``.
        device_map: Optional override for the device map. Useful when
            ``"auto"`` would spill bnb-4bit weights to CPU (which bnb
            refuses) — pass ``{"": 0}`` to force the entire model onto
            GPU 0 and OOM-fail-fast otherwise.
    """
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of {sorted(VALID_MODES)}.")
    if mode in ("nf4", "int8"):
        _require_bitsandbytes(mode)

    # Try Ollama resolution for non-HF, non-local model IDs
    if _is_ollama_id(model_id):
        from .gguf_reader import resolve_ollama_blob, load_gguf_as_hf
        blob = resolve_ollama_blob(model_id)
        if blob is not None:
            _GGUF_DTYPE = {
                "nf4": torch.bfloat16, "int8": torch.bfloat16,
                "bf16": torch.bfloat16, "fp16": torch.float16,
                "fp32": torch.float32, "fp32-cpu": torch.float32,
            }
            model, tokenizer = load_gguf_as_hf(blob, dtype=_GGUF_DTYPE.get(mode, torch.float16))
            if mode == "nf4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = _quantize_in_place(
                    model, bnb_config, _gguf_quant_device(device_map, max_memory)
                )
            elif mode == "int8":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = _quantize_in_place(
                    model, bnb_config, _gguf_quant_device(device_map, max_memory)
                )
            return model, tokenizer

    is_local = os.path.isdir(model_id)
    cached = (not is_local) and _is_cached(model_id, revision=revision)

    common_kwargs: dict[str, Any] = {
        "use_safetensors": True,
        "revision": revision,
    }
    if not is_local:
        common_kwargs["cache_dir"] = MODEL_CACHE_DIR
        common_kwargs["local_files_only"] = cached

    mode_kwargs: dict[str, Any]
    if mode == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        mode_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    elif mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        mode_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    elif mode == "fp32-cpu":
        mode_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[mode]
        mode_kwargs = {"torch_dtype": dtype}

    if device_map is not None and mode != "fp32-cpu":
        mode_kwargs["device_map"] = device_map
    if max_memory is not None:
        if mode == "fp32-cpu" or "device_map" not in mode_kwargs:
            raise ValueError(
                f"max_memory needs a device map, which mode {mode!r} does not use "
                "here; pass device_map as well (not supported for fp32-cpu)."
            )
        mode_kwargs["max_memory"] = max_memory

    # Try safetensors first (the secure default — pickle-format .bin can
    # exec arbitrary code on load). On a "safetensors not found" failure,
    # fall back to legacy .bin loading. This handles older Hub models
    # that never shipped safetensors AND local caches with stale
    # `.no_exist/<sha>/model.safetensors` markers (HF's negative cache
    # records "we looked once and the file wasn't there" but doesn't
    # invalidate when the file later appears).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **common_kwargs, **mode_kwargs,
        )
    except OSError as e:
        if not _is_missing_safetensors(e):
            raise
        logger.warning(
            "Model '%s' has no safetensors file accessible — falling back "
            "to legacy .bin format (less safe but expected for older models)",
            model_id,
        )
        retry_kwargs = {k: v for k, v in common_kwargs.items() if k != "use_safetensors"}
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **retry_kwargs, **mode_kwargs,
        )

    # AutoTokenizer does not accept use_safetensors — strip it.
    tok_kwargs = {k: v for k, v in common_kwargs.items() if k != "use_safetensors"}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    return model, tokenizer
