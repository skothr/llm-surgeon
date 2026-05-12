"""Model loading and layer surgery operations."""

import copy
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("llm_surgeon.surgery")

# Dedicated cache for clean HF model downloads.
# Override with LLM_SURGEON_CACHE_DIR env var.
MODEL_CACHE_DIR = os.environ.get(
    "LLM_SURGEON_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".cache" / "models"),
)

# Threshold above which duplicate_layer issues a memory warning. Sized for a
# 32 GB host with headroom for activations + Python overhead.
_FP16_GB_WARN_THRESHOLD = 28.0


def _is_cached(model_id: str, cache_dir: str | None = None) -> bool:
    """True if a local HF cache has at least a config.json snapshot for model_id.

    Wraps huggingface_hub.try_to_load_from_cache. Used as the boolean probe
    that replaces _snapshot_dir for "is this model present locally?".
    """
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(
        model_id, filename="config.json",
        cache_dir=cache_dir or MODEL_CACHE_DIR,
    )
    return path is not None


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
    """Keep only the layers at the specified indices, remove all others."""
    layers = model.model.layers
    num_before = len(layers)

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


def _head_dim(model) -> int:
    return model.config.hidden_size // model.config.num_attention_heads


def zero_heads(model, layer: int, heads: list[int]) -> SurgeryLog:
    """Zero out specific attention heads by zeroing their o_proj columns.

    The head still exists structurally but contributes nothing to the
    residual stream. This is the standard ablation approach.
    """
    _validate_head_args(model, layer, heads)
    hd = _head_dim(model)
    o_proj = model.model.layers[layer].self_attn.o_proj
    with torch.no_grad():
        for h in heads:
            o_proj.weight.data[:, h * hd : (h + 1) * hd] = 0

    return SurgeryLog.inplace(model, "zero_heads", f"Zeroed heads {heads} in layer {layer}")


def scale_heads(model, layer: int, heads: list[int], factor: float) -> SurgeryLog:
    """Scale specific heads' contribution by multiplying their o_proj columns."""
    _validate_head_args(model, layer, heads)
    hd = _head_dim(model)
    o_proj = model.model.layers[layer].self_attn.o_proj
    with torch.no_grad():
        for h in heads:
            o_proj.weight.data[:, h * hd : (h + 1) * hd] *= factor

    return SurgeryLog.inplace(
        model, "scale_heads", f"Scaled heads {heads} in layer {layer} by {factor}"
    )


def swap_heads(model, layer: int, h1: int, h2: int) -> SurgeryLog:
    """Exchange two heads' weight slices in q_proj, k_proj, v_proj, and o_proj."""
    _validate_head_args(model, layer, [h1, h2])
    hd = _head_dim(model)
    attn = model.model.layers[layer].self_attn

    with torch.no_grad():
        # Swap q_proj rows (each head's query projection)
        q = attn.q_proj.weight.data
        q[h1 * hd : (h1 + 1) * hd, :], q[h2 * hd : (h2 + 1) * hd, :] = (
            q[h2 * hd : (h2 + 1) * hd, :].clone(),
            q[h1 * hd : (h1 + 1) * hd, :].clone(),
        )

        # Swap k_proj and v_proj rows if heads map 1:1 to KV heads
        # (GQA: multiple Q heads share one KV head — only swap if they map to different KV heads)
        num_kv_heads = model.config.num_key_value_heads
        num_q_heads = model.config.num_attention_heads
        kv_group_size = num_q_heads // num_kv_heads
        kv1 = h1 // kv_group_size
        kv2 = h2 // kv_group_size
        if kv1 != kv2:
            for proj in [attn.k_proj, attn.v_proj]:
                w = proj.weight.data
                kv_hd = w.shape[0] // num_kv_heads
                w[kv1 * kv_hd : (kv1 + 1) * kv_hd, :], w[kv2 * kv_hd : (kv2 + 1) * kv_hd, :] = (
                    w[kv2 * kv_hd : (kv2 + 1) * kv_hd, :].clone(),
                    w[kv1 * kv_hd : (kv1 + 1) * kv_hd, :].clone(),
                )

        # Swap o_proj columns (each head's output contribution)
        o = attn.o_proj.weight.data
        o[:, h1 * hd : (h1 + 1) * hd], o[:, h2 * hd : (h2 + 1) * hd] = (
            o[:, h2 * hd : (h2 + 1) * hd].clone(),
            o[:, h1 * hd : (h1 + 1) * hd].clone(),
        )

    return SurgeryLog.inplace(model, "swap_heads", f"Swapped heads {h1} and {h2} in layer {layer}")


def zero_mlp(model, layer: int) -> SurgeryLog:
    """Zero out a layer's MLP by zeroing down_proj weights.

    The MLP still exists structurally but contributes nothing to the
    residual stream (the residual connection passes through unchanged).
    """
    num_layers = len(model.model.layers)
    if layer < 0 or layer >= num_layers:
        raise IndexError(f"Layer index {layer} out of range [0, {num_layers - 1}]")
    with torch.no_grad():
        model.model.layers[layer].mlp.down_proj.weight.data.zero_()
    return SurgeryLog.inplace(model, "zero_mlp", f"Zeroed MLP in layer {layer}")


def zero_attention(model, layer: int) -> SurgeryLog:
    """Zero out a layer's entire attention by zeroing o_proj weights.

    The attention module still exists structurally but contributes nothing
    to the residual stream.
    """
    num_layers = len(model.model.layers)
    if layer < 0 or layer >= num_layers:
        raise IndexError(f"Layer index {layer} out of range [0, {num_layers - 1}]")
    with torch.no_grad():
        model.model.layers[layer].self_attn.o_proj.weight.data.zero_()
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
    """
    input_norm: list[torch.Tensor]
    post_attn_norm: list[torch.Tensor]

    @property
    def num_layers(self) -> int:
        return len(self.input_norm)

    def __len__(self) -> int:
        return self.num_layers


@dataclass
class CalibrationReport:
    """Summary of a :func:`calibrate` run.

    Attributes:
        layers_calibrated: Count of layer/norm pairs whose weight was rescaled.
        channels_clipped: Total channels whose scale hit the clip bounds
            (indicates severe variance mismatch — often a dead channel).
        channels_skipped: Channels skipped due to sub-threshold variance in
            either baseline or current stats.
        per_layer_scale_mean: Mean of the applied scale vector per layer/norm
            (diagnostic — values far from 1.0 indicate large drift).
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
    element-wise by ``sqrt(baseline_mean_sq / current_mean_sq)``. That is
    the exact scalar that restores each channel's post-norm magnitude to
    its pre-surgery value (since RMSNorm's scale is linear in the gain).

    This does NOT correct directional drift in the residual stream — layer
    removal changes the direction of activations, and no amount of gain
    scaling fixes that. The goal is more modest: keep each channel's
    post-norm output in the magnitude regime the downstream layer was
    trained on, so non-linearities and attention softmaxes don't saturate.

    Args:
        model: The surgically-modified model to calibrate.
        tokenizer: Tokenizer matching the model.
        baseline_stats: :class:`CalibrationStats` from the pre-surgery model.
            If ``None``, calibration is skipped with a warning.
        layer_map: Maps current-model layer index → baseline layer index. Use
            this when ``remove_layers`` or ``reorder_layers`` shifted the
            correspondence. If ``None``, uses identity mapping up to the
            shorter depth (assumes no reordering).
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
    report = CalibrationReport()
    if baseline_stats is None:
        warnings.warn(
            "calibrate() called without baseline_stats. "
            "Call capture_calibration_stats() on the original model before surgery, "
            "then pass the result here. Skipping calibration.",
            UserWarning,
        )
        return report

    current_stats = _capture_norm_outputs(
        model, tokenizer, text=text, dataset=dataset, num_samples=num_samples
    )

    current_layers = model.model.layers
    if layer_map is None:
        layer_map = list(range(min(len(current_layers), baseline_stats.num_layers)))

    for cur_idx, base_idx in enumerate(layer_map):
        if cur_idx >= len(current_layers):
            break
        if base_idx >= baseline_stats.num_layers:
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


def _capture_norm_outputs(
    model,
    tokenizer,
    text: str | None = None,
    dataset: str | None = None,
    num_samples: int = 128,
) -> CalibrationStats:
    """Run the calibration corpus through ``model`` and capture per-channel
    mean-square of each layer's RMSNorm outputs.

    Uses forward hooks on ``input_layernorm`` and ``post_attention_layernorm``
    so the captured tensors are genuine post-norm activations (not the
    pre-norm input). All tensors are moved to CPU before returning so callers
    can keep them around without pinning GPU memory.
    """
    if text is None and dataset is None:
        dataset = "wikitext2"

    if text is None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        available = len(ds["text"])
        if available < num_samples:
            warnings.warn(
                f"_capture_norm_outputs: requested {num_samples} samples but "
                f"wikitext-2 train has {available} — calibration stats may be noisy.",
                UserWarning,
                stacklevel=2,
            )
        text = " ".join(ds["text"][:num_samples])

    device = model.get_input_embeddings().weight.device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    num_layers = len(model.model.layers)
    input_ms: list[torch.Tensor | None] = [None] * num_layers
    post_ms: list[torch.Tensor | None] = [None] * num_layers
    hooks = []

    def _make_hook(idx: int, target: list[torch.Tensor | None]):
        def hook(_module, _inp, out):
            # out: (batch, seq, hidden). Per-channel mean-square over (batch, seq).
            y = out.detach().float()
            target[idx] = y.pow(2).mean(dim=(0, 1)).cpu()
        return hook

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "input_layernorm"):
            hooks.append(layer.input_layernorm.register_forward_hook(_make_hook(i, input_ms)))
        if hasattr(layer, "post_attention_layernorm"):
            hooks.append(layer.post_attention_layernorm.register_forward_hook(_make_hook(i, post_ms)))

    try:
        model.eval()
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

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
    )


def _is_ollama_id(model_id: str) -> bool:
    """Check if model_id looks like an Ollama model (name:tag, no '/')."""
    return "/" not in model_id and not os.path.isdir(model_id)


def _quantize_in_place(model, bnb_config):
    """Quantize an in-memory model's Linear layers with BitsAndBytes.

    Wraps each nn.Linear weight as a BnB Params4bit/Int8Params, then moves
    to GPU (which triggers quantization). No disk round-trip needed.
    """
    import bitsandbytes as bnb
    is_4bit = getattr(bnb_config, "load_in_4bit", False)
    quant_type = getattr(bnb_config, "bnb_4bit_quant_type", "nf4")
    compute_dtype = getattr(bnb_config, "bnb_4bit_compute_dtype", torch.float16)

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
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

    model = model.to("cuda:0")
    model.eval()
    return model


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

    Modes:
        nf4:      4-bit NormalFloat on GPU (smallest, for surgery/inspection)
        int8:     8-bit LLM.int8() on GPU (balanced quality/memory)
        fp16:     half-precision with auto device map
        fp32:     full precision with auto device map
        fp32-cpu: full precision forced to CPU (for export)

    Supports HuggingFace Hub IDs, local paths, and Ollama model IDs
    (e.g. 'tinyllama:latest'). Ollama models are loaded from GGUF and
    dequantized into standard HuggingFace models.

    Args:
        revision: Optional HF Hub commit SHA / branch / tag. Pass to pin an
            experiment to an exact model snapshot. Ignored for local paths
            and Ollama IDs.
        max_memory: Optional accelerate-style budget passed to
            ``device_map="auto"`` (e.g. ``{0: "5.5GiB", "cpu": "20GiB"}``).
            Use to force a near-full-fit on a small GPU when the auto-mapper
            would otherwise dispatch layers to CPU (bnb 4-bit can't span
            CPU+GPU without ``llm_int8_enable_fp32_cpu_offload``).
        device_map: Optional override for the device map. Useful when
            ``"auto"`` would spill bnb-4bit weights to CPU (which bnb
            refuses) — pass ``{"": 0}`` to force the entire model onto
            GPU 0 and OOM-fail-fast otherwise.
    """
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of {sorted(VALID_MODES)}.")

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
                model = _quantize_in_place(model, bnb_config)
            elif mode == "int8":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = _quantize_in_place(model, bnb_config)
            return model, tokenizer

    is_local = os.path.isdir(model_id)
    cached = (not is_local) and _is_cached(model_id)

    common_kwargs: dict[str, Any] = {
        "use_safetensors": True,
        "revision": revision,
    }
    if not is_local:
        common_kwargs["cache_dir"] = MODEL_CACHE_DIR
        common_kwargs["local_files_only"] = cached

    if mode == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        mode_kwargs: dict[str, Any] = {"quantization_config": bnb_config, "device_map": "auto"}
        if max_memory is not None:
            mode_kwargs["max_memory"] = max_memory
        if device_map is not None:
            mode_kwargs["device_map"] = device_map
    elif mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        mode_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        if max_memory is not None:
            mode_kwargs["max_memory"] = max_memory
        if device_map is not None:
            mode_kwargs["device_map"] = device_map
    elif mode == "bf16":
        mode_kwargs = {"torch_dtype": torch.bfloat16}
        if device_map is not None:
            mode_kwargs["device_map"] = device_map
    elif mode == "fp16":
        mode_kwargs = {"torch_dtype": torch.float16}
        if device_map is not None:
            mode_kwargs["device_map"] = device_map
    elif mode == "fp32":
        mode_kwargs = {"torch_dtype": torch.float32}
        if device_map is not None:
            mode_kwargs["device_map"] = device_map
    elif mode == "fp32-cpu":
        mode_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

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
    except (OSError, EnvironmentError) as e:
        if "safetensors" not in str(e).lower():
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
