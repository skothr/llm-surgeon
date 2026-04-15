"""Model loading and layer surgery operations."""

import copy
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Dedicated cache for clean HF model downloads.
# Override with LLM_SURGEON_CACHE_DIR env var.
MODEL_CACHE_DIR = os.environ.get(
    "LLM_SURGEON_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".cache" / "models"),
)


def _snapshot_dir(model_id: str, cache_dir: str | None = None) -> Path | None:
    """Resolve the snapshot directory for a cached HF model."""
    base = Path(cache_dir or MODEL_CACHE_DIR)
    slug = "models--" + model_id.replace("/", "--")
    model_dir = base / slug
    if not model_dir.exists():
        return None
    refs = model_dir / "refs" / "main"
    if refs.exists():
        sha = refs.read_text().strip()
        snap = model_dir / "snapshots" / sha
        if snap.exists():
            return snap
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        children = sorted(snapshots.iterdir())
        if children:
            return children[-1]
    return None


def _has_safetensors(model_id: str, cache_dir: str | None = None) -> bool:
    """Check if a cached model has safetensors files."""
    if os.path.isdir(model_id):
        return any(Path(model_id).glob("*.safetensors"))
    snap = _snapshot_dir(model_id, cache_dir)
    if snap is None:
        return False
    return any(snap.glob("*.safetensors"))


def convert_to_safetensors(model_id: str, cache_dir: str | None = None) -> dict:
    """Convert a cached model from .bin to safetensors format in-place.

    Loads the model on CPU, saves as safetensors into the same snapshot
    directory, then removes the old .bin files.

    Returns dict with conversion stats.
    """
    snap = _snapshot_dir(model_id, cache_dir)
    if snap is None:
        raise ValueError(f"No cached snapshot found for '{model_id}'")
    if any(snap.glob("*.safetensors")):
        return {"status": "already_safetensors", "model_id": model_id}

    bin_files = list(snap.glob("*.bin"))
    if not bin_files:
        raise ValueError(f"No .bin files found in {snap}")

    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(snap), torch_dtype=torch.float16, device_map="cpu",
        )
        model.save_pretrained(str(snap), safe_serialization=True)
        del model
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline

    bin_size = sum(f.stat().st_size for f in bin_files)
    for f in bin_files:
        f.unlink()

    safetensor_files = list(snap.glob("*.safetensors"))
    st_size = sum(f.stat().st_size for f in safetensor_files)

    return {
        "status": "converted",
        "model_id": model_id,
        "removed_bin_files": len(bin_files),
        "removed_bin_mb": round(bin_size / 1e6, 1),
        "safetensor_files": len(safetensor_files),
        "safetensor_mb": round(st_size / 1e6, 1),
    }


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
    ops: List[SurgeryOp] = field(default_factory=list)

    def add(self, operation: str, description: str, before: int, after: int) -> None:
        self.ops.append(SurgeryOp(operation, description, before, after))

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


def get_layer_info(model) -> Dict[str, Any]:
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


def remove_layers(model, layer_indices: List[int]) -> SurgeryLog:
    """Remove layers at the specified indices. Indices are current positions."""
    layers = model.model.layers
    num_before = len(layers)

    for idx in layer_indices:
        if idx < 0 or idx >= num_before:
            raise IndexError(f"Layer index {idx} out of range [0, {num_before - 1}]")

    for idx in sorted(layer_indices, reverse=True):
        del layers[idx]

    model.config.num_hidden_layers = len(layers)
    _renumber_layers(model)

    log = SurgeryLog()
    log.add("remove_layers", f"Removed layers {layer_indices}", num_before, len(layers))
    return log


def keep_layers(model, layer_indices: List[int]) -> SurgeryLog:
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

    log = SurgeryLog()
    log.add("keep_layers", f"Kept layers {layer_indices}", num_before, len(new_layers))
    return log


def reorder_layers(model, new_order: List[int]) -> SurgeryLog:
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

    log = SurgeryLog()
    log.add("reorder_layers", f"Reordered to {new_order}", num_before, len(new_layers))
    return log


def swap_layers(model, i: int, j: int) -> SurgeryLog:
    """Swap two layers' positions."""
    layers = model.model.layers
    num_before = len(layers)

    for idx in (i, j):
        if idx < 0 or idx >= num_before:
            raise IndexError(f"Layer index {idx} out of range [0, {num_before - 1}]")

    layers[i], layers[j] = layers[j], layers[i]
    _renumber_layers(model)

    log = SurgeryLog()
    log.add("swap_layers", f"Swapped layers {i} and {j}", num_before, len(layers))
    return log


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
    if est_gb > 28:
        warnings.warn(
            f"Model is ~{est_gb:.1f} GB in fp16, approaching 32 GB RAM limit. "
            f"Duplicating a layer will increase this.",
            ResourceWarning,
        )

    new_layer = copy.deepcopy(layers[src])
    layers.insert(dst, new_layer)
    model.config.num_hidden_layers = len(layers)
    _renumber_layers(model)

    log = SurgeryLog()
    log.add("duplicate_layer", f"Duplicated layer {src} -> position {dst}", num_before, len(layers))
    return log


# ---------------------------------------------------------------------------
# Attention head surgery
# ---------------------------------------------------------------------------

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


def zero_heads(model, layer: int, heads: List[int]) -> SurgeryLog:
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

    log = SurgeryLog()
    log.add("zero_heads", f"Zeroed heads {heads} in layer {layer}", len(model.model.layers), len(model.model.layers))
    return log


def scale_heads(model, layer: int, heads: List[int], factor: float) -> SurgeryLog:
    """Scale specific heads' contribution by multiplying their o_proj columns."""
    _validate_head_args(model, layer, heads)
    hd = _head_dim(model)
    o_proj = model.model.layers[layer].self_attn.o_proj
    with torch.no_grad():
        for h in heads:
            o_proj.weight.data[:, h * hd : (h + 1) * hd] *= factor

    log = SurgeryLog()
    log.add("scale_heads", f"Scaled heads {heads} in layer {layer} by {factor}", len(model.model.layers), len(model.model.layers))
    return log


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

    log = SurgeryLog()
    log.add("swap_heads", f"Swapped heads {h1} and {h2} in layer {layer}", len(model.model.layers), len(model.model.layers))
    return log


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
    log = SurgeryLog()
    log.add("zero_mlp", f"Zeroed MLP in layer {layer}", num_layers, num_layers)
    return log


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
    log = SurgeryLog()
    log.add("zero_attention", f"Zeroed attention in layer {layer}", num_layers, num_layers)
    return log


def capture_calibration_stats(
    model,
    tokenizer,
    text: str = None,
    dataset: str = None,
    num_samples: int = 128,
) -> List[float]:
    """Capture per-layer input RMS values for calibration reference.

    Run this BEFORE surgery on the original model. The returned stats are
    passed to calibrate() after surgery to correct for distribution shift.

    Returns:
        List of per-layer RMS values (one float per layer).
    """
    return _capture_rms(model, tokenizer, text=text, dataset=dataset, num_samples=num_samples)


def calibrate(
    model,
    tokenizer,
    baseline_stats: List[float] = None,
    text: str = None,
    dataset: str = None,
    num_samples: int = 128,
) -> None:
    """Rescale RMSNorm gains to compensate for residual stream shift after surgery.

    Uses a ratio-based approach: compares the per-layer input RMS of the
    modified model against baseline stats from the original model, then
    scales each layer's input_layernorm.weight by (baseline_rms / current_rms).
    This ensures downstream layers see the same magnitude they expect.

    Args:
        model: The modified model to calibrate.
        tokenizer: Tokenizer matching the model.
        baseline_stats: Per-layer RMS from capture_calibration_stats() on the
            original model. If None, calibration is skipped with a warning.
        text: Calibration text. If None, loads from dataset.
        dataset: Dataset name ('wikitext2'). Used if text is None.
        num_samples: Number of samples from dataset.
    """
    if baseline_stats is None:
        warnings.warn(
            "calibrate() called without baseline_stats. "
            "Call capture_calibration_stats() on the original model before surgery, "
            "then pass the result here. Skipping calibration.",
            UserWarning,
        )
        return

    # Capture current RMS values on the modified model
    current_stats = _capture_rms(model, tokenizer, text=text, dataset=dataset, num_samples=num_samples)

    # Apply scalar correction per layer: scale weight by baseline/current ratio
    num_to_correct = min(len(baseline_stats), len(current_stats), len(model.model.layers))
    for i in range(num_to_correct):
        layer = model.model.layers[i]
        if not hasattr(layer, "input_layernorm"):
            continue

        baseline_rms = baseline_stats[i]
        current_rms = current_stats[i]

        if current_rms < 1e-8:
            continue

        ratio = baseline_rms / current_rms
        norm_weight = layer.input_layernorm.weight
        with torch.no_grad():
            norm_weight.data *= torch.tensor(ratio, device=norm_weight.device, dtype=norm_weight.dtype)


def _capture_rms(
    model,
    tokenizer,
    text: str = None,
    dataset: str = None,
    num_samples: int = 128,
) -> List[float]:
    """Capture per-layer input RMS values by running calibration data through the model."""
    if text is None and dataset is None:
        dataset = "wikitext2"

    if text is None:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = " ".join(ds["text"][:num_samples])

    device = model.model.embed_tokens.weight.device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    num_layers = len(model.model.layers)
    rms_values: List[float] = [0.0] * num_layers
    hooks = []

    def _make_hook(idx):
        def hook(module, args):
            hs = args[0].detach().float()  # (batch, seq, hidden)
            # RMS across all positions: sqrt(mean(x^2))
            rms = hs.pow(2).mean().sqrt().item()
            rms_values[idx] = rms
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(_make_hook(i)))

    try:
        model.eval()
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return rms_values


def load_model(model_id: str, mode: str = "inspect") -> Tuple:
    """Load a model and tokenizer.

    Modes:
        inspect: 4-bit quantized on GPU (fast, for inspection)
        eval: fp16 with auto device map (for perplexity measurement)
        export: fp16 on CPU only (for clean checkpoint export)

    For HF Hub model IDs (not local paths), downloads are cached in
    MODEL_CACHE_DIR. Subsequent loads use the cache without network access.
    """
    if mode not in ("inspect", "eval", "export"):
        raise ValueError(f"Unknown mode: '{mode}'. Must be 'inspect', 'eval', or 'export'.")

    # Determine if model_id is a local path or a HF Hub repo ID
    is_local = os.path.isdir(model_id)
    cache_kwargs = {} if is_local else {"cache_dir": MODEL_CACHE_DIR}

    # Check if model is already cached — if so, go offline to prevent
    # unnecessary network requests (auto-conversion, telemetry, etc.)
    if not is_local and _snapshot_dir(model_id, cache_kwargs.get("cache_dir")) is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # If safetensors exist in the snapshot, load from the snapshot directory
    # directly — the hub cache resolver doesn't know about converted files.
    snap = _snapshot_dir(model_id, cache_kwargs.get("cache_dir"))
    if snap and _has_safetensors(model_id, cache_kwargs.get("cache_dir")):
        load_id = str(snap)
        load_kwargs = {"use_safetensors": True}
    else:
        load_id = model_id
        load_kwargs = dict(cache_kwargs)

    if mode == "inspect":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            load_id, quantization_config=bnb_config, device_map="auto",
            **load_kwargs,
        )
    elif mode == "eval":
        model = AutoModelForCausalLM.from_pretrained(
            load_id, dtype=torch.float16, device_map="auto",
            **load_kwargs,
        )
    elif mode == "export":
        model = AutoModelForCausalLM.from_pretrained(
            load_id, dtype=torch.float16, device_map="cpu",
            **load_kwargs,
        )

    tok_id = str(snap) if snap else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, **({} if snap else cache_kwargs))

    # Ensure offline mode for all subsequent loads in this process
    os.environ["HF_HUB_OFFLINE"] = "1"

    return model, tokenizer
