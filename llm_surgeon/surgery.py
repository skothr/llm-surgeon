"""Model loading and layer surgery operations."""

import copy
import gc
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
    text: Optional[str] = None,
    dataset: Optional[str] = None,
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
    baseline_stats: Optional[List[float]] = None,
    text: Optional[str] = None,
    dataset: Optional[str] = None,
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
    text: Optional[str] = None,
    dataset: Optional[str] = None,
    num_samples: int = 128,
) -> List[float]:
    """Capture per-layer input RMS values by running calibration data through the model."""
    if text is None and dataset is None:
        dataset = "wikitext2"

    if text is None:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        available = len(ds["text"])
        if available < num_samples:
            import warnings
            warnings.warn(
                f"_capture_rms: requested {num_samples} samples but "
                f"wikitext-2 train has {available} — RMS stats may be noisy.",
                UserWarning,
                stacklevel=2,
            )
        text = " ".join(ds["text"][:num_samples])

    device = model.get_input_embeddings().weight.device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    num_layers = len(model.model.layers)
    rms_values: List[float] = [0.0] * num_layers
    hooks = []

    def _make_hook(idx):
        def hook(_module, args):
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
                w, requires_grad=False, quant_type=quant_type,
                compress_statistics=True,
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


def load_model(model_id: str, mode: str = "nf4") -> Tuple:
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

    # Determine if model_id is a local path or a HF Hub repo ID
    is_local = os.path.isdir(model_id)
    cache_kwargs = {} if is_local else {"cache_dir": MODEL_CACHE_DIR}

    # If safetensors exist in the snapshot, load from the snapshot directory
    # directly — the hub cache resolver doesn't know about converted files.
    snap = _snapshot_dir(model_id, cache_kwargs.get("cache_dir"))
    if snap and _has_safetensors(model_id, cache_kwargs.get("cache_dir")):
        load_id = str(snap)
        load_kwargs = {"use_safetensors": True}
    else:
        load_id = model_id
        load_kwargs = dict(cache_kwargs)

    # Go offline for this load if the model is cached, but restore env after
    # so callers that expect online semantics aren't silently latched offline.
    _saved_offline = os.environ.get("HF_HUB_OFFLINE")
    if not is_local and snap is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if mode == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                load_id, quantization_config=bnb_config, device_map="auto",
                **load_kwargs,
            )
        elif mode == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                load_id, quantization_config=bnb_config, device_map="auto",
                **load_kwargs,
            )
        elif mode == "bf16":
            model = AutoModelForCausalLM.from_pretrained(
                load_id, torch_dtype=torch.bfloat16,
                **load_kwargs,
            )
        elif mode == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                load_id, torch_dtype=torch.float16,
                **load_kwargs,
            )
        elif mode == "fp32":
            model = AutoModelForCausalLM.from_pretrained(
                load_id, torch_dtype=torch.float32,
                **load_kwargs,
            )
        elif mode == "fp32-cpu":
            model = AutoModelForCausalLM.from_pretrained(
                load_id, torch_dtype=torch.float32, device_map="cpu",
                **load_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: '{mode}'")

        tok_id = str(snap) if snap else model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_id, **({} if snap else cache_kwargs))
    finally:
        if _saved_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = _saved_offline

    return model, tokenizer
