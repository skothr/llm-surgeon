"""Model loading and layer surgery operations."""

import copy
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def load_model(model_id: str, mode: str = "inspect") -> Tuple:
    """Load a model and tokenizer.

    Modes:
        inspect: 4-bit quantized on GPU (fast, for inspection)
        eval: fp16 with auto device map (for perplexity measurement)
        export: fp16 on CPU only (for clean checkpoint export)
    """
    if mode not in ("inspect", "eval", "export"):
        raise ValueError(f"Unknown mode: '{mode}'. Must be 'inspect', 'eval', or 'export'.")

    if mode == "inspect":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
    elif mode == "eval":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        )
    elif mode == "export":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
