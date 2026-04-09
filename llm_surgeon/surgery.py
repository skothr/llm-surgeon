"""Model loading and layer surgery operations."""

import copy
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List

import torch.nn as nn


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
