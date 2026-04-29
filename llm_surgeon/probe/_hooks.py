"""Forward-hook helpers shared across probe submodules."""

from __future__ import annotations

import torch

def _get_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device


def _unwrap_hook_output(
    out: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Return the primary tensor from a forward-hook ``output`` argument.

    PyTorch's ``register_forward_hook`` delivers either a single Tensor or
    a tuple whose first element is the primary activation (subsequent
    elements are auxiliaries like attention weights or KV-cache state).
    """
    return out[0] if isinstance(out, tuple) else out


def _make_capture_output_hook(store, key, *, retain_grad: bool = False):
    """Build a forward hook that stores the output tensor at ``store[key]``."""
    def hook(_mod, _inp, out):
        t = _unwrap_hook_output(out)
        if retain_grad and t.requires_grad:
            t.retain_grad()
        store[key] = t
    return hook


def _make_capture_input_hook(store, key, *, retain_grad: bool = False):
    """Build a pre-hook that stores ``args[0]`` (the module input) at ``store[key]``."""
    def hook(_mod, args):
        t = args[0]
        if retain_grad and t.requires_grad:
            t.retain_grad()
        store[key] = t
    return hook


def _attach_reader_grad_hooks(model, store, layers=None):
    """Register pre-hooks to capture pre-norm residual states with retain_grad.

    Stores tensors at keys ``("attn_in", L)`` (input to layer-L's input_layernorm),
    ``("ffn_in", L)`` (input to layer-L's post_attention_layernorm), and
    ``("logits", N)`` (input to the final norm; ``N`` is the layer count).
    Returns the list of hook handles for caller cleanup.
    """
    n = len(model.model.layers)
    target = range(n) if layers is None else layers
    hooks = []
    for L in target:
        layer = model.model.layers[L]
        hooks.append(layer.input_layernorm.register_forward_pre_hook(
            _make_capture_input_hook(store, ("attn_in", L), retain_grad=True)
        ))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            _make_capture_input_hook(store, ("ffn_in", L), retain_grad=True)
        ))
    hooks.append(model.model.norm.register_forward_pre_hook(
        _make_capture_input_hook(store, ("logits", n), retain_grad=True)
    ))
    return hooks

