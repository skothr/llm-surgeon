"""Forward-pass residual-stream capture.

The non-grad ``_capture_residual_stream`` is the lightweight path used
by logit-lens and intervention; ``_capture_residual_stream_with_grad``
keeps tensors attached to the autograd graph for attribution patching.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from llm_surgeon.probe._hooks import (
    _attach_reader_grad_hooks,
    _get_input_device,
    _make_capture_input_hook,
    _make_capture_output_hook,
    _unwrap_hook_output,
)

_CAPTURE_SUBLAYERS = frozenset({"attn", "ffn", "embed"})


def _resolve_layers(layers: Iterable[int] | None, num_layers: int) -> list[int]:
    """Normalize a layer subset to sorted, unique, non-negative indices.

    Negative indices count from the end (``-1`` is the last layer), as in
    ``model.model.layers[i]``. Out-of-range indices raise IndexError; without
    this, a negative index captured under key ``(-1, ...)`` and matched no
    intervention hook.
    """
    if layers is None:
        return list(range(num_layers))
    resolved = set()
    for layer in layers:
        if not -num_layers <= layer < num_layers:
            raise IndexError(f"layer {layer} out of range for {num_layers} layers")
        resolved.add(layer % num_layers)
    return sorted(resolved)


def _capture_residual_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    sublayers: tuple[str, ...] = ("ffn",),
    layers: Iterable[int] | None = None,
) -> tuple[dict[tuple[int, str], torch.Tensor], list[str]]:
    """Run a forward pass and capture residual stream states via hooks.

    ``layers`` accepts negative indices (normalized to ``num_layers + i``);
    ``sublayers`` must be a subset of {"attn", "ffn", "embed"}.

    Returns:
        captured: dict mapping (layer_idx, sublayer_name) -> Tensor (seq_len, d_model)
        prompt_tokens: list of token strings

    Sublayer keys:
      - "attn": h_in + attn_out at each (target) layer, keyed (L, "attn")
      - "ffn":  layer output at each (target) layer, keyed (L, "ffn")
      - "embed": output of model.model.embed_tokens (residual stream
                 BEFORE any layer modifies it), keyed (0, "embed")
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    unknown = set(sublayers) - _CAPTURE_SUBLAYERS
    if unknown:
        raise ValueError(
            f"unknown sublayer(s) {sorted(unknown)}; expected a subset of "
            f"{sorted(_CAPTURE_SUBLAYERS)}"
        )
    num_layers = len(model.model.layers)
    target_layers = _resolve_layers(layers, num_layers)
    capture_attn = "attn" in sublayers
    capture_ffn = "ffn" in sublayers
    capture_embed = "embed" in sublayers

    captured: dict[tuple[int, str], torch.Tensor] = {}
    layer_block_inputs: dict[int, torch.Tensor] = {}
    hooks: list[RemovableHandle] = []

    if capture_embed:
        def embed_hook(_mod, _inp, out):
            captured[(0, "embed")] = _unwrap_hook_output(out).detach()[0]
        hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    def make_block_pre_hook(idx):
        def hook(_mod, args):
            layer_block_inputs[idx] = args[0].detach()
        return hook

    def make_attn_hook(idx):
        def hook(_mod, _inp, out):
            attn_out = _unwrap_hook_output(out)
            captured[(idx, "attn")] = (layer_block_inputs[idx] + attn_out.detach())[0]
        return hook

    def make_ffn_hook(idx):
        def hook(_mod, _inp, out):
            captured[(idx, "ffn")] = _unwrap_hook_output(out).detach()[0]
        return hook

    for i in target_layers:
        layer = model.model.layers[i]
        if capture_attn:
            hooks.append(layer.register_forward_pre_hook(make_block_pre_hook(i)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        if capture_ffn:
            hooks.append(layer.register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured, prompt_tokens


def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: tuple[str, ...] = ("attn", "ffn"),
    layers: list[int] | None = None,
    capture_concat_z: bool = False,
    capture_reader_grads: bool = False,
    capture_ffn_out: bool = False,
    capture_ffn_act: bool = False,
) -> tuple[
    dict[tuple[int, str], torch.Tensor],
    dict[int, torch.Tensor],
    torch.Tensor,
    list[str],
    dict[int, torch.Tensor],
    dict[tuple, torch.Tensor],
    dict[int, torch.Tensor],
]:
    """Capture residual-stream states with autograd graph intact.

    Mirrors _capture_residual_stream but keeps tensors attached to the graph
    so a downstream .backward() populates .grad on each captured tensor.
    Caller is responsible for providing a torch.enable_grad() context.

    For "attn" rows, `captured[(L, "attn")]` stores the self_attn output (the
    attn delta), which IS in the graph and accepts retain_grad(). The
    corresponding residual-stream-post-attn value `h_post_attn = h_in + attn_out`
    cannot be intercepted as a live graph node (LLaMA's layer.forward computes
    it as an ephemeral intermediate), so callers must reconstruct it from
    `h_ins[L] + captured[(L, "attn")]`. The gradient `captured[(L,"attn")].grad`
    equals `∂L/∂h_post_attn` by chain rule through `+` (since
    `h_post_attn = h_in + attn_out` and h_in is not a function of attn_out),
    so using attn_out.grad as the gradient of h_post_attn is correct.

    For "ffn" rows, `captured[(L, "ffn")]` is the decoder layer's full output
    (residual stream post-layer) and matches exact AP's ffn-row semantics.

    ``layers`` accepts negative indices (normalized to ``num_layers + i``).

    Returns: (captured_states, h_ins, output_logits, prompt_tokens,
              concat_z_captured, reader_inputs, ffn_acts).
        concat_z_captured is empty when capture_concat_z=False, and also
        when "attn" is not in sublayers (concat_z is only captured alongside
        the attn rows).
        reader_inputs is empty when capture_reader_grads=False; otherwise
        holds pre-LN residual tensors keyed by ("attn_in", L), ("ffn_in", L),
        ("logits", N_L) with retain_grad() called so .grad is populated after
        backward().
        When capture_ffn_out=True, captured also contains (L, "ffn_out") keys
        holding the raw MLP output before the residual add; retain_grad() is
        called on the ffn_out tensor so .grad is populated after backward()
        (needed for per-neuron attribution).
        ffn_acts is empty when capture_ffn_act=False; otherwise holds the
        input tensor to each mlp.down_proj (i.e. the MLP intermediate
        activation, shape [batch, seq, intermediate_size]) keyed by layer
        index.
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Build inputs_embeds as a grad-tracking leaf (in enable_grad
    # contexts) so the autograd graph has somewhere to anchor when the
    # caller has frozen all model parameters via requires_grad_(False).
    # Callers may freeze params (requires_grad_(False)) for memory and
    # safety; without an input that requires grad, every
    # downstream tensor would inherit requires_grad=False and
    # `metric.backward()` would raise "element 0 of tensors does not
    # require grad". Using inputs_embeds (rather than enabling grads on
    # all params) keeps memory cost at one extra embedding tensor
    # instead of an entire parameter-grad set — critical for 3B+ models
    # on consumer GPUs (RTX 2080 = 8 GB).
    #
    # Test mocks that don't extend PreTrainedModel have no
    # get_input_embeddings; they typically have grad-enabled params already,
    # so the input_ids path works for them. Only the attribute lookup falls
    # back: an AttributeError raised inside the embedding forward propagates.
    inputs_embeds: torch.Tensor | None = None
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    embed_layer = get_input_embeddings() if get_input_embeddings is not None else None
    if embed_layer is not None:
        embedded: torch.Tensor = embed_layer(input_ids)
        if torch.is_grad_enabled():
            embedded = embedded.detach().requires_grad_(True)
        inputs_embeds = embedded

    num_layers = len(model.model.layers)
    target_layers = set(_resolve_layers(layers, num_layers))

    captured: dict[tuple[int, str], torch.Tensor] = {}
    h_ins: dict[int, torch.Tensor] = {}
    concat_z_captured: dict[int, torch.Tensor] = {}
    reader_inputs: dict[tuple, torch.Tensor] = {}
    ffn_acts: dict[int, torch.Tensor] = {}
    hooks: list = []

    # Pre-hook captures layer input (h_in). Needed for the attn-row value
    # reconstruction (h_post_attn = h_in + attn_out). Always register when
    # "attn" is targeted; cost is trivial.
    capture_h_in = "attn" in sublayers

    for i in range(num_layers):
        if i not in target_layers:
            continue
        layer = model.model.layers[i]

        if capture_h_in:
            hooks.append(layer.register_forward_pre_hook(
                _make_capture_input_hook(h_ins, i)
            ))
        if "attn" in sublayers:
            hooks.append(layer.self_attn.register_forward_hook(
                _make_capture_output_hook(captured, (i, "attn"), retain_grad=True)
            ))
        if "ffn" in sublayers:
            hooks.append(layer.register_forward_hook(
                _make_capture_output_hook(captured, (i, "ffn"), retain_grad=True)
            ))
        if capture_concat_z and "attn" in sublayers:
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
                _make_capture_input_hook(concat_z_captured, i, retain_grad=True)
            ))
        if capture_ffn_out:
            hooks.append(layer.mlp.register_forward_hook(
                _make_capture_output_hook(captured, (i, "ffn_out"), retain_grad=True)
            ))
        if capture_ffn_act:
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                _make_capture_input_hook(ffn_acts, i, retain_grad=True)
            ))

    if capture_reader_grads:
        hooks.extend(_attach_reader_grad_hooks(model, reader_inputs, layers=target_layers))

    try:
        if inputs_embeds is not None:
            model_output = model(inputs_embeds=inputs_embeds)
        else:
            model_output = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured, h_ins, model_output.logits[0], prompt_tokens, concat_z_captured, reader_inputs, ffn_acts

