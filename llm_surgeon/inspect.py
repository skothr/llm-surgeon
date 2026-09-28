"""Inspection and activation analysis tools for LLaMA models."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from llm_surgeon.probe._hooks import (
    _make_capture_input_hook,
    _make_capture_output_hook,
    _unwrap_hook_output,
)


def _get_input_device(model) -> torch.device:
    """Get the device where input_ids should be sent (embedding layer's device)."""
    return model.get_input_embeddings().weight.device


def _raise_if_missing(fired: Iterable[int], num_layers: int) -> None:
    """Raise if a layer's forward hook did not fire during the forward pass."""
    seen = set(fired)
    missing = [i for i in range(num_layers) if i not in seen]
    if missing:
        raise RuntimeError(
            f"Forward hooks did not fire for layer(s) {missing} of {num_layers}; "
            f"the model's forward skipped them"
        )


@contextmanager
def _eager_attention(model) -> Iterator[None]:
    """Switch the model to eager attention (sdpa cannot return attention
    weights) and restore the original implementation on exit.

    The original is restored by assignment, also when it is ``None``:
    ``PretrainedConfig._attn_implementation`` is a property with no deleter.
    """
    orig_attn = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = orig_attn


# Block Influence

_METRIC_KEYS = ("magnitude_ratio", "contribution_norm", "bi_score")

# Spans of one block's residual stream, as (name, start point, end point):
# "in" is the block input, "mid" the residual after attention (the input to
# post_attention_layernorm), "out" the block output.
_TOTAL_SPAN = (("total", "in", "out"),)
_SUBLAYER_SPANS = (
    ("attention", "in", "mid"),
    ("mlp", "mid", "out"),
    ("total", "in", "out"),
)


def _compute_metrics(flat_in: torch.Tensor, flat_out: torch.Tensor) -> dict[str, float]:
    """Compute magnitude_ratio, contribution_norm, bi_score for a pair of tensors.

    Both inputs should be shaped (num_tokens, hidden_dim) in float. Each
    metric is a per-token value averaged over tokens.
    """
    in_norms = flat_in.norm(dim=-1)
    out_norms = flat_out.norm(dim=-1)
    ratio = out_norms / in_norms.clamp(min=1e-10)

    contrib = (flat_out - flat_in).norm(dim=-1)

    cos_sim = F.cosine_similarity(flat_in, flat_out, dim=-1)

    return {
        "magnitude_ratio": ratio.mean().item(),
        "contribution_norm": contrib.mean().item(),
        "bi_score": max(0.0, min(1.0, 1.0 - cos_sim.mean().item())),
    }


def _layer_influence(
    model, tokenizer, prompts: list[str], spans: tuple[tuple[str, str, str], ...]
) -> dict[int, dict[str, dict[str, float]]]:
    """Average :func:`_compute_metrics` over prompts for each layer and span.

    Runs one forward pass per prompt. Metrics are computed right after each
    pass and only that prompt's hidden states are held, so memory does not
    grow with the number of prompts.
    """
    layers = model.model.layers
    num_layers = len(layers)
    points = sorted({pt for _name, a, b in spans for pt in (a, b)})
    current: dict[tuple[str, int], torch.Tensor] = {}

    def make_block_hook(idx: int):
        def hook(_module, inp, out):
            current[("in", idx)] = inp[0].detach()
            current[("out", idx)] = _unwrap_hook_output(out).detach()
        return hook

    hooks = [layer.register_forward_hook(make_block_hook(i)) for i, layer in enumerate(layers)]
    if "mid" in points:
        hooks += [
            layer.post_attention_layernorm.register_forward_pre_hook(
                _make_capture_input_hook(current, ("mid", i))
            )
            for i, layer in enumerate(layers)
        ]

    sums: dict[int, dict[str, dict[str, float]]] = {
        i: {name: dict.fromkeys(_METRIC_KEYS, 0.0) for name, _a, _b in spans}
        for i in range(num_layers)
    }
    n_prompts = 0
    device = _get_input_device(model)
    try:
        for prompt in prompts:
            current.clear()
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids)
            _raise_if_missing(
                (i for i in range(num_layers) if all((pt, i) in current for pt in points)),
                num_layers,
            )
            for i in range(num_layers):
                flat = {
                    pt: current[(pt, i)].reshape(-1, current[(pt, i)].shape[-1]).float()
                    for pt in points
                }
                for name, a, b in spans:
                    for key, value in _compute_metrics(flat[a], flat[b]).items():
                        sums[i][name][key] += value
            n_prompts += 1
    finally:
        for h in hooks:
            h.remove()
        current.clear()

    denom = n_prompts or 1
    return {
        i: {name: {k: v / denom for k, v in m.items()} for name, m in per_span.items()}
        for i, per_span in sums.items()
    }


def block_influence(model, tokenizer, prompts: list[str]) -> dict[int, float]:
    """Compute Block Influence (BI) score per layer: 1 - cos(input, output).

    Equivalent to the ``bi_score`` field of :func:`magnitude_influence`.
    Returns a dict mapping layer index -> float score in [0, 1].
    """
    return {
        i: m["bi_score"]
        for i, m in magnitude_influence(model, tokenizer, prompts).items()
    }


def magnitude_influence(
    model, tokenizer, prompts: list[str]
) -> dict[int, dict[str, float]]:
    """Compute magnitude-aware influence metrics for each transformer layer.

    Complements block_influence (angle-only) with magnitude information.
    Uses forward hooks to capture layer input/output hidden states.

    Returns a dict mapping layer index -> dict with:
        magnitude_ratio: ||output|| / ||input||, averaged over tokens and prompts.
            >1 means the layer amplifies, <1 means it attenuates.
        contribution_norm: ||output - input||, the L2 size of the layer's
            residual contribution, averaged over tokens and prompts.
        bi_score: 1 - cosine_similarity(input, output), averaged over tokens,
            clamped to [0, 1] per prompt, then averaged over prompts.
    """
    return {
        i: spans["total"]
        for i, spans in _layer_influence(model, tokenizer, prompts, _TOTAL_SPAN).items()
    }


def sublayer_influence(
    model, tokenizer, prompts: list[str]
) -> dict[int, dict[str, dict[str, float]]]:
    """Decompose per-layer influence into attention and MLP contributions.

    Each LLaMA block does:
        h_mid = h_in + attention(RMSNorm(h_in))
        h_out = h_mid + mlp(RMSNorm(h_mid))

    Hooks capture h_in (block input), h_mid (between attention and MLP,
    via pre-hook on post_attention_layernorm), and h_out (block output).

    Returns a dict mapping layer index -> dict with:
        attention: {magnitude_ratio, contribution_norm, bi_score} for h_in -> h_mid
        mlp:       {magnitude_ratio, contribution_norm, bi_score} for h_mid -> h_out
        total:     {magnitude_ratio, contribution_norm, bi_score} for h_in -> h_out
    """
    return _layer_influence(model, tokenizer, prompts, _SUBLAYER_SPANS)


# Weight norms and SVD

def weight_norms(model) -> list[dict]:
    """Compute Frobenius norms of attention and MLP parameter groups per layer.

    Returns a list of dicts:
        [{"layer": int, "attn_norm": float, "mlp_norm": float, "total_norm": float}, ...]
    """
    results = []
    for i, layer in enumerate(model.model.layers):
        attn_tensors = []
        mlp_tensors = []

        # Collect attention weights
        for _name, param in layer.self_attn.named_parameters():
            attn_tensors.append(param.detach().float())

        # Collect MLP weights
        for _name, param in layer.mlp.named_parameters():
            mlp_tensors.append(param.detach().float())

        def _combined_frob(tensors):
            if not tensors:
                return 0.0
            # Stack flattened tensors and compute overall Frobenius norm
            flat = torch.cat([t.flatten() for t in tensors])
            return flat.norm().item()

        attn_n = _combined_frob(attn_tensors)
        mlp_n = _combined_frob(mlp_tensors)
        total_n = _combined_frob(attn_tensors + mlp_tensors)

        results.append({
            "layer": i,
            "attn_norm": attn_n,
            "mlp_norm": mlp_n,
            "total_norm": total_n,
        })

    return results


def weight_svd(model, layers: list[int] | None = None) -> dict[int, dict]:
    """Compute singular values of key weight matrices for specified layers.

    Args:
        model: LlamaForCausalLM instance.
        layers: List of layer indices, or None to process all layers.

    Returns:
        Dict mapping layer index -> dict of {proj_name: singular_values_tensor}.
        Matrices inspected: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
    """
    num_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(num_layers))

    result: dict[int, dict] = {}
    for i in layers:
        layer = model.model.layers[i]
        layer_svd = {}

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.self_attn, proj_name, None)
            if proj is not None:
                w = proj.weight.detach().float()
                layer_svd[proj_name] = torch.linalg.svdvals(w)

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(layer.mlp, proj_name, None)
            if proj is not None:
                w = proj.weight.detach().float()
                layer_svd[proj_name] = torch.linalg.svdvals(w)

        result[i] = layer_svd

    return result


# Activation analysis

def attention_entropy(model, tokenizer, prompt: str) -> dict[int, list[float]]:
    """Compute entropy of attention distributions per head per layer.

    Uses model(input_ids, output_attentions=True) to obtain attention weights.
    Entropy per head = -sum(p * log(p + eps)), averaged over query positions.

    Returns:
        Dict mapping layer index -> list of per-head entropy floats.
    """
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(_get_input_device(model))

    with _eager_attention(model), torch.no_grad():
        out = model(input_ids, output_attentions=True)

    # out.attentions: tuple of (batch, heads, seq_q, seq_k) per layer
    eps = 1e-10
    result: dict[int, list[float]] = {}
    for layer_idx, attn_weights in enumerate(out.attentions):
        # attn_weights: (batch, num_heads, seq_q, seq_k)
        # Work with first (only) batch element
        aw = attn_weights[0].float()  # (num_heads, seq_q, seq_k)
        num_heads = aw.shape[0]
        head_entropies = []
        for h in range(num_heads):
            # entropy per query position, then averaged
            p = aw[h]  # (seq_q, seq_k)
            ent = -(p * torch.log(p + eps)).sum(dim=-1)  # (seq_q,)
            head_entropies.append(ent.mean().item())
        result[layer_idx] = head_entropies

    return result


def residual_stream_norms(model, tokenizer, prompt: str) -> list[float]:
    """Compute L2 norm of the residual stream at each stage of the model.

    Captures:
        - Output of embed_tokens (position 0)
        - Output of each transformer layer (positions 1..num_layers)

    Returns a list of length num_layers + 1. Raises ``RuntimeError`` naming
    the layers whose hooks did not fire.
    """
    num_layers = len(model.model.layers)
    # Key 0 is the embedding output, key i + 1 the output of layer i.
    activations: dict[int, torch.Tensor] = {}
    hooks = [
        model.model.embed_tokens.register_forward_hook(
            _make_capture_output_hook(activations, 0)
        )
    ]
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(_make_capture_output_hook(activations, i + 1)))

    try:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(_get_input_device(model))
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    if 0 not in activations:
        raise RuntimeError("Forward hook on embed_tokens did not fire")
    _raise_if_missing((k - 1 for k in activations if k > 0), num_layers)

    # Mean norm across tokens and batch
    return [
        activations[k].float().norm(dim=-1).mean().item()
        for k in range(num_layers + 1)
    ]


# Individual head inspection

def inspect_head(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    head: int,
) -> dict[str, Any]:
    """Inspect a single attention head: its attention pattern, output norm, and entropy.

    Returns dict with:
        attention_pattern: (seq_len, seq_len) tensor of attention weights
        output_norm: mean L2 norm of this head's output across tokens
        entropy: mean entropy of this head's attention distribution
    """
    num_heads = model.config.num_attention_heads
    if head < 0 or head >= num_heads:
        raise IndexError(f"Head {head} out of range [0, {num_heads - 1}]")
    num_layers = len(model.model.layers)
    if layer < 0 or layer >= num_layers:
        raise IndexError(f"Layer {layer} out of range [0, {num_layers - 1}]")

    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    o_proj = model.model.layers[layer].self_attn.o_proj
    # The o_proj input is num_heads * head_dim wide. head_dim can differ from
    # hidden_size // num_heads (config.head_dim), so derive it from o_proj.
    head_dim = o_proj.in_features // num_heads

    # Use a pre-hook on o_proj to capture head outputs before mixing;
    # its input has shape (batch, seq, num_heads * head_dim).
    o_proj_input: dict[str, torch.Tensor] = {}

    # Need attention weights — force eager attention
    with _eager_attention(model):
        hook = o_proj.register_forward_pre_hook(
            _make_capture_input_hook(o_proj_input, "val")
        )
        try:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True)
        finally:
            hook.remove()

    # Extract attention pattern for this head at this layer
    # outputs.attentions is a tuple: one (batch, num_heads, seq, seq) per layer
    attn_weights = outputs.attentions[layer][0, head].float()  # (seq, seq)

    # Extract this head's output (before o_proj mixing)
    # o_proj_input shape: (batch, seq, num_heads * head_dim)
    full_output = o_proj_input["val"][0].float()  # (seq, num_heads * head_dim)
    head_slice = full_output[:, head * head_dim : (head + 1) * head_dim]  # (seq, head_dim)
    output_norm = head_slice.norm(dim=-1).mean().item()

    # Entropy of attention distribution
    eps = 1e-10
    ent_per_pos = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (seq,)
    entropy = ent_per_pos.mean().item()

    return {
        "attention_pattern": attn_weights,
        "output_norm": output_norm,
        "entropy": entropy,
    }
