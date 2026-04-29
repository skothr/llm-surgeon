"""Inspection and activation analysis tools for LLaMA models."""

from typing import Any

import torch
import torch.nn.functional as F


def _get_input_device(model) -> torch.device:
    """Get the device where input_ids should be sent (embedding layer's device)."""
    return model.get_input_embeddings().weight.device


def _capture_layer_io(
    model, tokenizer, prompts: list[str]
) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
    """Run a forward pass per prompt and capture each layer's input + output.

    Returns ``(layer_inputs, layer_outputs)``: dicts mapping layer index
    to a list of detached tensors of shape ``(batch, seq, hidden)``,
    one entry per prompt.
    """
    num_layers = len(model.model.layers)
    layer_inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    layer_outputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    def make_hook(idx: int):
        def hook(_module, inp, out):
            layer_inputs[idx].append(inp[0].detach())
            hidden_out = out[0].detach() if isinstance(out, tuple) else out.detach()
            layer_outputs[idx].append(hidden_out)
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    device = _get_input_device(model)
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return layer_inputs, layer_outputs


# Block Influence

def block_influence(model, tokenizer, prompts: list[str]) -> dict[int, float]:
    """Compute Block Influence (BI) score for each transformer layer.

    BI = 1 - cosine_similarity(layer_input, layer_output), averaged over
    all token positions and all prompts.

    Returns a dict mapping layer index -> float score in [0, 1].
    """
    layer_inputs, layer_outputs = _capture_layer_io(model, tokenizer, prompts)
    num_layers = len(model.model.layers)

    scores: dict[int, float] = {}
    for i in range(num_layers):
        sims = []
        for h_in, h_out in zip(layer_inputs[i], layer_outputs[i]):
            # h_in, h_out: (batch, seq, hidden)
            # flatten to (batch*seq, hidden)
            flat_in = h_in.reshape(-1, h_in.shape[-1]).float()
            flat_out = h_out.reshape(-1, h_out.shape[-1]).float()
            # cosine similarity per token
            cos_sim = F.cosine_similarity(flat_in, flat_out, dim=-1)  # (batch*seq,)
            sims.append(cos_sim.mean().item())
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        scores[i] = 1.0 - avg_sim
        # clamp to [0, 1] for numerical safety
        scores[i] = max(0.0, min(1.0, scores[i]))

    return scores


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
        bi_score: 1 - cosine_similarity(input, output), same as block_influence.
    """
    layer_inputs, layer_outputs = _capture_layer_io(model, tokenizer, prompts)
    num_layers = len(model.model.layers)

    results: dict[int, dict[str, float]] = {}
    for i in range(num_layers):
        mag_ratios = []
        contrib_norms = []
        cosine_sims = []

        for h_in, h_out in zip(layer_inputs[i], layer_outputs[i]):
            flat_in = h_in.reshape(-1, h_in.shape[-1]).float()
            flat_out = h_out.reshape(-1, h_out.shape[-1]).float()

            # Per-token magnitude ratio: ||out|| / ||in||
            in_norms = flat_in.norm(dim=-1)
            out_norms = flat_out.norm(dim=-1)
            # Avoid division by zero
            ratio = out_norms / in_norms.clamp(min=1e-10)
            mag_ratios.append(ratio.mean().item())

            # Per-token contribution norm: ||out - in||
            contrib = (flat_out - flat_in).norm(dim=-1)
            contrib_norms.append(contrib.mean().item())

            # Cosine similarity (same as block_influence)
            cos_sim = F.cosine_similarity(flat_in, flat_out, dim=-1)
            cosine_sims.append(cos_sim.mean().item())

        n = len(mag_ratios) or 1
        avg_ratio = sum(mag_ratios) / n
        avg_contrib = sum(contrib_norms) / n
        avg_sim = sum(cosine_sims) / n

        results[i] = {
            "magnitude_ratio": avg_ratio,
            "contribution_norm": avg_contrib,
            "bi_score": max(0.0, min(1.0, 1.0 - avg_sim)),
        }

    return results


def _compute_metrics(flat_in: torch.Tensor, flat_out: torch.Tensor) -> dict[str, float]:
    """Compute magnitude_ratio, contribution_norm, bi_score for a pair of tensors.

    Both inputs should be shaped (num_tokens, hidden_dim) in float.
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
    num_layers = len(model.model.layers)
    layer_h_in: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    layer_h_mid: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    layer_h_out: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    hooks = []

    def make_block_hook(idx):
        def hook(module, inp, out):
            layer_h_in[idx].append(inp[0].detach())
            hidden_out = out[0].detach() if isinstance(out, tuple) else out.detach()
            layer_h_out[idx].append(hidden_out)
        return hook

    def make_mid_hook(idx):
        def hook(module, args):
            # pre-hook on post_attention_layernorm: input is h_mid
            layer_h_mid[idx].append(args[0].detach())
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_block_hook(i)))
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(make_mid_hook(i))
        )

    device = _get_input_device(model)
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    results: dict[int, dict[str, dict[str, float]]] = {}
    for i in range(num_layers):
        attn_metrics_list = []
        mlp_metrics_list = []
        total_metrics_list = []

        for h_in, h_mid, h_out in zip(
            layer_h_in[i], layer_h_mid[i], layer_h_out[i]
        ):
            flat_in = h_in.reshape(-1, h_in.shape[-1]).float()
            flat_mid = h_mid.reshape(-1, h_mid.shape[-1]).float()
            flat_out = h_out.reshape(-1, h_out.shape[-1]).float()

            attn_metrics_list.append(_compute_metrics(flat_in, flat_mid))
            mlp_metrics_list.append(_compute_metrics(flat_mid, flat_out))
            total_metrics_list.append(_compute_metrics(flat_in, flat_out))

        def _avg(metrics_list):
            n = len(metrics_list) or 1
            return {
                key: sum(m[key] for m in metrics_list) / n
                for key in ("magnitude_ratio", "contribution_norm", "bi_score")
            }

        results[i] = {
            "attention": _avg(attn_metrics_list),
            "mlp": _avg(mlp_metrics_list),
            "total": _avg(total_metrics_list),
        }

    return results


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

    # sdpa does not support output_attentions; switch to eager temporarily
    orig_attn = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
    finally:
        if orig_attn is not None:
            model.config._attn_implementation = orig_attn
        else:
            del model.config._attn_implementation

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

    Returns a list of length num_layers + 1.
    """
    num_layers = len(model.model.layers)
    activations: list[torch.Tensor | None] = [None] * (num_layers + 1)
    hooks = []

    def embed_hook(module, inp, out):
        activations[0] = out.detach()

    def make_layer_hook(idx):
        def hook(module, inp, out):
            hidden = out[0].detach() if isinstance(out, tuple) else out.detach()
            activations[idx + 1] = hidden
        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))

    try:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(_get_input_device(model))
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    norms = []
    for act in activations:
        if act is None:
            norms.append(0.0)
        else:
            # Mean norm across tokens and batch
            norms.append(act.float().norm(dim=-1).mean().item())

    return norms


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

    # Need attention weights — force eager attention
    orig_attn = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"

    head_dim = model.config.hidden_size // num_heads

    # Use a pre-hook on o_proj to capture head outputs before mixing
    o_proj_input = {}
    def _o_proj_pre_hook(module, args):
        # args[0] shape: (batch, seq, num_heads * head_dim)
        o_proj_input["val"] = args[0].detach()

    hook = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(_o_proj_pre_hook)

    try:
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
    finally:
        hook.remove()
        if orig_attn is not None:
            model.config._attn_implementation = orig_attn
        else:
            try:
                del model.config._attn_implementation
            except AttributeError:
                pass

    # Extract attention pattern for this head at this layer
    # outputs.attentions is a tuple: one (batch, num_heads, seq, seq) per layer
    attn_weights = outputs.attentions[layer][0, head].float()  # (seq, seq)

    # Extract this head's output (before o_proj mixing)
    # o_proj_input shape: (batch, seq, num_heads * head_dim)
    if "val" in o_proj_input:
        full_output = o_proj_input["val"][0].float()  # (seq, num_heads * head_dim)
        head_slice = full_output[:, head * head_dim : (head + 1) * head_dim]  # (seq, head_dim)
        output_norm = head_slice.norm(dim=-1).mean().item()
    else:
        output_norm = 0.0

    # Entropy of attention distribution
    eps = 1e-10
    ent_per_pos = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (seq,)
    entropy = ent_per_pos.mean().item()

    return {
        "attention_pattern": attn_weights,
        "output_norm": output_norm,
        "entropy": entropy,
    }
