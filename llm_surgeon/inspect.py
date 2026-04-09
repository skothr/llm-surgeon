"""Inspection and activation analysis tools for LLaMA models."""

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _get_input_device(model) -> torch.device:
    """Get the device where input_ids should be sent (embedding layer's device)."""
    return model.model.embed_tokens.weight.device


# ---------------------------------------------------------------------------
# Task 1: Block Influence
# ---------------------------------------------------------------------------

def block_influence(model, tokenizer, prompts: List[str]) -> Dict[int, float]:
    """Compute Block Influence (BI) score for each transformer layer.

    BI = 1 - cosine_similarity(layer_input, layer_output), averaged over
    all token positions and all prompts.

    Returns a dict mapping layer index -> float score in [0, 1].
    """
    num_layers = len(model.model.layers)
    layer_inputs: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
    layer_outputs: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}

    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            # inp is a tuple; first element is the hidden states
            hidden_in = inp[0].detach()  # (batch, seq, hidden)
            # out may be a tuple or a tensor
            if isinstance(out, tuple):
                hidden_out = out[0].detach()
            else:
                hidden_out = out.detach()
            layer_inputs[idx].append(hidden_in)
            layer_outputs[idx].append(hidden_out)
        return hook

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

    scores: Dict[int, float] = {}
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


# ---------------------------------------------------------------------------
# Task 2: Weight norms and SVD
# ---------------------------------------------------------------------------

def weight_norms(model) -> List[dict]:
    """Compute Frobenius norms of attention and MLP parameter groups per layer.

    Returns a list of dicts:
        [{"layer": int, "attn_norm": float, "mlp_norm": float, "total_norm": float}, ...]
    """
    results = []
    for i, layer in enumerate(model.model.layers):
        attn_tensors = []
        mlp_tensors = []

        # Collect attention weights
        for name, param in layer.self_attn.named_parameters():
            attn_tensors.append(param.detach().float())

        # Collect MLP weights
        for name, param in layer.mlp.named_parameters():
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


def weight_svd(model, layers: Optional[List[int]] = None) -> Dict[int, dict]:
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

    result: Dict[int, dict] = {}
    for i in layers:
        layer = model.model.layers[i]
        layer_svd = {}

        attn_projs = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp_projs = {"gate_proj", "up_proj", "down_proj"}

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


# ---------------------------------------------------------------------------
# Task 3: Activation analysis
# ---------------------------------------------------------------------------

def attention_entropy(model, tokenizer, prompt: str) -> Dict[int, List[float]]:
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
    result: Dict[int, List[float]] = {}
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


def residual_stream_norms(model, tokenizer, prompt: str) -> List[float]:
    """Compute L2 norm of the residual stream at each stage of the model.

    Captures:
        - Output of embed_tokens (position 0)
        - Output of each transformer layer (positions 1..num_layers)

    Returns a list of length num_layers + 1.
    """
    num_layers = len(model.model.layers)
    activations: List[Optional[torch.Tensor]] = [None] * (num_layers + 1)
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
