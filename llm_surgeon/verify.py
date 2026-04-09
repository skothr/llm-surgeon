"""Structural verification of modified models."""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class VerifyReport:
    """Result of structural verification checks."""
    passed: bool = True
    checks: List[dict] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"VerifyReport: {status}"]
        for check in self.checks:
            mark = "[pass]" if check["passed"] else "[FAIL]"
            lines.append(f"  {mark} {check['name']}: {check['detail']}")
        return "\n".join(lines)


def check_structure(model, surgery_log=None) -> VerifyReport:
    """Validate model structural integrity after surgery.
    Raises ValueError if any critical check fails.
    """
    report = VerifyReport()

    actual_layers = len(model.model.layers)
    config_layers = model.config.num_hidden_layers
    report.add_check(
        "layer_count_matches_config",
        actual_layers == config_layers,
        f"actual={actual_layers}, config={config_layers}",
    )

    embed_dim = model.model.embed_tokens.embedding_dim
    hidden_size = model.config.hidden_size
    report.add_check(
        "embedding_dim_consistent",
        embed_dim == hidden_size,
        f"embed_dim={embed_dim}, hidden_size={hidden_size}",
    )

    lm_head_out = model.lm_head.out_features
    vocab_size = model.config.vocab_size
    report.add_check(
        "lm_head_vocab_consistent",
        lm_head_out == vocab_size,
        f"lm_head_out={lm_head_out}, vocab_size={vocab_size}",
    )

    lm_head_in = model.lm_head.in_features
    report.add_check(
        "lm_head_hidden_consistent",
        lm_head_in == hidden_size,
        f"lm_head_in={lm_head_in}, hidden_size={hidden_size}",
    )

    if surgery_log is not None:
        for op in surgery_log.ops:
            report.add_check(
                f"surgery_log_{op.operation}",
                actual_layers == op.layer_count_after,
                f"expected={op.layer_count_after} after {op.operation}, actual={actual_layers}",
            )

    if not report.passed:
        raise ValueError(f"Structural verification failed:\n{report}")

    return report


# ---------------------------------------------------------------------------
# Task 4: Activation capture, comparison, and caching
# ---------------------------------------------------------------------------

def _capture_layer_activations(model, tokenizer, prompt: str) -> List[torch.Tensor]:
    """Capture the output tensor of each transformer layer for the given prompt.

    Returns a list of tensors, one per layer, each of shape (batch, seq, hidden).
    """
    num_layers = len(model.model.layers)
    activations: List[Optional[torch.Tensor]] = [None] * num_layers
    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            hidden = out[0].detach() if isinstance(out, tuple) else out.detach()
            activations[idx] = hidden
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return [a for a in activations if a is not None]


def _compare_activation_lists(
    acts_a: List[torch.Tensor],
    acts_b: List[torch.Tensor],
) -> List[dict]:
    """Layer-by-layer comparison up to the shorter model's depth."""
    depth = min(len(acts_a), len(acts_b))
    results = []
    for i in range(depth):
        a = acts_a[i].float().reshape(-1, acts_a[i].shape[-1])  # (tokens, hidden)
        b = acts_b[i].float().reshape(-1, acts_b[i].shape[-1])

        cos_sim = F.cosine_similarity(a, b, dim=-1).mean().item()
        diff = (a - b)
        l2_dist = diff.norm().item()
        max_abs_diff = diff.abs().max().item()

        results.append({
            "layer": i,
            "cosine_sim": cos_sim,
            "l2_dist": l2_dist,
            "max_abs_diff": max_abs_diff,
        })
    return results


def compare_activations(original, modified, tokenizer, prompt: str) -> List[dict]:
    """Compare layer activations between two models for the same prompt.

    Compares layer-by-layer up to the depth of the shallower model.

    Returns a list of dicts per layer:
        [{"layer": int, "cosine_sim": float, "l2_dist": float, "max_abs_diff": float}, ...]
    """
    acts_orig = _capture_layer_activations(original, tokenizer, prompt)
    acts_mod = _capture_layer_activations(modified, tokenizer, prompt)
    return _compare_activation_lists(acts_orig, acts_mod)


def _prompt_cache_path(cache_dir: str, prompt: str) -> str:
    """Return the .pt file path for a given prompt, keyed by sha256 hash."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{h}.pt")


def cache_baseline(model, tokenizer, prompts: List[str], cache_dir: str) -> None:
    """Capture and save activations for each prompt to disk as .pt files.

    Each file is named by the sha256 hash of the prompt text and contains
    a list of layer activation tensors.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for prompt in prompts:
        acts = _capture_layer_activations(model, tokenizer, prompt)
        path = _prompt_cache_path(cache_dir, prompt)
        torch.save(acts, path)


def compare_to_baseline(
    model,
    tokenizer,
    prompts: List[str],
    cache_dir: str,
) -> Dict[str, List[dict]]:
    """Load cached activations and compare against the current model.

    Returns a dict mapping prompt text -> list of per-layer comparison dicts.
    """
    results: Dict[str, List[dict]] = {}
    for prompt in prompts:
        path = _prompt_cache_path(cache_dir, prompt)
        cached_acts = torch.load(path)
        current_acts = _capture_layer_activations(model, tokenizer, prompt)
        results[prompt] = _compare_activation_lists(cached_acts, current_acts)
    return results
