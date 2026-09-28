"""Structural verification of modified models."""

import hashlib
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from llm_surgeon.inspect import _get_input_device
from llm_surgeon.probe._hooks import _make_capture_output_hook


@dataclass
class VerifyReport:
    """Result of structural verification checks."""

    passed: bool = True
    checks: list[dict] = field(default_factory=list)

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

    With ``surgery_log``, also checks that its ops chain (each op's
    ``layer_count_before`` equals the previous op's ``layer_count_after``)
    and that the last op's ``layer_count_after`` equals the model's layer
    count. The log may hold several ops, as the combined log of a recipe does.

    Returns the populated report on success; raises ``ValueError`` (with the
    report rendered into the message) if any check fails — callers therefore
    only ever observe a passing report.
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

    if surgery_log is not None and surgery_log.ops:
        ops = surgery_log.ops
        for prev, op in zip(ops, ops[1:]):
            report.add_check(
                "surgery_log_chain",
                op.layer_count_before == prev.layer_count_after,
                f"{prev.operation} left {prev.layer_count_after} layers, "
                f"next {op.operation} started from {op.layer_count_before}",
            )
        last = ops[-1]
        report.add_check(
            f"surgery_log_{last.operation}",
            actual_layers == last.layer_count_after,
            f"expected={last.layer_count_after} after {last.operation}, actual={actual_layers}",
        )

    if not report.passed:
        raise ValueError(f"Structural verification failed:\n{report}")

    return report


# Activation capture, comparison, and caching


def _encode(model, tokenizer, prompt: str) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt")
    return enc["input_ids"].to(_get_input_device(model))


def _capture_layer_activations(model, tokenizer, prompt: str) -> list[torch.Tensor]:
    """Capture the output tensor of each transformer layer for the given prompt.

    Returns a list of tensors, one per layer, each of shape (batch, seq, hidden).
    Raises ``RuntimeError`` naming the layers whose hooks did not fire.
    """
    num_layers = len(model.model.layers)
    captured: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(_make_capture_output_hook(captured, i))
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        input_ids = _encode(model, tokenizer, prompt)
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    missing = [i for i in range(num_layers) if i not in captured]
    if missing:
        raise RuntimeError(
            f"Forward hooks did not fire for layer(s) {missing} of {num_layers}; "
            f"the model's forward skipped them"
        )
    return [captured[i].detach() for i in range(num_layers)]


def _compare_activation_lists(
    acts_a: list[torch.Tensor],
    acts_b: list[torch.Tensor],
    layer_map: Mapping[int, int] | None = None,
) -> list[dict]:
    """Compare ``acts_b`` (modified) with ``acts_a`` (original), layer by layer.

    Without ``layer_map``, layer ``i`` is compared with layer ``i`` up to the
    shorter list's length. With it, each modified index ``k`` is compared
    with original index ``layer_map[k]``. Tensors are moved to CPU, so the
    two sides may come from different devices.
    """
    if layer_map is None:
        pairs = [(i, i) for i in range(min(len(acts_a), len(acts_b)))]
    else:
        pairs = sorted(layer_map.items())
        for mod_idx, orig_idx in pairs:
            if not (0 <= mod_idx < len(acts_b) and 0 <= orig_idx < len(acts_a)):
                raise IndexError(
                    f"layer_map entry {mod_idx} -> {orig_idx} out of range: "
                    f"modified has {len(acts_b)} layers, original has {len(acts_a)}"
                )

    results = []
    for mod_idx, orig_idx in pairs:
        a = acts_a[orig_idx].detach().float().cpu()
        b = acts_b[mod_idx].detach().float().cpu()
        a = a.reshape(-1, a.shape[-1])  # (tokens, hidden)
        b = b.reshape(-1, b.shape[-1])

        diff = a - b
        results.append(
            {
                "layer": mod_idx,
                "original_layer": orig_idx,
                "cosine_sim": F.cosine_similarity(a, b, dim=-1).mean().item(),
                "l2_dist": diff.norm(dim=-1).mean().item(),
                "max_abs_diff": diff.abs().max().item(),
            }
        )
    return results


def compare_activations(
    original,
    modified,
    tokenizer,
    prompt: str,
    layer_map: Mapping[int, int] | None = None,
) -> list[dict]:
    """Compare layer activations between two models for the same prompt.

    By default layers are aligned by position (layer ``i`` of ``modified``
    against layer ``i`` of ``original``), up to the depth of the shallower
    model. After a structural change such as ``remove_layers([3])`` this
    compares modified layer 3 (originally layer 4) with original layer 3,
    so the numbers measure the index shift rather than the surgery's effect.
    Pass ``layer_map`` (modified layer index -> original layer index) to
    align layers by identity instead; only the mapped layers are compared.

    Returns a list of dicts per compared layer:
        [{"layer": int, "original_layer": int, "cosine_sim": float,
          "l2_dist": float, "max_abs_diff": float}, ...]
    ``cosine_sim`` and ``l2_dist`` are per-token values averaged over tokens;
    ``max_abs_diff`` is the maximum over all elements.
    """
    acts_orig = _capture_layer_activations(original, tokenizer, prompt)
    acts_mod = _capture_layer_activations(modified, tokenizer, prompt)
    return _compare_activation_lists(acts_orig, acts_mod, layer_map)


def _prompt_cache_path(cache_dir: str, prompt: str) -> str:
    """Return the .pt file path for a given prompt, keyed by sha256 hash."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{h}.pt")


def _model_identity(model) -> dict:
    """Fields of a baseline that a later comparison must match.

    Layer count is left out on purpose: the usual comparison is against a
    model after layer surgery.
    """
    return {
        "name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
    }


def cache_baseline(model, tokenizer, prompts: list[str], cache_dir: str) -> None:
    """Capture and save activations for each prompt to disk as .pt files.

    Each file is named by the sha256 hash of the prompt text and contains a
    dict with the layer activations (on CPU), the prompt's token ids and the
    model's name/hidden/vocab size, which :func:`compare_to_baseline` checks.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for prompt in prompts:
        acts = _capture_layer_activations(model, tokenizer, prompt)
        payload = {
            **_model_identity(model),
            "input_ids": _encode(model, tokenizer, prompt).cpu(),
            "activations": [a.cpu() for a in acts],
        }
        torch.save(payload, _prompt_cache_path(cache_dir, prompt))


def _check_baseline(payload: dict, model, tokenizer, prompt: str, path: str) -> None:
    current = _model_identity(model)
    for key in ("hidden_size", "vocab_size"):
        if payload.get(key) != current[key]:
            raise ValueError(
                f"Baseline {path} was cached from a model with {key}="
                f"{payload.get(key)}, but the current model has {key}={current[key]}"
            )
    cached_ids = payload.get("input_ids")
    current_ids = _encode(model, tokenizer, prompt).cpu()
    if cached_ids is not None and not torch.equal(cached_ids, current_ids):
        raise ValueError(
            f"Baseline {path} for prompt {prompt!r} has different token ids than "
            f"the current tokenizer produces; it was cached with another tokenizer"
        )
    if payload.get("name_or_path") != current["name_or_path"]:
        warnings.warn(
            f"Baseline {path} was cached from {payload.get('name_or_path')!r}; "
            f"comparing against {current['name_or_path']!r}",
            stacklevel=3,
        )


def compare_to_baseline(
    model,
    tokenizer,
    prompts: list[str],
    cache_dir: str,
    layer_map: Mapping[int, int] | None = None,
) -> dict[str, list[dict]]:
    """Load cached activations and compare against the current model.

    The cached baseline plays the ``original`` role of
    :func:`compare_activations`; ``layer_map`` has the same meaning there.
    Raises ``ValueError`` if the baseline came from a model with a different
    hidden or vocab size, or its prompt tokenized differently.

    Returns a dict mapping prompt text -> list of per-layer comparison dicts.
    """
    results: dict[str, list[dict]] = {}
    for prompt in prompts:
        path = _prompt_cache_path(cache_dir, prompt)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached baseline for prompt {prompt!r} at {path}. "
                f"Run cache_baseline() with this prompt first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, list):
            # Cache written before metadata was stored: activations only.
            cached_acts = payload
        else:
            _check_baseline(payload, model, tokenizer, prompt, path)
            cached_acts = payload["activations"]
        current_acts = _capture_layer_activations(model, tokenizer, prompt)
        results[prompt] = _compare_activation_lists(
            cached_acts, current_acts, layer_map
        )
    return results
