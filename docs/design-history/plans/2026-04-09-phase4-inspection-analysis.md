# Phase 4: Inspection + Activation Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.
>
> **Tool rules (for subagents):**
> - Use Read (not cat/head/tail), Grep (not grep/rg/awk), Glob (not find/ls), Edit (not sed/awk) for all file operations
> - You are already in the project root (/home/ai/ai-projects/llm) — never cd
> - Python venv: `/home/ai/ai-projects/llm/testing/.venv/bin/python`
> - Run tests: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`

**Goal:** Build `inspect.py` (Block Influence scores, weight norms, SVD, attention entropy, residual stream norms) and add activation comparison + cached baselines to `verify.py`. These tools help decide *which* layers to cut and measure *where* outputs diverge after surgery.

**Architecture:** `inspect.py` provides read-only analysis functions that use PyTorch forward hooks to capture activations. `verify.py` gets Level 2 activation comparison and disk-cached baselines. All functions work on 4-bit (inspect mode) models.

**Tech Stack:** PyTorch (hooks, SVD, cosine similarity), numpy

**Reference:** `docs/superpowers/specs/2026-04-08-llm-surgeon-design.md` (v2), Phase 4 section of phase plan.

---

## File Map

```
testing/
  llm_surgeon/
    inspect.py           — CREATE — block_influence, weight_norms, weight_svd, attention_entropy, residual_stream_norms
    verify.py            — MODIFY — add compare_activations, cache_baseline, compare_to_baseline
  tests/
    test_inspect.py      — CREATE — tests for inspect module
    test_verify.py       — MODIFY — add tests for activation comparison + cached baselines
```

---

### Task 1: Block Influence scores

**Files:**
- Create: `testing/llm_surgeon/inspect.py`
- Create: `testing/tests/test_inspect.py`
- Modify: `testing/llm_surgeon/__init__.py`

- [ ] **Step 1: Write tests for block_influence**

Create `testing/tests/test_inspect.py`:

```python
"""Tests for inspect module."""

import pytest
import torch
from tests.conftest import _make_tiny_tokenizer
from llm_surgeon.inspect import block_influence


class TestBlockInfluence:
    def test_returns_dict_with_all_layers(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        bi = block_influence(tiny_llama, tokenizer, prompts=["tok3 tok4 tok5 tok6 tok7"])
        assert isinstance(bi, dict)
        assert len(bi) == 8  # tiny_llama has 8 layers

    def test_scores_between_zero_and_one(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        bi = block_influence(tiny_llama, tokenizer, prompts=["tok3 tok4 tok5 tok6 tok7"])
        for layer_idx, score in bi.items():
            assert 0.0 <= score <= 1.0, f"Layer {layer_idx}: BI={score} out of range"

    def test_keys_are_layer_indices(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        bi = block_influence(tiny_llama, tokenizer, prompts=["tok3 tok4 tok5 tok6 tok7"])
        assert set(bi.keys()) == set(range(8))

    def test_multiple_prompts_averages(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        bi_single = block_influence(tiny_llama, tokenizer, prompts=["tok3 tok4 tok5"])
        bi_multi = block_influence(tiny_llama, tokenizer, prompts=["tok3 tok4 tok5", "tok6 tok7 tok8"])
        # Both should return valid scores (values may differ due to averaging)
        assert len(bi_single) == len(bi_multi) == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_inspect.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement block_influence**

Create `testing/llm_surgeon/inspect.py`:

```python
"""Model inspection: layer analysis tools for surgery planning."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def block_influence(
    model,
    tokenizer,
    prompts: List[str],
) -> Dict[int, float]:
    """Compute Block Influence (BI) scores for each layer.

    BI = 1 - cosine_similarity(layer_input, layer_output), averaged across
    tokens and prompts. Higher BI = more transformative layer. Lower BI =
    layer is closer to identity (safer to remove).

    From: ShortGPT (Men et al., 2024)
    """
    num_layers = len(model.model.layers)
    # Accumulate cosine sims per layer across all prompts
    layer_cosine_sums = {i: 0.0 for i in range(num_layers)}
    total_counts = {i: 0 for i in range(num_layers)}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)

        # Capture layer inputs and outputs via hooks
        layer_inputs = {}
        layer_outputs = {}

        hooks = []
        for i, layer in enumerate(model.model.layers):
            def make_hook(idx):
                def hook_fn(module, inp, out):
                    # inp is a tuple; first element is the hidden state
                    layer_inputs[idx] = inp[0].detach()
                    # out is a tuple for LlamaDecoderLayer; first element is hidden state
                    if isinstance(out, tuple):
                        layer_outputs[idx] = out[0].detach()
                    else:
                        layer_outputs[idx] = out.detach()
                return hook_fn
            hooks.append(layer.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            model(input_ids)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute cosine similarity per layer
        for i in range(num_layers):
            inp = layer_inputs[i].float()  # (batch, seq, hidden)
            out = layer_outputs[i].float()
            # Cosine sim per token, then average
            cos_sim = F.cosine_similarity(inp, out, dim=-1)  # (batch, seq)
            layer_cosine_sums[i] += cos_sim.mean().item()
            total_counts[i] += 1

    # BI = 1 - avg_cosine_similarity
    bi_scores = {}
    for i in range(num_layers):
        avg_cos = layer_cosine_sums[i] / max(total_counts[i], 1)
        bi_scores[i] = 1.0 - avg_cos

    return bi_scores
```

- [ ] **Step 4: Update __init__.py**

```python
"""LLM Surgeon — surgical layer-level manipulation of LLaMA models."""

from llm_surgeon import surgery, verify, export, benchmark, inspect
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_inspect.py::TestBlockInfluence -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/inspect.py testing/tests/test_inspect.py testing/llm_surgeon/__init__.py
git commit -m "feat: add block_influence scores for layer importance analysis"
```

---

### Task 2: Weight analysis (weight_norms, weight_svd)

**Files:**
- Modify: `testing/llm_surgeon/inspect.py`
- Modify: `testing/tests/test_inspect.py`

- [ ] **Step 1: Write tests**

Add to `testing/tests/test_inspect.py`:

```python
from llm_surgeon.inspect import weight_norms, weight_svd


class TestWeightNorms:
    def test_returns_dict_per_layer(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        assert isinstance(norms, list)
        assert len(norms) == 8

    def test_each_entry_has_expected_keys(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        for entry in norms:
            assert "layer" in entry
            assert "attn_norm" in entry
            assert "mlp_norm" in entry
            assert "total_norm" in entry

    def test_norms_are_positive(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        for entry in norms:
            assert entry["attn_norm"] > 0
            assert entry["mlp_norm"] > 0
            assert entry["total_norm"] > 0


class TestWeightSvd:
    def test_returns_dict_per_layer(self, tiny_llama):
        svd = weight_svd(tiny_llama, layers=[0, 3, 7])
        assert isinstance(svd, dict)
        assert set(svd.keys()) == {0, 3, 7}

    def test_singular_values_are_tensors(self, tiny_llama):
        svd = weight_svd(tiny_llama, layers=[0])
        for key, values in svd[0].items():
            assert isinstance(values, torch.Tensor)
            assert len(values) > 0

    def test_singular_values_are_non_negative(self, tiny_llama):
        svd = weight_svd(tiny_llama, layers=[0])
        for key, values in svd[0].items():
            assert (values >= 0).all()

    def test_default_all_layers(self, tiny_llama):
        svd = weight_svd(tiny_llama)
        assert len(svd) == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — ImportError

- [ ] **Step 3: Implement weight_norms and weight_svd**

Add to `testing/llm_surgeon/inspect.py`:

```python
def weight_norms(model) -> list:
    """Compute Frobenius norms of attention and MLP weights per layer.

    Returns list of dicts with keys: layer, attn_norm, mlp_norm, total_norm.
    """
    results = []
    for i, layer in enumerate(model.model.layers):
        attn_params = list(layer.self_attn.parameters())
        mlp_params = list(layer.mlp.parameters())

        attn_norm = sum(p.float().norm().item() ** 2 for p in attn_params) ** 0.5
        mlp_norm = sum(p.float().norm().item() ** 2 for p in mlp_params) ** 0.5
        total_norm = (attn_norm ** 2 + mlp_norm ** 2) ** 0.5

        results.append({
            "layer": i,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "total_norm": total_norm,
        })

        print(f"  Layer {i:2d}: attn={attn_norm:.2f}  mlp={mlp_norm:.2f}  total={total_norm:.2f}")

    return results


def weight_svd(model, layers: Optional[List[int]] = None) -> Dict[int, Dict[str, torch.Tensor]]:
    """Compute singular value spectra of key weight matrices per layer.

    Args:
        model: The model to analyze
        layers: Layer indices to analyze. None = all layers.

    Returns:
        Dict mapping layer index to dict of matrix_name -> singular values tensor.
    """
    num_layers = len(model.model.layers)
    if layers is None:
        layers = list(range(num_layers))

    results = {}
    for i in layers:
        layer = model.model.layers[i]
        layer_svd = {}

        # Key weight matrices in a LlamaDecoderLayer
        matrices = {
            "q_proj": layer.self_attn.q_proj.weight,
            "k_proj": layer.self_attn.k_proj.weight,
            "v_proj": layer.self_attn.v_proj.weight,
            "o_proj": layer.self_attn.o_proj.weight,
        }

        # MLP matrices
        if hasattr(layer.mlp, "gate_proj"):
            matrices["gate_proj"] = layer.mlp.gate_proj.weight
            matrices["up_proj"] = layer.mlp.up_proj.weight
            matrices["down_proj"] = layer.mlp.down_proj.weight

        for name, weight in matrices.items():
            s = torch.linalg.svdvals(weight.float())
            layer_svd[name] = s

        results[i] = layer_svd

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add testing/llm_surgeon/inspect.py testing/tests/test_inspect.py
git commit -m "feat: add weight_norms and weight_svd analysis"
```

---

### Task 3: Activation analysis (attention_entropy, residual_stream_norms)

**Files:**
- Modify: `testing/llm_surgeon/inspect.py`
- Modify: `testing/tests/test_inspect.py`

- [ ] **Step 1: Write tests**

Add to `testing/tests/test_inspect.py`:

```python
from llm_surgeon.inspect import attention_entropy, residual_stream_norms


class TestAttentionEntropy:
    def test_returns_dict_per_layer(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        entropy = attention_entropy(tiny_llama, tokenizer, prompt="tok3 tok4 tok5 tok6 tok7")
        assert isinstance(entropy, dict)
        assert len(entropy) == 8

    def test_per_head_values(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        entropy = attention_entropy(tiny_llama, tokenizer, prompt="tok3 tok4 tok5 tok6 tok7")
        # tiny_llama has 4 attention heads
        for layer_idx, heads in entropy.items():
            assert isinstance(heads, list)
            assert len(heads) == 4

    def test_entropy_is_non_negative(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        entropy = attention_entropy(tiny_llama, tokenizer, prompt="tok3 tok4 tok5 tok6 tok7")
        for layer_idx, heads in entropy.items():
            for h_val in heads:
                assert h_val >= 0.0


class TestResidualStreamNorms:
    def test_returns_list(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        norms = residual_stream_norms(tiny_llama, tokenizer, prompt="tok3 tok4 tok5 tok6 tok7")
        assert isinstance(norms, list)
        # 8 layers + 1 for input embedding = 9 boundary points
        assert len(norms) == 9

    def test_norms_are_positive(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        norms = residual_stream_norms(tiny_llama, tokenizer, prompt="tok3 tok4 tok5 tok6 tok7")
        for n in norms:
            assert n > 0.0

    def test_first_entry_is_embedding_norm(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        norms = residual_stream_norms(tiny_llama, tokenizer, prompt="tok3 tok4 tok5 tok6 tok7")
        # First entry should be the norm of the embedding output (before any layers)
        assert isinstance(norms[0], float)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — ImportError

- [ ] **Step 3: Implement attention_entropy and residual_stream_norms**

Add to `testing/llm_surgeon/inspect.py`:

```python
def attention_entropy(
    model,
    tokenizer,
    prompt: str,
) -> Dict[int, List[float]]:
    """Compute entropy of attention distributions per layer, per head.

    Returns dict mapping layer index to list of per-head entropy values.
    Low entropy = focused attention, high entropy = diffuse/uniform.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    # We need attention weights — use output_attentions=True
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq) per layer

    results = {}
    for i, attn in enumerate(attentions):
        # attn shape: (batch, num_heads, seq_len, seq_len)
        # Compute entropy per head, averaged over query positions
        attn_probs = attn[0].float()  # (num_heads, seq, seq)
        # Entropy: -sum(p * log(p)) per query position, then average
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy_per_pos = -(attn_probs * (attn_probs + eps).log()).sum(dim=-1)  # (heads, seq)
        mean_entropy_per_head = entropy_per_pos.mean(dim=-1)  # (heads,)
        results[i] = mean_entropy_per_head.tolist()

    return results


def residual_stream_norms(
    model,
    tokenizer,
    prompt: str,
) -> List[float]:
    """Compute the magnitude of the residual stream at each layer boundary.

    Returns list of norms: [embedding_output, after_layer_0, ..., after_layer_N-1].
    Length = num_layers + 1.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    norms = []

    # Hook on embedding output
    embed_output = {}
    def embed_hook(module, inp, out):
        embed_output["val"] = out.detach()

    h = model.model.embed_tokens.register_forward_hook(embed_hook)

    # Hooks on each layer output
    layer_outputs = {}
    hooks = [h]
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    layer_outputs[idx] = out[0].detach()
                else:
                    layer_outputs[idx] = out.detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # Compute norms (mean over batch and seq dims)
    norms.append(embed_output["val"].float().norm(dim=-1).mean().item())
    for i in range(len(model.model.layers)):
        norms.append(layer_outputs[i].float().norm(dim=-1).mean().item())

    return norms
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add testing/llm_surgeon/inspect.py testing/tests/test_inspect.py
git commit -m "feat: add attention_entropy and residual_stream_norms analysis"
```

---

### Task 4: Activation comparison + cached baselines (verify.py)

**Files:**
- Modify: `testing/llm_surgeon/verify.py`
- Modify: `testing/tests/test_verify.py`

- [ ] **Step 1: Write tests**

Add to `testing/tests/test_verify.py`:

```python
from tests.conftest import _make_tiny_tokenizer
from llm_surgeon.verify import compare_activations, cache_baseline, compare_to_baseline
from llm_surgeon.surgery import remove_layers
from transformers import LlamaForCausalLM
import torch


class TestCompareActivations:
    def test_returns_list_of_dicts(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        # Create a modified copy
        model2 = LlamaForCausalLM(tiny_llama_config)
        model2.eval()
        model2.load_state_dict(tiny_llama.state_dict())
        remove_layers(model2, [7])

        diff = compare_activations(
            tiny_llama, model2, tokenizer,
            prompt="tok3 tok4 tok5 tok6 tok7",
        )
        assert isinstance(diff, list)
        # Should have min(8, 7) = 7 entries (aligns to shorter model)
        assert len(diff) == 7

    def test_each_entry_has_metrics(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        model2 = LlamaForCausalLM(tiny_llama_config)
        model2.eval()
        model2.load_state_dict(tiny_llama.state_dict())
        remove_layers(model2, [7])

        diff = compare_activations(
            tiny_llama, model2, tokenizer,
            prompt="tok3 tok4 tok5 tok6 tok7",
        )
        for entry in diff:
            assert "layer" in entry
            assert "cosine_sim" in entry
            assert "l2_dist" in entry

    def test_identical_models_have_cosine_one(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        diff = compare_activations(
            tiny_llama, tiny_llama, tokenizer,
            prompt="tok3 tok4 tok5 tok6 tok7",
        )
        for entry in diff:
            assert abs(entry["cosine_sim"] - 1.0) < 1e-5


class TestCachedBaseline:
    def test_cache_creates_files(self, tiny_llama, tiny_llama_config, tmp_path):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        cache_dir = str(tmp_path / "baseline")
        cache_baseline(
            tiny_llama, tokenizer,
            prompts=["tok3 tok4 tok5"],
            cache_dir=cache_dir,
        )
        # Should have created .pt files
        import os
        files = os.listdir(cache_dir)
        assert len(files) > 0
        assert any(f.endswith(".pt") for f in files)

    def test_compare_to_baseline_works(self, tiny_llama, tiny_llama_config, tmp_path):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        cache_dir = str(tmp_path / "baseline")

        # Cache original
        cache_baseline(
            tiny_llama, tokenizer,
            prompts=["tok3 tok4 tok5"],
            cache_dir=cache_dir,
        )

        # Modify model
        model2 = LlamaForCausalLM(tiny_llama_config)
        model2.eval()
        model2.load_state_dict(tiny_llama.state_dict())
        remove_layers(model2, [7])

        diff = compare_to_baseline(
            model2, tokenizer,
            prompts=["tok3 tok4 tok5"],
            cache_dir=cache_dir,
        )
        assert isinstance(diff, dict)
        # Should have one entry per prompt
        assert len(diff) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — ImportError

- [ ] **Step 3: Implement compare_activations, cache_baseline, compare_to_baseline**

Add to `testing/llm_surgeon/verify.py`:

```python
import os
import hashlib
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional


def _capture_layer_activations(model, tokenizer, prompt: str) -> List[torch.Tensor]:
    """Run a forward pass and capture the output of each layer."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    layer_outputs = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    layer_outputs[idx] = out[0].detach().cpu()
                else:
                    layer_outputs[idx] = out.detach().cpu()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return [layer_outputs[i] for i in range(len(model.model.layers))]


def compare_activations(
    original,
    modified,
    tokenizer,
    prompt: str,
    layers: str = "all",
) -> list:
    """Compare activations between two models layer by layer.

    For models with different layer counts, aligns from layer 0 and
    compares up to the shorter model's depth.

    Returns list of dicts with: layer, cosine_sim, l2_dist, max_abs_diff.
    """
    acts_orig = _capture_layer_activations(original, tokenizer, prompt)
    acts_mod = _capture_layer_activations(modified, tokenizer, prompt)

    num_compare = min(len(acts_orig), len(acts_mod))

    results = []
    for i in range(num_compare):
        a = acts_orig[i].float()
        b = acts_mod[i].float()

        cos_sim = F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
        l2 = (a - b).norm().item()
        max_diff = (a - b).abs().max().item()

        results.append({
            "layer": i,
            "cosine_sim": cos_sim,
            "l2_dist": l2,
            "max_abs_diff": max_diff,
        })

    return results


def cache_baseline(
    model,
    tokenizer,
    prompts: List[str],
    cache_dir: str,
) -> None:
    """Cache activations from a model for later comparison.

    Saves per-layer activations as .pt files keyed by prompt hash.
    """
    os.makedirs(cache_dir, exist_ok=True)

    for prompt in prompts:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        activations = _capture_layer_activations(model, tokenizer, prompt)
        save_path = os.path.join(cache_dir, f"{prompt_hash}.pt")
        torch.save({
            "prompt": prompt,
            "activations": activations,
            "num_layers": len(activations),
        }, save_path)


def compare_to_baseline(
    model,
    tokenizer,
    prompts: List[str],
    cache_dir: str,
) -> Dict[str, list]:
    """Compare a model's activations against a cached baseline.

    Returns dict mapping prompt to comparison results (same format as compare_activations).
    """
    results = {}

    for prompt in prompts:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"{prompt_hash}.pt")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached baseline for prompt (hash={prompt_hash}). "
                f"Run cache_baseline first."
            )

        cached = torch.load(cache_path, weights_only=False)
        acts_baseline = cached["activations"]

        # Get current model activations
        acts_current = _capture_layer_activations(model, tokenizer, prompt)

        num_compare = min(len(acts_baseline), len(acts_current))

        comparisons = []
        for i in range(num_compare):
            a = acts_baseline[i].float()
            b = acts_current[i].float()

            cos_sim = F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
            l2 = (a - b).norm().item()
            max_diff = (a - b).abs().max().item()

            comparisons.append({
                "layer": i,
                "cosine_sim": cos_sim,
                "l2_dist": l2,
                "max_abs_diff": max_diff,
            })

        results[prompt] = comparisons

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`
Expected: All prior tests + new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/verify.py testing/tests/test_verify.py
git commit -m "feat: add activation comparison and cached baselines to verify.py"
```

---

## Final State

```
testing/
  llm_surgeon/
    __init__.py          — imports surgery, verify, export, benchmark, inspect
    surgery.py           — (Phase 1)
    verify.py            — check_structure + compare_activations, cache_baseline, compare_to_baseline
    export.py            — (Phase 2)
    benchmark.py         — (Phase 3)
    inspect.py           — block_influence, weight_norms, weight_svd, attention_entropy, residual_stream_norms
  tests/
    test_surgery.py      — (Phase 1)
    test_verify.py       — structural checks + activation comparison tests
    test_export.py       — (Phase 2)
    test_benchmark.py    — (Phase 3)
    test_inspect.py      — BI scores, weight analysis, activation analysis tests
```
