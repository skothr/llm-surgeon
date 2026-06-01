# probe.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `probe.py` module to llm_surgeon that extracts hidden states at sub-layer granularity, projects them into token space (logit lens), and supports forward-pass intervention with predefined and custom operations.

**Architecture:** Single new module `testing/llm_surgeon/probe.py` using PyTorch forward hooks to capture/modify residual stream states at post-attention and post-FFN points within each transformer layer. Dataclasses (`LogitLensResult`, `HiddenStates`, `InterventionResult`) hold results. An `ops` namespace provides predefined intervention callables. All functions accept an optional `on_layer` callback for streaming. Recipe integration via a new `analyze` phase in `recipe.py`.

**Tech Stack:** PyTorch (hooks, tensors), HuggingFace Transformers (LlamaForCausalLM), dataclasses, existing llm_surgeon tracking/recipe infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-11-probe-module-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `testing/llm_surgeon/probe.py` | Create | All probe/intervention logic: dataclasses, logit_lens, extract_hidden_states, intervene, ops namespace, layer_predictions_table |
| `testing/tests/test_probe.py` | Create | Tests for all probe.py public API |
| `testing/llm_surgeon/__init__.py` | Modify | Add `probe` to imports |
| `testing/llm_surgeon/recipe.py` | Modify | Add `analyze` phase calling probe functions |
| `testing/tests/test_recipe.py` | Modify | Add test for `analyze` recipe section |

---

## Phase 1: Observation

### Task 1: Dataclasses and module skeleton

**Files:**
- Create: `testing/llm_surgeon/probe.py`
- Create: `testing/tests/test_probe.py`
- Modify: `testing/llm_surgeon/__init__.py`

- [ ] **Step 1: Write failing test for LogitLensResult**

In `testing/tests/test_probe.py`:

```python
"""Tests for probe module — logit lens, hidden state extraction, intervention."""

from llm_surgeon.probe import LogitLensResult


def test_logit_lens_result_summary():
    result = LogitLensResult(
        predictions=[
            {"layer": 0, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "hello", "token_id": 5, "prob": 0.8, "rank": 0},
                       {"token": "world", "token_id": 6, "prob": 0.1, "rank": 1}]},
            {"layer": 1, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "world", "token_id": 6, "prob": 0.7, "rank": 0},
                       {"token": "hello", "token_id": 5, "prob": 0.2, "rank": 1}]},
        ],
        logits=None,
        prompt_tokens=["the"],
    )
    text = result.summary(position=0)
    assert "hello" in text
    assert "world" in text


def test_logit_lens_result_prediction_flips():
    result = LogitLensResult(
        predictions=[
            {"layer": 0, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "a", "token_id": 1, "prob": 0.5, "rank": 0}]},
            {"layer": 1, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "b", "token_id": 2, "prob": 0.5, "rank": 0}]},
            {"layer": 2, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "b", "token_id": 2, "prob": 0.6, "rank": 0}]},
            {"layer": 3, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "a", "token_id": 1, "prob": 0.7, "rank": 0}]},
        ],
        logits=None,
        prompt_tokens=["x"],
    )
    assert result.prediction_flips(position=0) == 2


def test_logit_lens_result_first_correct_layer():
    result = LogitLensResult(
        predictions=[
            {"layer": 0, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "wrong", "token_id": 1, "prob": 0.5, "rank": 0}]},
            {"layer": 1, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "right", "token_id": 2, "prob": 0.6, "rank": 0}]},
        ],
        logits=None,
        prompt_tokens=["x"],
    )
    assert result.first_correct_layer(position=0, target_token="right") == 1
    assert result.first_correct_layer(position=0, target_token="missing") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_logit_lens_result_summary tests/test_probe.py::test_logit_lens_result_prediction_flips tests/test_probe.py::test_logit_lens_result_first_correct_layer -v`

Expected: FAIL — `ImportError: cannot import name 'LogitLensResult' from 'llm_surgeon.probe'`

- [ ] **Step 3: Write LogitLensResult dataclass**

In `testing/llm_surgeon/probe.py`:

```python
"""Hidden state probing, logit lens, and forward-pass intervention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _get_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device


@dataclass
class LogitLensResult:
    predictions: List[Dict]
    logits: Optional[Dict[Tuple[int, str], torch.Tensor]]
    prompt_tokens: List[str]

    def summary(self, position: int = -1) -> str:
        filtered = [p for p in self.predictions if p["position"] == position]
        if not filtered and position == -1:
            max_pos = max((p["position"] for p in self.predictions), default=0)
            filtered = [p for p in self.predictions if p["position"] == max_pos]
        lines = []
        lines.append(f"{'Layer':>7} {'Sub':>5} {'Top-1':>12} {'Prob':>7} {'Top-3'}")
        lines.append("-" * 55)
        for p in filtered:
            top = p["top_k"]
            top1 = top[0]["token"] if top else "?"
            prob = f"{top[0]['prob']:.3f}" if top else "?"
            top3 = ", ".join(t["token"] for t in top[:3])
            lines.append(f"{p['layer']:>7} {p['sublayer']:>5} {top1:>12} {prob:>7} {top3}")
        return "\n".join(lines)

    def first_correct_layer(self, position: int, target_token: str) -> Optional[int]:
        for p in self.predictions:
            if p["position"] != position:
                continue
            if p["top_k"] and p["top_k"][0]["token"] == target_token:
                return p["layer"]
        return None

    def prediction_flips(self, position: int) -> int:
        tokens = []
        for p in self.predictions:
            if p["position"] != position:
                continue
            if p["top_k"]:
                tokens.append(p["top_k"][0]["token"])
        flips = 0
        for i in range(1, len(tokens)):
            if tokens[i] != tokens[i - 1]:
                flips += 1
        return flips
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_logit_lens_result_summary tests/test_probe.py::test_logit_lens_result_prediction_flips tests/test_probe.py::test_logit_lens_result_first_correct_layer -v`

Expected: 3 passed

- [ ] **Step 5: Write failing test for HiddenStates**

Append to `testing/tests/test_probe.py`:

```python
import torch

from llm_surgeon.probe import HiddenStates


def test_hidden_states_cosine_similarity_identical():
    t = torch.randn(5, 32)
    hs = HiddenStates(
        states={(0, "ffn"): t, (1, "ffn"): t},
        prompt_tokens=["a"] * 5,
    )
    sim = hs.cosine_similarity((0, "ffn"), (1, "ffn"), position=-1)
    assert abs(sim - 1.0) < 1e-5


def test_hidden_states_save_load(tmp_path):
    t = torch.randn(5, 32)
    hs = HiddenStates(
        states={(0, "ffn"): t, (2, "attn"): t * 2},
        prompt_tokens=["a", "b", "c", "d", "e"],
    )
    path = str(tmp_path / "states.pt")
    hs.save(path)
    loaded = HiddenStates.load(path)
    assert set(loaded.states.keys()) == {(0, "ffn"), (2, "attn")}
    assert loaded.prompt_tokens == hs.prompt_tokens
    assert torch.allclose(loaded.states[(0, "ffn")], t)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_hidden_states_cosine_similarity_identical tests/test_probe.py::test_hidden_states_save_load -v`

Expected: FAIL — `ImportError: cannot import name 'HiddenStates'`

- [ ] **Step 7: Write HiddenStates dataclass**

Append to `testing/llm_surgeon/probe.py`:

```python
@dataclass
class HiddenStates:
    states: Dict[Tuple[int, str], torch.Tensor]
    prompt_tokens: List[str]

    def cosine_similarity(
        self, a: Tuple[int, str], b: Tuple[int, str], position: int = -1
    ) -> float:
        va = self.states[a][position].float()
        vb = self.states[b][position].float()
        return F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()

    def save(self, path: str) -> None:
        serializable_states = {f"{k[0]}_{k[1]}": v for k, v in self.states.items()}
        torch.save(
            {"states": serializable_states, "prompt_tokens": self.prompt_tokens},
            path,
        )

    @staticmethod
    def load(path: str) -> "HiddenStates":
        data = torch.load(path, weights_only=False)
        states = {}
        for k, v in data["states"].items():
            parts = k.split("_", 1)
            states[(int(parts[0]), parts[1])] = v
        return HiddenStates(states=states, prompt_tokens=data["prompt_tokens"])
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_hidden_states_cosine_similarity_identical tests/test_probe.py::test_hidden_states_save_load -v`

Expected: 2 passed

- [ ] **Step 9: Update __init__.py**

Change `testing/llm_surgeon/__init__.py` to:

```python
"""LLM Surgeon — surgical layer-level manipulation of LLaMA models."""

from llm_surgeon import surgery, verify, export, benchmark, inspect, tracking, recipe, probe
```

- [ ] **Step 10: Commit**

```bash
cd /home/ai/ai-projects/llm/testing
git add llm_surgeon/probe.py tests/test_probe.py llm_surgeon/__init__.py
git commit -m "feat(probe): add LogitLensResult and HiddenStates dataclasses with tests"
```

---

### Task 2: Hook infrastructure and extract_hidden_states

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe.py`

- [ ] **Step 1: Write failing tests for extract_hidden_states**

Append to `testing/tests/test_probe.py`:

```python
from llm_surgeon.probe import extract_hidden_states


def test_extract_hidden_states_ffn_only(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    hs = extract_hidden_states(tiny_llama, tokenizer, prompt)
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(hs.states) == num_layers
    for i in range(num_layers):
        assert (i, "ffn") in hs.states
        assert hs.states[(i, "ffn")].shape[-1] == tiny_llama_config.hidden_size


def test_extract_hidden_states_both_sublayers(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    hs = extract_hidden_states(tiny_llama, tokenizer, prompt, sublayers=("attn", "ffn"))
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(hs.states) == num_layers * 2
    for i in range(num_layers):
        assert (i, "attn") in hs.states
        assert (i, "ffn") in hs.states


def test_extract_hidden_states_specific_layers(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    hs = extract_hidden_states(tiny_llama, tokenizer, prompt, layers=[0, 3, 7])
    assert set(hs.states.keys()) == {(0, "ffn"), (3, "ffn"), (7, "ffn")}


def test_extract_hidden_states_callback(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    calls = []
    def cb(layer, sublayer, data):
        calls.append((layer, sublayer))
        assert "hidden_state" in data
    extract_hidden_states(tiny_llama, tokenizer, prompt, on_layer=cb)
    assert len(calls) == tiny_llama_config.num_hidden_layers
```

Also add this helper at the top of the test file (after the imports):

```python
from tests.conftest import _make_tiny_tokenizer

def _make_test_tokenizer(vocab_size):
    return _make_tiny_tokenizer(vocab_size)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_extract_hidden_states_ffn_only tests/test_probe.py::test_extract_hidden_states_both_sublayers tests/test_probe.py::test_extract_hidden_states_specific_layers tests/test_probe.py::test_extract_hidden_states_callback -v`

Expected: FAIL — `ImportError: cannot import name 'extract_hidden_states'`

- [ ] **Step 3: Implement extract_hidden_states**

Append to `testing/llm_surgeon/probe.py`:

```python
def _capture_residual_stream(model, tokenizer, prompt, sublayers=("ffn",), layers=None):
    """Run a forward pass and capture residual stream states via hooks.

    Returns:
        captured: dict mapping (layer_idx, sublayer_name) -> Tensor (seq_len, d_model)
        prompt_tokens: list of token strings
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)
    target_layers = set(layers) if layers is not None else set(range(num_layers))
    capture_attn = "attn" in sublayers
    capture_ffn = "ffn" in sublayers

    captured: Dict[Tuple[int, str], torch.Tensor] = {}
    layer_block_inputs: Dict[int, torch.Tensor] = {}
    hooks = []

    if capture_attn:
        def make_block_pre_hook(idx):
            def hook(module, args):
                layer_block_inputs[idx] = args[0].detach()
            return hook

        def make_attn_hook(idx):
            def hook(module, inp, out):
                attn_out = out[0] if isinstance(out, tuple) else out
                h_in = layer_block_inputs[idx]
                captured[(idx, "attn")] = (h_in + attn_out.detach())[0]
            return hook

        for i in target_layers:
            layer = model.model.layers[i]
            hooks.append(layer.register_forward_pre_hook(make_block_pre_hook(i)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))

    if capture_ffn:
        def make_ffn_hook(idx):
            def hook(module, inp, out):
                hidden = out[0].detach() if isinstance(out, tuple) else out.detach()
                captured[(idx, "ffn")] = hidden[0]
            return hook

        for i in target_layers:
            layer = model.model.layers[i]
            hooks.append(layer.register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured, prompt_tokens


def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layers: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("ffn",),
    detach: bool = True,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
) -> HiddenStates:
    """Extract raw hidden state tensors at specified residual stream capture points.

    Capture points are on the residual stream: post-attention residual add ("attn")
    and post-FFN residual add / layer output ("ffn").
    """
    captured, prompt_tokens = _capture_residual_stream(
        model, tokenizer, prompt, sublayers=sublayers, layers=layers,
    )

    if on_layer is not None:
        for (layer_idx, sub), tensor in sorted(captured.items()):
            on_layer(layer_idx, sub, {"hidden_state": tensor})

    return HiddenStates(states=captured, prompt_tokens=prompt_tokens)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_extract_hidden_states_ffn_only tests/test_probe.py::test_extract_hidden_states_both_sublayers tests/test_probe.py::test_extract_hidden_states_specific_layers tests/test_probe.py::test_extract_hidden_states_callback -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing
git add llm_surgeon/probe.py tests/test_probe.py
git commit -m "feat(probe): add extract_hidden_states with sub-layer hook capture"
```

---

### Task 3: logit_lens

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe.py`

- [ ] **Step 1: Write failing tests for logit_lens**

Append to `testing/tests/test_probe.py`:

```python
from llm_surgeon.probe import logit_lens, layer_predictions_table


def test_logit_lens_basic(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=5)

    assert isinstance(result, LogitLensResult)
    assert result.logits is None
    assert len(result.prompt_tokens) > 0

    num_layers = tiny_llama_config.num_hidden_layers
    num_positions = len(result.prompt_tokens)
    # Default sublayers is ("attn", "ffn"), so 2 capture points per layer per position
    expected_predictions = num_layers * 2 * num_positions
    assert len(result.predictions) == expected_predictions

    for p in result.predictions:
        assert "layer" in p
        assert "sublayer" in p
        assert p["sublayer"] in ("attn", "ffn")
        assert "position" in p
        assert "top_k" in p
        assert len(p["top_k"]) <= 5
        if p["top_k"]:
            assert "token" in p["top_k"][0]
            assert "prob" in p["top_k"][0]
            assert p["top_k"][0]["prob"] >= 0.0


def test_logit_lens_full_logits(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=3, full_logits=True)

    assert result.logits is not None
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(result.logits) == num_layers * 2
    for key, tensor in result.logits.items():
        layer_idx, sublayer = key
        assert tensor.shape[-1] == tiny_llama_config.vocab_size


def test_logit_lens_positions_filter(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=3, positions=[-1])

    positions_seen = set(p["position"] for p in result.predictions)
    assert len(positions_seen) == 1


def test_logit_lens_callback(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    calls = []
    def cb(layer, sublayer, data):
        calls.append((layer, sublayer))
        assert "hidden_state" in data
        assert "top_k" in data
    logit_lens(tiny_llama, tokenizer, prompt, top_k=3, on_layer=cb)
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(calls) == num_layers * 2


def test_layer_predictions_table(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=3)
    table = layer_predictions_table(result, position=-1)
    assert isinstance(table, str)
    assert len(table) > 0
    assert "Layer" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_logit_lens_basic tests/test_probe.py::test_logit_lens_full_logits tests/test_probe.py::test_logit_lens_positions_filter tests/test_probe.py::test_logit_lens_callback tests/test_probe.py::test_layer_predictions_table -v`

Expected: FAIL — `ImportError: cannot import name 'logit_lens'`

- [ ] **Step 3: Implement logit_lens and layer_predictions_table**

Append to `testing/llm_surgeon/probe.py`:

```python
def _project_to_logits(model, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply final RMSNorm + lm_head to a hidden state tensor.

    Args:
        hidden_state: (seq_len, d_model) tensor from the residual stream.

    Returns:
        (seq_len, vocab_size) logit tensor.
    """
    h = hidden_state.unsqueeze(0).to(_get_input_device(model))
    h = model.model.norm(h)
    return model.lm_head(h)[0]


def logit_lens(
    model,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    full_logits: bool = False,
    positions: Optional[List[int]] = None,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
) -> LogitLensResult:
    """Project each layer's residual stream state through the output head.

    Captures at both post-attention and post-FFN points (sub-layer granularity).
    """
    captured, prompt_tokens = _capture_residual_stream(
        model, tokenizer, prompt, sublayers=("attn", "ffn"),
    )

    seq_len = len(prompt_tokens)
    if positions is not None:
        resolved_positions = [p % seq_len for p in positions]
    else:
        resolved_positions = list(range(seq_len))

    predictions = []
    logits_dict: Dict[Tuple[int, str], torch.Tensor] = {} if full_logits else None

    for (layer_idx, sublayer), hidden in sorted(captured.items()):
        with torch.no_grad():
            layer_logits = _project_to_logits(model, hidden)

        if full_logits:
            logits_dict[(layer_idx, sublayer)] = layer_logits.cpu()

        probs = F.softmax(layer_logits.float(), dim=-1)

        cb_top_k_summary = []
        for pos in resolved_positions:
            pos_probs = probs[pos]
            topk_probs, topk_ids = pos_probs.topk(min(top_k, pos_probs.shape[0]))
            top_k_list = []
            for rank, (tid, tp) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist())):
                token_str = tokenizer.decode([tid])
                top_k_list.append({
                    "token": token_str,
                    "token_id": tid,
                    "prob": tp,
                    "rank": rank,
                })
            predictions.append({
                "layer": layer_idx,
                "sublayer": sublayer,
                "position": pos,
                "top_k": top_k_list,
            })
            if rank == 0:
                cb_top_k_summary.append((token_str, tp))

        if on_layer is not None:
            cb_logits = layer_logits.cpu() if full_logits else None
            on_layer(layer_idx, sublayer, {
                "hidden_state": hidden,
                "top_k": cb_top_k_summary,
                "logits": cb_logits,
            })

    return LogitLensResult(
        predictions=predictions,
        logits=logits_dict,
        prompt_tokens=prompt_tokens,
    )


def layer_predictions_table(result: LogitLensResult, position: int = -1) -> str:
    """Format a single position's logit lens predictions as a readable table."""
    return result.summary(position=position)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_logit_lens_basic tests/test_probe.py::test_logit_lens_full_logits tests/test_probe.py::test_logit_lens_positions_filter tests/test_probe.py::test_logit_lens_callback tests/test_probe.py::test_layer_predictions_table -v`

Expected: 5 passed

- [ ] **Step 5: Run all probe tests so far**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py -v`

Expected: All tests pass (10 total)

- [ ] **Step 6: Commit**

```bash
cd /home/ai/ai-projects/llm/testing
git add llm_surgeon/probe.py tests/test_probe.py
git commit -m "feat(probe): add logit_lens and layer_predictions_table"
```

---

## Phase 2: Intervention

### Task 4: Predefined operations (ops namespace)

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe.py`

- [ ] **Step 1: Write failing tests for ops**

Append to `testing/tests/test_probe.py`:

```python
from llm_surgeon.probe import ops


def test_ops_scale():
    fn = ops.scale(2.0)
    t = torch.ones(4, 8)
    result = fn(t, 0)
    assert torch.allclose(result, t * 2.0)
    assert "scale(2.0)" in repr(fn)


def test_ops_scale_identity():
    fn = ops.scale(1.0)
    t = torch.randn(4, 8)
    result = fn(t, 0)
    assert torch.allclose(result, t)


def test_ops_zero_dims():
    fn = ops.zero_dims([0, 2, 4])
    t = torch.ones(3, 8)
    result = fn(t, 0)
    assert result[:, 0].sum() == 0
    assert result[:, 2].sum() == 0
    assert result[:, 4].sum() == 0
    assert result[:, 1].sum() == 3  # untouched
    assert "zero_dims" in repr(fn)


def test_ops_clamp():
    fn = ops.clamp(-0.5, 0.5)
    t = torch.tensor([[-1.0, 0.0, 1.0]])
    result = fn(t, 0)
    assert result.min().item() >= -0.5
    assert result.max().item() <= 0.5
    assert "clamp" in repr(fn)


def test_ops_noise():
    fn = ops.noise(0.1, seed=42)
    t = torch.zeros(4, 8)
    result = fn(t, 0)
    assert not torch.allclose(result, t)
    assert result.abs().max().item() < 1.0  # noise should be small
    # Deterministic with seed
    result2 = ops.noise(0.1, seed=42)(t, 0)
    assert torch.allclose(result, result2)
    assert "noise" in repr(fn)


def test_ops_replace():
    replacement = torch.ones(4, 8) * 5
    fn = ops.replace(replacement)
    t = torch.zeros(4, 8)
    result = fn(t, 0)
    assert torch.allclose(result, replacement)
    assert "replace" in repr(fn)


def test_ops_project_out():
    direction = torch.zeros(8)
    direction[0] = 1.0  # unit vector along dim 0
    fn = ops.project_out(direction)
    t = torch.ones(3, 8)
    result = fn(t, 0)
    assert result[:, 0].abs().max().item() < 1e-5  # component along direction removed
    assert torch.allclose(result[:, 1:], t[:, 1:])  # other dims untouched
    assert "project_out" in repr(fn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_ops_scale tests/test_probe.py::test_ops_zero_dims tests/test_probe.py::test_ops_clamp tests/test_probe.py::test_ops_noise tests/test_probe.py::test_ops_replace tests/test_probe.py::test_ops_project_out -v`

Expected: FAIL — `ImportError: cannot import name 'ops'`

- [ ] **Step 3: Implement ops namespace**

Append to `testing/llm_surgeon/probe.py`:

```python
class _Op:
    """Callable wrapper with descriptive repr for experiment logging."""

    def __init__(self, fn, name: str):
        self._fn = fn
        self._name = name

    def __call__(self, hidden_state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._fn(hidden_state, layer_idx)

    def __repr__(self) -> str:
        return self._name


class _Ops:
    """Factory namespace for predefined intervention operations."""

    @staticmethod
    def scale(factor: float) -> _Op:
        return _Op(lambda h, _: h * factor, f"scale({factor})")

    @staticmethod
    def zero_dims(dims: List[int]) -> _Op:
        def fn(h, _):
            out = h.clone()
            out[:, dims] = 0
            return out
        return _Op(fn, f"zero_dims({dims})")

    @staticmethod
    def clamp(min_val: float, max_val: float) -> _Op:
        return _Op(lambda h, _: h.clamp(min=min_val, max=max_val), f"clamp({min_val}, {max_val})")

    @staticmethod
    def noise(std: float, seed: Optional[int] = None) -> _Op:
        def fn(h, _):
            gen = torch.Generator(device=h.device)
            if seed is not None:
                gen.manual_seed(seed)
            return h + torch.randn_like(h, generator=gen) * std
        return _Op(fn, f"noise(std={std})")

    @staticmethod
    def replace(tensor: torch.Tensor) -> _Op:
        return _Op(lambda h, _: tensor.to(h.device), "replace(<tensor>)")

    @staticmethod
    def project_out(direction: torch.Tensor) -> _Op:
        def fn(h, _):
            d = direction.to(h.device).float()
            d = d / d.norm()
            proj = (h.float() @ d).unsqueeze(-1) * d.unsqueeze(0)
            return (h.float() - proj).to(h.dtype)
        return _Op(fn, "project_out(<direction>)")


ops = _Ops()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_ops_scale tests/test_probe.py::test_ops_zero_dims tests/test_probe.py::test_ops_clamp tests/test_probe.py::test_ops_noise tests/test_probe.py::test_ops_replace tests/test_probe.py::test_ops_project_out tests/test_probe.py::test_ops_scale_identity -v`

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing
git add llm_surgeon/probe.py tests/test_probe.py
git commit -m "feat(probe): add predefined intervention ops (scale, zero_dims, clamp, noise, replace, project_out)"
```

---

### Task 5: intervene()

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe.py`

- [ ] **Step 1: Write failing tests for intervene**

Append to `testing/tests/test_probe.py`:

```python
from llm_surgeon.probe import intervene, Intervention, InterventionResult


def test_intervene_scale_identity(tiny_llama, tiny_llama_config):
    """Scaling by 1.0 should produce the same output as an unmodified forward pass."""
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"

    # Baseline: normal forward pass
    device = tiny_llama.model.embed_tokens.weight.device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        baseline_logits = tiny_llama(input_ids).logits[0]

    # Intervention: scale(1.0) at layer 0 ffn — should be identity
    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=0, sublayer="ffn", fn=ops.scale(1.0))],
    )
    assert isinstance(result, InterventionResult)
    assert torch.allclose(result.output_logits, baseline_logits, atol=1e-4)


def test_intervene_scale_zero_changes_output(tiny_llama, tiny_llama_config):
    """Zeroing layer 0 should produce different output."""
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"

    device = tiny_llama.model.embed_tokens.weight.device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        baseline_logits = tiny_llama(input_ids).logits[0]

    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=0, sublayer="ffn", fn=ops.scale(0.0))],
    )
    assert not torch.allclose(result.output_logits, baseline_logits, atol=1e-3)


def test_intervene_with_logit_lens(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=2, sublayer="ffn", fn=ops.scale(0.5))],
        capture_logit_lens=True,
        top_k=3,
    )
    assert result.logit_lens_result is not None
    assert len(result.logit_lens_result.predictions) > 0


def test_intervene_callback_modified_flag(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    calls = []
    def cb(layer, sublayer, data):
        calls.append((layer, sublayer, data["modified"]))
    intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=3, sublayer="ffn", fn=ops.scale(2.0))],
        on_layer=cb,
    )
    modified_calls = [(l, s) for l, s, m in calls if m]
    assert (3, "ffn") in modified_calls
    unmodified_calls = [(l, s) for l, s, m in calls if not m]
    assert len(unmodified_calls) > 0


def test_intervene_metadata(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=1, sublayer="ffn", fn=ops.noise(0.1, seed=0))],
    )
    assert len(result.interventions_applied) == 1
    meta = result.interventions_applied[0]
    assert meta["layer"] == 1
    assert meta["sublayer"] == "ffn"
    assert "noise" in meta["op_repr"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_intervene_scale_identity tests/test_probe.py::test_intervene_scale_zero_changes_output tests/test_probe.py::test_intervene_with_logit_lens tests/test_probe.py::test_intervene_callback_modified_flag tests/test_probe.py::test_intervene_metadata -v`

Expected: FAIL — `ImportError: cannot import name 'intervene'`

- [ ] **Step 3: Implement Intervention dataclass and intervene()**

Append to `testing/llm_surgeon/probe.py`:

```python
@dataclass
class Intervention:
    layer: int
    sublayer: str  # "attn" or "ffn"
    fn: Callable[[torch.Tensor, int], torch.Tensor]


@dataclass
class InterventionResult:
    output_logits: torch.Tensor
    logit_lens_result: Optional[LogitLensResult]
    interventions_applied: List[Dict]


def intervene(
    model,
    tokenizer,
    prompt: str,
    interventions: List[Intervention],
    capture_logit_lens: bool = False,
    top_k: int = 10,
    on_layer: Optional[Callable[[int, str, Dict], None]] = None,
) -> InterventionResult:
    """Run a forward pass with hidden state modifications at specified points.

    Optionally captures logit lens data at every capture point to observe
    the downstream effect of interventions.
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)

    # Build lookup: (layer, sublayer) -> intervention fn
    intervention_map: Dict[Tuple[int, str], Callable] = {}
    for iv in interventions:
        intervention_map[(iv.layer, iv.sublayer)] = iv.fn

    # Capture storage for logit lens
    captured_states: Dict[Tuple[int, str], torch.Tensor] = {} if capture_logit_lens else None
    callback_data = []

    hooks = []
    layer_block_inputs: Dict[int, torch.Tensor] = {}

    # Pre-hooks to capture block inputs (needed for attn sublayer intervention)
    for i in range(num_layers):
        def make_pre(idx):
            def hook(module, args):
                layer_block_inputs[idx] = args[0].detach()
            return hook
        hooks.append(model.model.layers[i].register_forward_pre_hook(make_pre(i)))

    # Attention hooks
    for i in range(num_layers):
        def make_attn_hook(idx):
            def hook(module, inp, out):
                attn_out = out[0] if isinstance(out, tuple) else out
                h_post_attn = layer_block_inputs[idx] + attn_out.detach()
                state = h_post_attn[0]

                modified = False
                if (idx, "attn") in intervention_map:
                    state = intervention_map[(idx, "attn")](state, idx)
                    modified = True

                if captured_states is not None:
                    captured_states[(idx, "attn")] = state

                if on_layer is not None:
                    on_layer(idx, "attn", {
                        "hidden_state": state,
                        "modified": modified,
                        "top_k": None,
                    })

                if modified:
                    # Reconstruct: modified state - original input = new attn output
                    new_attn_out = state.unsqueeze(0) - layer_block_inputs[idx]
                    if isinstance(out, tuple):
                        return (new_attn_out,) + out[1:]
                    return new_attn_out
            return hook
        hooks.append(model.model.layers[i].self_attn.register_forward_hook(make_attn_hook(i)))

    # FFN hooks (on the full layer block)
    for i in range(num_layers):
        def make_ffn_hook(idx):
            def hook(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                state = hidden[0].detach()

                modified = False
                if (idx, "ffn") in intervention_map:
                    state = intervention_map[(idx, "ffn")](state, idx)
                    modified = True

                if captured_states is not None:
                    captured_states[(idx, "ffn")] = state

                if on_layer is not None:
                    on_layer(idx, "ffn", {
                        "hidden_state": state,
                        "modified": modified,
                        "top_k": None,
                    })

                if modified:
                    new_out = state.unsqueeze(0)
                    if isinstance(out, tuple):
                        return (new_out,) + out[1:]
                    return new_out
            return hook
        hooks.append(model.model.layers[i].register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            model_output = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    output_logits = model_output.logits[0]

    # Build logit lens result if requested
    logit_lens_result = None
    if capture_logit_lens and captured_states:
        seq_len = len(prompt_tokens)
        all_positions = list(range(seq_len))
        predictions = []
        for (layer_idx, sublayer), hidden in sorted(captured_states.items()):
            with torch.no_grad():
                layer_logits = _project_to_logits(model, hidden)
            probs = F.softmax(layer_logits.float(), dim=-1)
            for pos in all_positions:
                pos_probs = probs[pos]
                topk_probs, topk_ids = pos_probs.topk(min(top_k, pos_probs.shape[0]))
                top_k_list = []
                for rank, (tid, tp) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist())):
                    top_k_list.append({
                        "token": tokenizer.decode([tid]),
                        "token_id": tid,
                        "prob": tp,
                        "rank": rank,
                    })
                predictions.append({
                    "layer": layer_idx,
                    "sublayer": sublayer,
                    "position": pos,
                    "top_k": top_k_list,
                })
        logit_lens_result = LogitLensResult(
            predictions=predictions,
            logits=None,
            prompt_tokens=prompt_tokens,
        )

    interventions_applied = [
        {"layer": iv.layer, "sublayer": iv.sublayer, "op_repr": repr(iv.fn)}
        for iv in interventions
    ]

    return InterventionResult(
        output_logits=output_logits,
        logit_lens_result=logit_lens_result,
        interventions_applied=interventions_applied,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py::test_intervene_scale_identity tests/test_probe.py::test_intervene_scale_zero_changes_output tests/test_probe.py::test_intervene_with_logit_lens tests/test_probe.py::test_intervene_callback_modified_flag tests/test_probe.py::test_intervene_metadata -v`

Expected: 5 passed

- [ ] **Step 5: Run all probe tests**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_probe.py -v`

Expected: All tests pass (22 total)

- [ ] **Step 6: Commit**

```bash
cd /home/ai/ai-projects/llm/testing
git add llm_surgeon/probe.py tests/test_probe.py
git commit -m "feat(probe): add intervene() with hook-based hidden state modification"
```

---

## Integration

### Task 6: Recipe integration

**Files:**
- Modify: `testing/llm_surgeon/recipe.py`
- Modify: `testing/tests/test_recipe.py`

- [ ] **Step 1: Write failing test for analyze recipe section**

Append to `testing/tests/test_recipe.py`:

```python
def test_run_with_analyze(tiny_llama, tmp_path):
    """Recipe with an analyze section runs logit_lens and hidden_states."""
    import yaml
    from tests.conftest import _make_tiny_tokenizer
    from llm_surgeon import recipe

    tokenizer = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
    db_path = str(tmp_path / "test.db")

    recipe_data = {
        "name": "test-analyze",
        "base_model": "test",
        "surgery": [{"remove_layers": [7]}],
        "analyze": {
            "logit_lens": {"prompt": "word4 word5 word6", "top_k": 3},
            "hidden_states": {"prompt": "word4 word5 word6", "save": True},
        },
    }
    recipe_path = str(tmp_path / "analyze.yaml")
    with open(recipe_path, "w") as f:
        yaml.dump(recipe_data, f)

    result = recipe.run(
        recipe_path,
        model=tiny_llama,
        tokenizer=tokenizer,
        db_path=db_path,
        skip_export=True,
        skip_eval=True,
    )
    assert result["status"] == "completed"
    assert "analyze" in result
    assert "logit_lens" in result["analyze"]
    assert "hidden_states" in result["analyze"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_recipe.py::test_run_with_analyze -v`

Expected: FAIL — `analyze` key not in result

- [ ] **Step 3: Add analyze phase to recipe.py**

In `testing/llm_surgeon/recipe.py`, add import at top:

```python
from llm_surgeon import surgery, verify, tracking
```

Add `_run_analyze` function after `_run_evaluation`:

```python
def _run_analyze(
    model, tokenizer, analyze_cfg: Dict, exp: tracking.Experiment, verbose: bool,
) -> Dict[str, Any]:
    """Run analysis steps as specified in the recipe's analyze section."""
    from llm_surgeon import probe

    results = {}

    if "logit_lens" in analyze_cfg:
        ll_cfg = analyze_cfg["logit_lens"] or {}
        prompt = ll_cfg.get("prompt", "")
        top_k = int(ll_cfg.get("top_k", 5))
        _log(f"Running logit lens (top_k={top_k})...", verbose)
        ll_result = probe.logit_lens(model, tokenizer, prompt, top_k=top_k)

        # Log summary metrics
        num_positions = len(ll_result.prompt_tokens)
        last_pos = num_positions - 1
        exp.log_metric("logit_lens_prediction_flips", ll_result.prediction_flips(last_pos))

        results["logit_lens"] = {
            "num_layers_captured": len(set(p["layer"] for p in ll_result.predictions)),
            "prediction_flips": ll_result.prediction_flips(last_pos),
        }
        _log(f"Logit lens: {results['logit_lens']}", verbose)

    if "hidden_states" in analyze_cfg:
        hs_cfg = analyze_cfg["hidden_states"] or {}
        prompt = hs_cfg.get("prompt", "")
        _log(f"Extracting hidden states...", verbose)
        hs = probe.extract_hidden_states(model, tokenizer, prompt)
        results["hidden_states"] = {
            "num_capture_points": len(hs.states),
        }
        _log(f"Hidden states: {len(hs.states)} capture points", verbose)

    return results
```

In the `run()` function, add the analyze phase between verification and evaluation. After the line `_log(f"Logged {len(combined_log.ops)} surgery operations", verbose)` and before the `# Evaluation` comment, add:

```python
    # Analyze
    analyze_cfg = recipe_data.get("analyze", {})
    if analyze_cfg:
        _log("Running analysis...", verbose)
        analyze_results = _run_analyze(model, tokenizer, analyze_cfg, exp, verbose)
        result["analyze"] = analyze_results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_recipe.py::test_run_with_analyze -v`

Expected: PASS

- [ ] **Step 5: Run all recipe tests to check for regressions**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_recipe.py -v`

Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
cd /home/ai/ai-projects/llm/testing
git add llm_surgeon/recipe.py tests/test_recipe.py
git commit -m "feat(recipe): add analyze phase for logit_lens and hidden_states"
```

---

### Task 7: Full regression test

**Files:** None (test-only)

- [ ] **Step 1: Run the complete test suite**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/ -v --tb=short`

Expected: All tests pass, no regressions in existing modules.

- [ ] **Step 2: Verify probe module imports cleanly**

Run: `cd /home/ai/ai-projects/llm/testing && python -c "from llm_surgeon import probe; print(dir(probe))"`

Expected: Output includes `LogitLensResult`, `HiddenStates`, `Intervention`, `InterventionResult`, `extract_hidden_states`, `intervene`, `layer_predictions_table`, `logit_lens`, `ops`.

- [ ] **Step 3: Final commit if any fixups needed**

Only if previous steps revealed issues that needed fixing.
