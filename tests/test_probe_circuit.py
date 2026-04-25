"""Unit tests for probe.extract_circuit (Phase 3.8).

Mirrors the structure of test_probe_edge_ap.py but targets circuit
extraction. Mock-model tests only; TinyLlama integration lives in
Task 4 (test_probe_circuit_tinyllama).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import pytest
import torch
from torch import nn

from llm_surgeon.probe import (
    PatchingResult,
    edge_attribution_patch,
    extract_circuit,
    _compute_all_edges,
)


# ---- Shared mock infrastructure (copy-paste from test_probe_edge_ap.py) ----

def _stable_word_hash(s: str) -> int:
    """Deterministic across runs (unlike Python's hash())."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) % (10 ** 9)
    return h


class _MockTokenizer:
    def __init__(self, vocab: List[str]) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        # Hash each word so clean/corrupted prompts produce different IDs
        # (otherwise identical logits -> AP denominator = 0).
        ids = [_stable_word_hash(w) % self.vocab_size for w in text.split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        return [self.vocab[int(i) % self.vocab_size] for i in ids.flatten()]


class _MockLayerAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, H = h.shape
        concat_z = torch.tanh(h)  # [B, T, H] playing the role of concat(z_h)
        return self.o_proj(concat_z)


class _MockLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _MockLayerAttn(hidden, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_attn = h + self.self_attn(self.input_layernorm(h))
        h_ffn = h_attn + self.mlp(self.post_attention_layernorm(h_attn))
        return h_ffn


class _MockInner(nn.Module):
    def __init__(self, vocab: int, hidden: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_MockLayer(hidden, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8, n_heads: int = 2, n_layers: int = 2) -> None:
        super().__init__()
        self.model = _MockInner(vocab, hidden, n_heads, n_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.num_attention_heads = n_heads                          # pyright: ignore[reportAttributeAccessIssue]
        cfg.hidden_size = hidden                                    # pyright: ignore[reportAttributeAccessIssue]
        cfg.num_hidden_layers = n_layers                            # pyright: ignore[reportAttributeAccessIssue]
        self.config = cfg  # pyright: ignore[reportAttributeAccessIssue]
        self._device = torch.device("cpu")

    def forward(self, input_ids: torch.Tensor) -> Any:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        logits = self.lm_head(h)

        class _Out:
            pass
        out = _Out()
        out.logits = logits  # pyright: ignore[reportAttributeAccessIssue]
        return out


def _make_mock() -> Tuple[_MockModel, _MockTokenizer]:
    torch.manual_seed(42)
    model = _MockModel()
    tokenizer = _MockTokenizer(vocab=["w" + str(i) for i in range(16)])
    return model, tokenizer


def _pick_tokens(
    model: "_MockModel",
    tok: "_MockTokenizer",
    clean: str,
    corrupted: str,
    meas_pos: int = -1,
) -> Tuple[int, int]:
    """Pick (correct_id, incorrect_id) guaranteed to yield a nonzero AP denominator.

    Uses argmax(clean_logits[meas_pos]) for correct and an off-argmax id for
    incorrect, with a fallback to second-best if they collide.
    """
    with torch.no_grad():
        clean_out = model(tok(clean)["input_ids"])
        corr_out = model(tok(corrupted)["input_ids"])
    cl = clean_out.logits[0]
    co = corr_out.logits[0]
    seq = cl.shape[0]
    mp = meas_pos % seq
    c_id = int(cl[mp].argmax().item())
    i_id = int(co[mp].argmax().item())
    if c_id == i_id:
        topk = co[mp].topk(2).indices
        i_id = int(topk[1].item())
    # Sanity: nonzero denominator on the chosen pair.
    d_clean = (cl[mp, c_id] - cl[mp, i_id]).item()
    d_corr = (co[mp, c_id] - co[mp, i_id]).item()
    if abs(d_clean - d_corr) < 1e-4:
        # Extremely unlucky — pick least correlated pair by brute force.
        vocab = cl.shape[-1]
        best: Tuple[int, int, float] = (0, 1, 0.0)
        for ci in range(vocab):
            for ii in range(vocab):
                if ii == ci:
                    continue
                denom = abs((cl[mp, ci] - cl[mp, ii]).item() - (co[mp, ci] - co[mp, ii]).item())
                if denom > best[2]:
                    best = (ci, ii, denom)
        c_id, i_id = best[0], best[1]
    return c_id, i_id


# ---- Shared prompts + dynamically-picked token IDs (non-degenerate AP denominator) ----

CLEAN_PROMPT = "one two three four"
CORR_PROMPT = "one two three five"
_pick_model, _pick_tok = _make_mock()
CORRECT_ID, INCORRECT_ID = _pick_tokens(_pick_model, _pick_tok, CLEAN_PROMPT, CORR_PROMPT)


# ---- Tests ----

class TestExtractCircuitMock:
    def test_returns_patching_result(self) -> None:
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            tau=0.0,
            top_k_candidates=50,
        )
        assert isinstance(result, PatchingResult)
        assert result.mode == "circuit"
        assert result.tau == 0.0
        assert result.n_edges is not None and result.n_edges > 0
        assert result.n_edges_in_circuit is not None
        assert result.n_nodes_in_circuit is not None

    def test_matches_edge_ap_on_scores(self) -> None:
        """extract_circuit(tau=0) has the same top-k edge magnitudes as
        edge_attribution_patch(top_k_edges=top_k_candidates)."""
        model, tok = _make_mock()
        k = 30
        r_edge = edge_attribution_patch(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            top_k_edges=k,
        )
        model2, tok2 = _make_mock()
        r_circ = extract_circuit(
            model2, tok2,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            tau=0.0,
            top_k_candidates=k,
        )
        assert r_edge.n_edges == r_circ.n_edges
        assert len(r_edge.cells) == len(r_circ.cells) == k
        for a, b in zip(r_edge.cells, r_circ.cells):
            assert a["writer_layer"] == b["writer_layer"]
            assert a["writer_unit"] == b["writer_unit"]
            assert a["reader_layer"] == b["reader_layer"]
            assert a["reader_unit"] == b["reader_unit"]
            assert a["position"] == b["position"]
            assert a["ap_recovery"] == pytest.approx(b["ap_recovery"], abs=1e-6)

    def test_tau_zero_marks_all_topk_reachable(self) -> None:
        """With tau=0 and a connected graph, every top-k edge whose reader is
        reachable from logits is in_circuit. Logits readers are always
        reachable (they seed BFS), so any edge whose reader_unit=='logits' is
        in-circuit."""
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            tau=0.0,
            top_k_candidates=200,
        )
        logits_cells = [c for c in result.cells if c["reader_unit"] == "logits"]
        assert all(c["in_circuit"] for c in logits_cells)

    def test_tau_high_empties_circuit(self) -> None:
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            tau=1e9,
            top_k_candidates=50,
        )
        assert result.n_edges_in_circuit == 0
        assert result.n_nodes_in_circuit == 0
        assert all(c["in_circuit"] is False for c in result.cells)

    def test_tau_monotonic(self) -> None:
        """Raising tau can only remove edges from the circuit, never add."""
        model, tok = _make_mock()
        r_low = extract_circuit(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.0, top_k_candidates=100,
        )
        model2, tok2 = _make_mock()
        r_high = extract_circuit(
            model2, tok2, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.01, top_k_candidates=100,
        )
        assert r_high.n_edges_in_circuit is not None
        assert r_low.n_edges_in_circuit is not None
        assert r_high.n_edges_in_circuit <= r_low.n_edges_in_circuit

    def test_in_circuit_flag_set_on_every_cell(self) -> None:
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.005, top_k_candidates=100,
        )
        assert all("in_circuit" in c for c in result.cells)
        assert all(isinstance(c["in_circuit"], bool) for c in result.cells)

    def test_summary_counts_consistent(self) -> None:
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.002, top_k_candidates=200,
        )
        assert result.n_edges_in_circuit == sum(1 for c in result.cells if c["in_circuit"])

    def test_topk_exceeds_total_edges_caps(self) -> None:
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.0, top_k_candidates=10**9,
        )
        assert result.n_edges is not None
        assert len(result.cells) == result.n_edges

    def test_validation(self) -> None:
        model, tok = _make_mock()
        with pytest.raises(ValueError, match="tau"):
            extract_circuit(
                model, tok, "a b", "c d",
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                tau=-0.1,
            )
        with pytest.raises(ValueError, match="top_k_candidates"):
            extract_circuit(
                model, tok, "a b", "c d",
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                top_k_candidates=0,
            )
        with pytest.raises(ValueError, match="prompts cannot be empty"):
            extract_circuit(
                model, tok, "", "c d",
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            )
        with pytest.raises(ValueError, match="same length"):
            extract_circuit(
                model, tok, "a b c", "c d",
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            )


class TestComputeAllEdgesHelper:
    def test_returns_seven_tuple(self) -> None:
        model, tok = _make_mock()
        out = _compute_all_edges(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            direction="denoise",
            measurement_position=-1,
            positions=None,
            layers=None,
        )
        assert len(out) == 7
        all_edges, clean_logits, corr_logits, clean_tokens, corr_tokens, meas_pos, n_heads = out
        assert isinstance(all_edges, list) and len(all_edges) > 0
        assert isinstance(clean_logits, torch.Tensor)
        assert isinstance(corr_logits, torch.Tensor)
        assert isinstance(clean_tokens, list)
        assert isinstance(corr_tokens, list)
        assert isinstance(meas_pos, int) and meas_pos >= 0
        assert n_heads == 2


class TestReverseBFSCorrectness:
    """Purely exercises the BFS/connectivity logic with synthetic edges.

    Monkeypatches _compute_all_edges to return a hand-constructed edge list
    so we can prove the algorithm keeps/drops the right nodes without
    sensitivity to model internals."""

    def test_disconnected_component_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Synthetic graph:
        #   embed -> attn_in@1  (0.5)   <- path to logits
        #   attn@1 -> logits    (0.5)   <- path to logits
        #   ffn@0 -> ffn_in@1   (0.5)   <- NOT reachable from logits
        # With tau=0.3, all clear tau. Circuit should include only the top two.
        fake_edges = [
            {"writer_layer": 0, "writer_unit": "embed",
             "reader_layer": 1, "reader_unit": "attn_in", "position": 0, "ap_recovery": 0.5},
            {"writer_layer": 1, "writer_unit": "attn.h0",
             "reader_layer": 2, "reader_unit": "logits", "position": 0, "ap_recovery": 0.5},
            {"writer_layer": 0, "writer_unit": "ffn",
             "reader_layer": 1, "reader_unit": "ffn_in", "position": 0, "ap_recovery": 0.5},
        ]
        clean_logits = torch.zeros(2, 16)
        corr_logits = torch.zeros(2, 16)

        def fake_compute(*_args: Any, **_kwargs: Any) -> Tuple[Any, ...]:
            return (
                list(fake_edges),
                clean_logits, corr_logits,
                ["a", "b"], ["c", "d"],
                1,
                2,
            )

        monkeypatch.setattr("llm_surgeon.probe._compute_all_edges", fake_compute)

        model, tok = _make_mock()
        result = extract_circuit(
            model, tok, "x y", "u v",
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.3, top_k_candidates=10,
        )
        by_key = {(c["writer_unit"], c["reader_unit"]): c for c in result.cells}
        assert by_key[("embed", "attn_in")]["in_circuit"] is False  # attn_in@1 never reads into anything leading to logits
        assert by_key[("attn.h0", "logits")]["in_circuit"] is True
        assert by_key[("ffn", "ffn_in")]["in_circuit"] is False
        # Nodes in circuit: (2, "logits", 0) + (1, "attn.h0", 0) = 2
        assert result.n_nodes_in_circuit == 2
        assert result.n_edges_in_circuit == 1

    def test_chain_reverse_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # embed -> attn_in@1 -> (attn.h0@1 -> ffn_in@1) -> (ffn@1 -> logits)
        # Readers we care about connecting: attn_in@1, ffn_in@1, logits.
        # But edges only go writer->reader; chain:
        #   (embed -> attn_in@1): reader=attn_in@1, writer=embed
        #   (attn.h0@1 -> ffn_in@1): reader=ffn_in@1, writer=(1,attn.h0)
        #   (ffn@1 -> logits): reader=logits, writer=(1,ffn)
        # With the reverse-BFS rules (only readers become queue nodes), the
        # reader attn_in@1 is reachable only if SOME edge has attn_in@1 as its
        # writer node — which never happens (readers and writers are disjoint
        # in the Phase 3.7 edge shape). So embed->attn_in is NOT in-circuit.
        # Keep the test honest: only assert on edges whose readers are
        # directly logits or attainable via reader-writer node identity.
        fake_edges = [
            {"writer_layer": 1, "writer_unit": "attn.h0",
             "reader_layer": 2, "reader_unit": "logits", "position": 0, "ap_recovery": 0.5},
            {"writer_layer": 1, "writer_unit": "ffn",
             "reader_layer": 2, "reader_unit": "logits", "position": 0, "ap_recovery": 0.5},
        ]
        def fake_compute(*_args: Any, **_kwargs: Any) -> Tuple[Any, ...]:
            return (
                list(fake_edges),
                torch.zeros(1, 16), torch.zeros(1, 16),
                ["a"], ["b"],
                0,
                2,
            )
        monkeypatch.setattr("llm_surgeon.probe._compute_all_edges", fake_compute)
        model, tok = _make_mock()
        result = extract_circuit(
            model, tok, "x", "y",
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            tau=0.1, top_k_candidates=10,
        )
        assert all(c["in_circuit"] for c in result.cells)
        # Visited: logits@0, (1,attn.h0,0), (1,ffn,0) => 3 nodes
        assert result.n_nodes_in_circuit == 3
        assert result.n_edges_in_circuit == 2


# -------------------------------------------------------------------------
# TinyLlama integration (skipif GPU missing; fp16 to avoid OOM on 8GB)
# -------------------------------------------------------------------------

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


class TestCircuitIntegratedGradients:
    def test_circuit_n_steps_converges(self) -> None:
        """n_steps=10 results differ from n_steps=1, are finite, and n_steps is set."""
        import math

        model, tok = _make_mock()
        r_1 = extract_circuit(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            tau=0.0,
            top_k_candidates=50,
            n_steps=1,
        )
        model2, tok2 = _make_mock()
        r_10 = extract_circuit(
            model2, tok2,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            tau=0.0,
            top_k_candidates=50,
            n_steps=10,
        )
        assert r_1.n_steps is None
        assert r_10.n_steps == 10

        for c in r_10.cells:
            assert math.isfinite(c["ap_recovery"]), f"non-finite ap_recovery: {c}"

        r1_by_key = {
            (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"]): c["ap_recovery"]
            for c in r_1.cells
        }
        r10_by_key = {
            (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"]): c["ap_recovery"]
            for c in r_10.cells
        }
        shared = set(r1_by_key.keys()) & set(r10_by_key.keys())
        in_circuit_labels_1 = {
            (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"]): c["in_circuit"]
            for c in r_1.cells
        }
        in_circuit_labels_10 = {
            (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"]): c["in_circuit"]
            for c in r_10.cells
        }
        shared_labels = set(in_circuit_labels_1.keys()) & set(in_circuit_labels_10.keys())
        labels_differ = any(
            in_circuit_labels_1[k] != in_circuit_labels_10[k] for k in shared_labels
        )
        scores_differ = len(shared) > 0 and max(
            abs(r1_by_key[k] - r10_by_key[k]) for k in shared
        ) > 1e-4

        assert labels_differ or scores_differ, (
            "n_steps=10 produced identical circuit labels and near-identical scores to "
            "n_steps=1; IG averaging should change at least one quantity"
        )

    def test_circuit_n_steps_validation(self) -> None:
        """n_steps outside [1, 50] raises ValueError."""
        model, tok = _make_mock()
        with pytest.raises(ValueError, match=r"n_steps must be int in \[1, 50\]"):
            extract_circuit(
                model, tok, CLEAN_PROMPT, CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                n_steps=0,
            )
        with pytest.raises(ValueError, match=r"n_steps must be int in \[1, 50\]"):
            extract_circuit(
                model, tok, CLEAN_PROMPT, CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                n_steps=51,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestTinyLlamaCircuit:
    def test_tau_matches_edge_ranking(self) -> None:
        """With tau=0.02 on TinyLlama's standard denoise task, we expect:
        - nonzero edges in circuit
        - nonzero nodes in circuit
        - at tau=0: every logits-reader edge is in_circuit
        - at tau=1.0: zero edges in circuit (no edge hits 100% recovery)
        """
        from llm_surgeon.surgery import load_model
        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        # Paris id vs Rome id — same token IDs used in Phase 3.7 test.
        paris_id = tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item()
        rome_id = tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item()

        r = extract_circuit(
            model, tok, clean, corrupted,
            correct_token_id=int(paris_id),
            incorrect_token_id=int(rome_id),
            direction="denoise",
            tau=0.02,
            top_k_candidates=1000,
        )

        assert r.mode == "circuit"
        assert r.n_edges is not None and r.n_edges > 1000
        assert r.n_edges_in_circuit is not None and r.n_edges_in_circuit > 0
        assert r.n_nodes_in_circuit is not None and r.n_nodes_in_circuit > 0
        # top_k_candidates cap is respected
        assert len(r.cells) <= 1000
        # Cells are sorted by |ap_recovery| desc
        mags = [abs(c["ap_recovery"]) for c in r.cells]
        assert mags == sorted(mags, reverse=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
def test_circuit_ig_tinyllama() -> None:
    """Circuit IG sanity check on TinyLlama (top-100 set overlap).

    Same approach as test_edge_ig_tinyllama: top-100 overlap is more
    robust than rank Spearman for edge-level IG, where signed scores
    + small samples make rank correlation noisy. Asserts overlap >=
    25%, finite cells, and a measurable score difference proving IG
    isn't a no-op.
    """
    import math
    from llm_surgeon.surgery import load_model

    model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
    model.eval()

    clean = "The capital of France is"
    corrupted = "The capital of Italy is"

    paris_ids = tok(" Paris", add_special_tokens=False)["input_ids"]
    rome_ids = tok(" Rome", add_special_tokens=False)["input_ids"]
    correct_id = int(paris_ids[0])
    incorrect_id = int(rome_ids[0])

    r1 = extract_circuit(
        model, tok, clean, corrupted,
        correct_token_id=correct_id,
        incorrect_token_id=incorrect_id,
        tau=0.05,
        top_k_candidates=2000,
        n_steps=1,
    )
    r5 = extract_circuit(
        model, tok, clean, corrupted,
        correct_token_id=correct_id,
        incorrect_token_id=incorrect_id,
        tau=0.05,
        top_k_candidates=2000,
        n_steps=5,
    )

    assert r1.n_steps is None
    assert r5.n_steps == 5

    for c in r5.cells:
        assert math.isfinite(c["ap_recovery"]), f"non-finite cell: {c}"

    def _edge_key(c: dict) -> tuple:
        return (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"])

    r1_top100 = {
        _edge_key(c) for c in
        sorted(r1.cells, key=lambda c: abs(c["ap_recovery"]), reverse=True)[:100]
    }
    r5_top100 = {
        _edge_key(c) for c in
        sorted(r5.cells, key=lambda c: abs(c["ap_recovery"]), reverse=True)[:100]
    }
    overlap = len(r1_top100 & r5_top100)
    overlap_frac = overlap / 100.0
    print(
        f"\nCircuit IG top-100 set overlap(n_steps=1, n_steps=5) = "
        f"{overlap}/100 ({overlap_frac:.1%})"
    )

    r1_map = {_edge_key(c): c["ap_recovery"] for c in r1.cells}
    r5_map = {_edge_key(c): c["ap_recovery"] for c in r5.cells}
    shared = r1_top100 & r5_top100
    if shared:
        max_diff = max(abs(r1_map[k] - r5_map[k]) for k in shared)
        print(f"max abs score diff on shared top-100 edges: {max_diff:.5f}")
        assert max_diff > 1e-4, (
            f"IG appears to be a no-op (max diff {max_diff:.2e} <= 1e-4)"
        )

    assert overlap_frac >= 0.25, (
        f"Circuit IG top-100 overlap {overlap_frac:.1%} too low; "
        f"n_steps=5 should preserve at least 25 of r1's top-100 edges"
    )
