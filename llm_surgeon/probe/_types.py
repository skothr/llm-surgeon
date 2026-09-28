"""Result and intervention dataclasses for the probe package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

@dataclass
class LogitLensResult:
    """Logit-lens rows, one per (layer, sublayer, position).

    ``position`` in each row is the resolved (non-negative) index. The query
    methods below accept negative positions too, resolved against
    ``len(prompt_tokens)`` the same way ``logit_lens(positions=...)`` resolves
    them.
    """

    predictions: list[dict]
    logits: dict[tuple[int, str], torch.Tensor] | None
    prompt_tokens: list[str]

    def _resolve_position(self, position: int) -> int:
        n = len(self.prompt_tokens)
        return position % n if n and position < 0 else position

    def _top1_rows(self, position: int, sublayer: str | None) -> list[dict]:
        pos = self._resolve_position(position)
        return [
            p for p in self.predictions
            if p["position"] == pos and p["top_k"]
            and (sublayer is None or p["sublayer"] == sublayer)
        ]

    def summary(self, position: int = -1) -> str:
        filtered = [
            p for p in self.predictions if p["position"] == self._resolve_position(position)
        ]
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

    def first_correct_layer(
        self, position: int, target_token: str | int, sublayer: str | None = None,
    ) -> int | None:
        """Earliest layer whose top-1 prediction at ``position`` is the target.

        ``target_token`` is either a token id (exact) or a decoded string.
        Prefer the id: single-token decoding drops SentencePiece's leading
        space, so the id of ``"▁Paris"`` decodes to ``"Paris"``, and a string
        match against ``" Paris"`` never succeeds.

        ``sublayer`` restricts the scan to one capture point ("attn" or
        "ffn"); None scans both in row order (L0.attn, L0.ffn, L1.attn, ...),
        and the returned layer does not say which sublayer matched.
        """
        for p in self._top1_rows(position, sublayer):
            top1 = p["top_k"][0]
            hit = (
                top1.get("token_id") == target_token
                if isinstance(target_token, int)
                else top1["token"] == target_token
            )
            if hit:
                return p["layer"]
        return None

    def prediction_flips(self, position: int, sublayer: str | None = None) -> int:
        """Number of top-1 changes along the rows at ``position``.

        Compares token ids (distinct ids that decode alike count as a flip).
        With ``sublayer=None`` the sequence interleaves attn and ffn rows
        (L0.attn, L0.ffn, L1.attn, ...), so a flip inside one layer counts;
        pass ``sublayer="ffn"`` to count layer-to-layer flips only.
        """
        tokens = [
            p["top_k"][0].get("token_id", p["top_k"][0]["token"])
            for p in self._top1_rows(position, sublayer)
        ]
        flips = 0
        for i in range(1, len(tokens)):
            if tokens[i] != tokens[i - 1]:
                flips += 1
        return flips


@dataclass
class CompareLogitLensResult:
    """Result of comparing two models' logit-lens outputs on the same prompt.

    comparisons is a list of per-cell dicts with shape:
        {
          "original_layer": int,
          "sublayer": str,
          "position": int,
          "top_k_a": [...],       # same shape as LogitLensResult.predictions[i]["top_k"]
          "top_k_b": [...],
          "metrics_a": {...},     # _cell_metrics output for side A
          "metrics_b": {...},
          "compare": {...},       # _pair_metrics output
        }
    """
    comparisons: list[dict]
    prompt_tokens: list[str]
    aligned_keys: list[tuple[int, str]]  # (original_layer, sublayer) pairs that were compared


@dataclass
class HiddenStates:
    states: dict[tuple[int, str], torch.Tensor]
    prompt_tokens: list[str]

    def cosine_similarity(
        self, a: tuple[int, str], b: tuple[int, str], position: int = -1
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
        data = torch.load(path, weights_only=True)
        states = {}
        for k, v in data["states"].items():
            parts = k.split("_", 1)
            states[(int(parts[0]), parts[1])] = v
        return HiddenStates(states=states, prompt_tokens=data["prompt_tokens"])


@dataclass
class Intervention:
    """One hidden-state edit applied by ``intervene``.

    Both capture points are on the residual stream, not on a sublayer's own
    output:

    - ``"attn"``: ``h_in + attn_out``, the residual after the attention add
      (the input to the post-attention norm). ``ops.zero_dims`` here zeroes
      residual dims, not attention-output dims.
    - ``"ffn"``: the decoder layer's output ``h_in + attn_out + mlp_out``.
      ``ops.scale(0.0)`` here zeroes the whole residual stream, not only the
      MLP contribution.

    ``layer`` may be negative (``-1`` is the last layer). ``fn`` receives the
    state with the batch dim stripped, shape ``(seq_len, d_model)``, plus the
    resolved layer index, and returns a tensor of the same shape. Several
    interventions at one (layer, sublayer) are applied in list order.
    """

    layer: int
    sublayer: str  # "attn" or "ffn"
    fn: Callable[[torch.Tensor, int], torch.Tensor]


@dataclass
class InterventionResult:
    output_logits: torch.Tensor
    logit_lens_result: LogitLensResult | None
    interventions_applied: list[dict]


@dataclass
class PatchingResult:
    cells: list[dict]
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: list[str]
    prompt_tokens_corrupted: list[str]
    direction: str
    measurement_position: int
    mode: str = "exact"                           # "exact" | "approx" | "approx_head" | "edge" | "circuit" | "approx_neuron"
    n_heads: int | None = None                  # set by attribution_patch_per_head / edge_attribution_patch / extract_circuit
    n_edges: int | None = None                  # set by edge_attribution_patch / extract_circuit (pre-filter count)
    n_edges_in_circuit: int | None = None       # set by extract_circuit
    n_nodes_in_circuit: int | None = None       # set by extract_circuit (includes the logits sink)
    tau: float | None = None                    # set by extract_circuit (applied threshold)
    n_neurons: int | None = None                # set by attribution_patch_per_neuron (= intermediate_size)
    n_steps: int | None = None                  # set by attribution_patch when n_steps > 1 (IG path steps)

