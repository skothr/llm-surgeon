"""Result and intervention dataclasses for the probe package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

@dataclass
class LogitLensResult:
    predictions: list[dict]
    logits: dict[tuple[int, str], torch.Tensor] | None
    prompt_tokens: list[str]

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

    def first_correct_layer(self, position: int, target_token: str) -> int | None:
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
        data = torch.load(path, weights_only=False)
        states = {}
        for k, v in data["states"].items():
            parts = k.split("_", 1)
            states[(int(parts[0]), parts[1])] = v
        return HiddenStates(states=states, prompt_tokens=data["prompt_tokens"])


@dataclass
class Intervention:
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

