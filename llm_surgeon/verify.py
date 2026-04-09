"""Structural verification of modified models."""

from dataclasses import dataclass, field
from typing import List, Optional


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
