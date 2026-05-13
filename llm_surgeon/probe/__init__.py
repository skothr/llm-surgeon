"""Hidden state probing, logit lens, intervention, and NLA verbalization.

Re-exports the public API and the private helpers that tests / GUI
consume, so ``from llm_surgeon.probe import X`` keeps working
unchanged across the package split.
"""

from __future__ import annotations

from llm_surgeon.probe._hooks import (
    _attach_reader_grad_hooks,
    _get_input_device,
    _make_capture_input_hook,
    _make_capture_output_hook,
    _unwrap_hook_output,
)
from llm_surgeon.probe._types import (
    CompareLogitLensResult,
    HiddenStates,
    Intervention,
    InterventionResult,
    LogitLensResult,
    PatchingResult,
)
from llm_surgeon.probe._capture import (
    _capture_residual_stream,
    _capture_residual_stream_with_grad,
)
from llm_surgeon.probe._logit_lens import (
    _cell_metrics,
    _pair_metrics,
    _project_to_logits,
    compare_logit_lens,
    extract_hidden_states,
    layer_predictions_table,
    logit_lens,
)
from llm_surgeon.probe._intervention import (
    _Op,
    _Ops,
    _make_position_patch,
    activation_patch,
    intervene,
    ops,
)
from llm_surgeon.probe._attribution import (
    _compute_all_edges,
    _integrated_gradients_loop,
    _is_valid_attn_writer,
    _is_valid_ffn_writer,
    attribution_patch,
    attribution_patch_per_head,
    attribution_patch_per_neuron,
    edge_attribution_patch,
    extract_circuit,
)
from llm_surgeon.probe._nla import (
    AR_ID as AR_ID,
    AV_ID as AV_ID,
    load_ar as load_ar,
    load_ar_meta as load_ar_meta,
    load_av as load_av,
    load_av_meta as load_av_meta,
    nla_reconstruct as nla_reconstruct,
    nla_score as nla_score,
    nla_verbalize as nla_verbalize,
)

__all__ = [
    "AV_ID",
    "CompareLogitLensResult",
    "HiddenStates",
    "Intervention",
    "InterventionResult",
    "LogitLensResult",
    "PatchingResult",
    "activation_patch",
    "attribution_patch",
    "attribution_patch_per_head",
    "attribution_patch_per_neuron",
    "compare_logit_lens",
    "edge_attribution_patch",
    "extract_circuit",
    "extract_hidden_states",
    "intervene",
    "layer_predictions_table",
    "load_av",
    "load_av_meta",
    "logit_lens",
    "nla_verbalize",
    "ops",
]
