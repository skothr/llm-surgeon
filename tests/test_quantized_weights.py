"""Weight-level surgery and weight inspection on bitsandbytes-quantized layers.

bitsandbytes 0.45+ quantizes on CPU, so these run without a GPU: each
nn.Linear in the chosen layers is swapped for a Linear4bit / Linear8bitLt
and moved to CPU, which is the state an ``nf4``/``int8`` load leaves it in.
"""

import copy

import pytest
import torch
import torch.nn as nn

from llm_surgeon.inspect import weight_norms, weight_svd
from llm_surgeon.surgery import (
    scale_heads,
    swap_heads,
    zero_attention,
    zero_heads,
    zero_mlp,
)

bnb = pytest.importorskip("bitsandbytes")


def _quantize_layers(model, bits: int, layers=(0,)):
    """Replace every nn.Linear in the given decoder layers with a bnb layer on CPU."""
    for li in layers:
        layer = model.model.layers[li]
        for name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            parent_name, attr = name.rsplit(".", 1)
            parent = layer.get_submodule(parent_name)
            w = module.weight.data.clone()
            if bits == 4:
                new = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=False,
                    compute_dtype=torch.float32,
                    quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
            else:
                new = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=False,
                    has_fp16_weights=False,
                )
                new.weight = bnb.nn.Int8Params(w, requires_grad=False)
            setattr(parent, attr, new.to("cpu"))
    return model


@pytest.fixture(params=[4, 8], ids=["nf4", "int8"])
def quantized_pair(request, tiny_llama):
    """(dense reference, same model with layer 0 quantized)."""
    dense = copy.deepcopy(tiny_llama)
    return dense, _quantize_layers(tiny_llama, request.param)


class TestSurgeryRefusesQuantizedWeights:
    @pytest.mark.parametrize(
        "op",
        [
            lambda m: zero_heads(m, 0, [1]),
            lambda m: scale_heads(m, 0, [1], 0.5),
            lambda m: swap_heads(m, 0, 0, 1),
            lambda m: zero_mlp(m, 0),
            lambda m: zero_attention(m, 0),
        ],
        ids=["zero_heads", "scale_heads", "swap_heads", "zero_mlp", "zero_attention"],
    )
    def test_raises_type_error(self, quantized_pair, op):
        _, model = quantized_pair
        state_before = {
            n: p.clone() for n, p in model.model.layers[0].named_parameters()
        }
        with pytest.raises(TypeError, match="quantized weight"):
            op(model)
        for n, p in model.model.layers[0].named_parameters():
            assert torch.equal(p, state_before[n]), f"{n} was modified"

    def test_dense_layers_still_editable(self, quantized_pair):
        _, model = quantized_pair
        zero_mlp(model, 1)  # layer 1 is dense
        assert torch.all(model.model.layers[1].mlp.down_proj.weight == 0)


class TestInspectDequantizes:
    def test_weight_norms_match_dense(self, quantized_pair):
        dense, model = quantized_pair
        ref = weight_norms(dense)[0]
        got = weight_norms(model)[0]
        for key in ("attn_norm", "mlp_norm", "total_norm"):
            assert got[key] == pytest.approx(ref[key], rel=0.1), key

    def test_weight_svd_shapes_and_values_match_dense(self, quantized_pair):
        dense, model = quantized_pair
        ref = weight_svd(dense, layers=[0])[0]
        got = weight_svd(model, layers=[0])[0]
        for proj, sv in ref.items():
            assert got[proj].shape == sv.shape, proj
            assert torch.allclose(got[proj][0], sv[0], rtol=0.1), proj
