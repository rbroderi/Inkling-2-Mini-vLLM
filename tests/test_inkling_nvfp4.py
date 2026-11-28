"""CPU tests for the pure-torch NVFP4/MXFP8 codec."""

import os
from pathlib import Path

import pytest
import torch

from inkling_qlora.inkling_nvfp4 import E2M1_VALUES
from inkling_qlora.inkling_nvfp4 import _round_to_e4m3
from inkling_qlora.inkling_nvfp4 import dequantize_mxfp8
from inkling_qlora.inkling_nvfp4 import dequantize_nvfp4
from inkling_qlora.inkling_nvfp4 import e4m3_table
from inkling_qlora.inkling_nvfp4 import quantize_nvfp4

# Literal copy of the NVFP4 branch of Quant4bitExperts.weight (inkling.py), without the de-interleave and bf16 cast.
_REF_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _reference_nvfp4(codes, scale, factor):
    byte_to_values = torch.stack(
        (torch.tensor(_REF_VALUES)[torch.arange(256) & 0xF], torch.tensor(_REF_VALUES)[torch.arange(256) >> 4]), -1
    )
    out = []
    for e in range(codes.shape[0]):
        s = scale[e].float()
        rows = codes[e].shape[0]
        w = byte_to_values[codes[e].long()].view(*s.shape, 16) * s[..., None]
        out.append((w * factor[e]).reshape(rows, -1))
    return torch.stack(out)


def _random_nvfp4(experts=3, rows=8, columns=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (experts, rows, columns // 2), generator=g, dtype=torch.uint8)
    # finite positive e4m3 bytes only (0x08..0x7E), so scales are ordinary numbers
    scale = torch.randint(0x08, 0x7F, (experts, rows, columns // 16), generator=g, dtype=torch.uint8)
    factor = torch.rand(experts, generator=g) * 0.01 + 0.001
    return codes, scale, factor


def test_e4m3_table_finite():
    """Check e4m3 table finite."""
    t = e4m3_table()
    assert t.shape == (256,)
    assert t.dtype == torch.float32
    assert torch.isfinite(t).all()
    assert t[0x7E] == 448.0
    assert t[0x7F] == 0.0
    assert t[0xFF] == 0.0
    assert t[0x01] == 2.0**-9


def test_dequantize_matches_reference_exactly():
    """Check dequantize matches reference exactly."""
    codes, scale, factor = _random_nvfp4()
    ref = _reference_nvfp4(codes, scale.view(torch.float8_e4m3fn), factor)
    assert torch.equal(dequantize_nvfp4(codes, scale, factor), ref)
    assert torch.equal(dequantize_nvfp4(codes, scale.view(torch.float8_e4m3fn), factor), ref)
    assert ref.shape == (3, 8, 64)


def test_float64_and_scalar_factor():
    """Check float64 and scalar factor."""
    codes, scale, factor = _random_nvfp4()
    d64 = dequantize_nvfp4(codes, scale, factor, dtype=torch.float64)
    assert d64.dtype == torch.float64
    assert torch.allclose(d64.float(), dequantize_nvfp4(codes, scale, factor), rtol=1e-6)
    single = dequantize_nvfp4(codes[0], scale[0], factor[0])
    assert torch.equal(single, dequantize_nvfp4(codes, scale, factor)[0])


def test_nibble_order_and_values():
    """Check nibble order and values."""
    for nibble in range(16):
        # low nibble = even column, high nibble = odd column
        byte = torch.tensor([[nibble | (((nibble + 3) % 16) << 4)] * 8], dtype=torch.uint8)  # (1, 8): 16 columns
        scale = torch.tensor([[0x38]], dtype=torch.uint8)  # e4m3 1.0
        out = dequantize_nvfp4(byte, scale, torch.tensor(1.0))
        assert out[0, 0].item() == E2M1_VALUES[nibble]
        assert out[0, 1].item() == E2M1_VALUES[(nibble + 3) % 16]
    # negative zero decodes to zero
    assert (
        dequantize_nvfp4(
            torch.full((1, 8), 0x88, dtype=torch.uint8), torch.tensor([[0x38]], dtype=torch.uint8), torch.tensor(1.0)
        )
        .eq(0)
        .all()
    )


def _own_rule_matrix(seed=1):
    """Codes/scales whose every block holds a magnitude-6 code and one block at scale 448, with factor 2**-10."""
    g = torch.Generator().manual_seed(seed)
    rows, columns = 16, 64
    nibbles = torch.randint(0, 16, (rows, columns), generator=g, dtype=torch.uint8)
    nibbles.view(rows, columns // 16, 16)[..., 0] = torch.where(torch.rand(rows, 4, generator=g) < 0.5, 7, 15).to(
        torch.uint8
    )
    codes = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    scale = torch.randint(0x28, 0x60, (rows, columns // 16), generator=g, dtype=torch.uint8)
    scale[3, 2] = 0x7E  # 448
    return codes, scale, torch.tensor(2.0**-10)


def test_round_trip_exact_with_own_factor_rule():
    """Check round trip exact with own factor rule."""
    codes, scale, factor = _own_rule_matrix()
    w = dequantize_nvfp4(codes, scale, factor)
    assert w.abs().max() == 6 * 448 * factor  # so quantize_nvfp4 re-derives the same factor
    q_codes, q_scale, q_factor = quantize_nvfp4(w)
    assert q_factor == factor
    assert torch.equal(dequantize_nvfp4(q_codes, q_scale, q_factor), w)


def test_quantize_gaussian_error_and_scale_bytes():
    """Check quantize gaussian error and scale bytes."""
    w = torch.randn(256, 256, generator=torch.Generator().manual_seed(2))
    codes, scale, factor = quantize_nvfp4(w)
    assert codes.dtype == torch.uint8
    assert codes.shape == (256, 128)
    assert scale.dtype == torch.float8_e4m3fn
    assert scale.shape == (256, 16)
    assert factor.dtype == torch.float32
    assert factor.ndim == 0
    raw = scale.view(torch.uint8)
    assert ((raw >= 0x01) & (raw <= 0x7E)).all()  # finite, positive, never NaN or negative
    rel = (dequantize_nvfp4(codes, scale, factor) - w).norm() / w.norm()
    assert rel < 0.12


def test_zero_matrix_and_zero_block():
    """Check zero matrix and zero block."""
    codes, scale, factor = quantize_nvfp4(torch.zeros(4, 32))
    assert factor == 1.0
    assert (codes == 0).all()
    assert (scale.view(torch.uint8) == 0x01).all()
    assert dequantize_nvfp4(codes, scale, factor).eq(0).all()

    w = torch.randn(4, 64, generator=torch.Generator().manual_seed(3))
    w[1, 16:32] = 0.0
    codes, scale, factor = quantize_nvfp4(w)
    out = dequantize_nvfp4(codes, scale, factor)
    assert torch.isfinite(out).all()
    assert out[1, 16:32].eq(0).all()
    assert (codes[1, 8:16] == 0).all()
    assert scale.view(torch.uint8)[1, 1] == 0x01


def test_dequantize_mxfp8_matches_pr_formula():
    """Check dequantize mxfp8 matches pr formula."""
    g = torch.Generator().manual_seed(4)
    experts, rows, columns = 2, 4, 64
    values = (torch.randn(experts, rows, columns, generator=g) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 140, (experts, rows, columns // 32), generator=g, dtype=torch.uint8)
    # literal from Inkling2LoopDequantizeExperts.convert (amax is None branch), before the dtype cast
    block_scale = scale.float()
    ref = values.float().view(*block_scale.shape, 32)
    ref = (ref * torch.exp2(block_scale - 127)[..., None]).reshape(-1, rows, columns)
    out = dequantize_mxfp8(values, scale)
    assert out.dtype == torch.float32
    assert torch.equal(out, ref)


@pytest.mark.skipif(not os.environ.get("INKLING_SNAPSHOT"), reason="set INKLING_SNAPSHOT to a checkpoint snapshot dir")
def test_real_checkpoint_layer0_expert0_matches_reference():
    """Check real checkpoint layer0 expert0 matches reference."""
    import json  # noqa: PLC0415  # lazy import in a test

    from safetensors import safe_open  # noqa: PLC0415  # lazy import in a test

    snap = Path(os.environ["INKLING_SNAPSHOT"])
    weight_map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.llm.layers.0.mlp.experts.w2_weight"
    got = {}
    for suffix in ("", "_scale", "_amax", "_outer_scale_denominator"):
        with safe_open(snap / weight_map[prefix + suffix], "pt", device="cpu") as fh:
            got[suffix] = fh.get_tensor(prefix + suffix)
    factor = got["_amax"].float() / got["_outer_scale_denominator"].float()
    codes, scale = got[""][:1], got["_scale"][:1]
    assert torch.equal(dequantize_nvfp4(codes, scale, factor[:1]), _reference_nvfp4(codes, scale, factor[:1]))


def test_non_uint8_integer_scale_rejected():
    """Check non uint8 integer scale rejected."""
    codes = torch.zeros(1, 8, dtype=torch.uint8)
    for dtype in (torch.int64, torch.int32, torch.int8):
        with pytest.raises(TypeError):
            dequantize_nvfp4(codes, torch.tensor([[0x38]], dtype=dtype), torch.tensor(1.0))


def _anchored(row_values):
    """Matrix whose row 0 sets factor to exactly 2**-10 (a 2688 * f entry); row 1 is ``row_values``."""
    f = 2.0**-10
    w = torch.zeros(2, 16)
    w[0, 0] = 6 * 448 * f
    w[1] = torch.tensor(row_values) * f
    return w, f


def test_second_candidate_scale_chosen_when_better():
    """Check second candidate scale chosen when better."""
    w, f = _anchored([4.0] * 16)  # /4 gives scale 1.0 and exact codes; /6 gives scale ~0.6875 and a lossy 4.125
    codes, scale, factor = quantize_nvfp4(w)
    assert factor == f
    assert scale.view(torch.uint8)[1, 0] == 0x38
    assert torch.equal(dequantize_nvfp4(codes, scale, factor)[1], w[1])
    assert scale.view(torch.uint8)[0, 0] == 0x7E  # row 0 is exact with the /6 candidate (448)


def test_tie_rules():
    # a value exactly on the midpoint between magnitudes 0 and 0.5 goes to the smaller magnitude (nibble 0)
    """Check tie rules."""
    w, _ = _anchored([6.0, 0.25] + [0.0] * 14)
    codes, scale, _ = quantize_nvfp4(w)
    assert scale.view(torch.uint8)[1, 0] == 0x38  # equal-error candidates: /6 wins
    assert codes[1, 0] == 0x07  # 6 -> nibble 7, 0.25 -> nibble 0
    # a scale exactly between two e4m3 codes goes to the lower code
    positive = e4m3_table()[1:0x7F]
    x = torch.tensor([1.0625, 1.1875, 448.0, 1e9, 1e-9])  # mid(1, 1.125), mid(1.125, 1.25), clamps
    assert _round_to_e4m3(x, positive, 1).tolist() == [0x38, 0x39, 0x7E, 0x7E, 0x01]
