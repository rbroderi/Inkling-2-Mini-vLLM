"""Pure-torch NVFP4 and MXFP8 codec for Inkling 2 Mini's expert weights (CPU or CUDA, imports only torch).

Layouts follow the checkpoint (see ``Quant4bitExperts`` in ``inkling.py`` and transformers'
``Inkling2LoopDequantizeExperts``):

- NVFP4: two E2M1 codes per byte (even column in the low nibble), one float8_e4m3 scale per 16 columns, and one float32
  factor per expert (``amax / outer_scale_denominator``). Rows are in checkpoint order; nothing here de-interleaves.
- MXFP8: float8_e4m3 values with one power-of-two scale ``2 ** (byte - 127)`` per 32 columns.
"""

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # boundaries between neighbouring magnitudes
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32


def e4m3_table(device: torch.device | str | None = None) -> torch.Tensor:
    """float32 (256,) value of every e4m3fn byte, with the NaN codes set to 0."""
    table = torch.arange(256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).float()
    return torch.where(torch.isfinite(table), table, torch.zeros_like(table))


def _scale_as_float(scale: torch.Tensor) -> torch.Tensor:
    """Block scales as float32; uint8 is read as e4m3 bytes (a reinterpretation, never a numeric cast)."""
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e4m3fn)
    elif scale.dtype != torch.float8_e4m3fn:
        msg = f"scale must be uint8 (e4m3 bytes) or float8_e4m3fn, got {scale.dtype}"
        raise TypeError(msg)
    return scale.float()


def dequantize_nvfp4(
    codes: torch.Tensor, scale: torch.Tensor, factor: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Decode NVFP4 ``codes`` (..., R, C/2) uint8 with ``scale`` (..., R, C/16) and ``factor`` into (..., R, C).

    Computed as ``(table[codes] * scale[..., None]) * factor`` in float32 (float64 when ``dtype`` is float64), the same
    order as the reference, so the result equals it before the reference's cast to bf16.
    """
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    device = codes.device
    values = torch.tensor(E2M1_VALUES, dtype=compute, device=device)
    byte = torch.arange(256, device=device)
    byte_to_values = torch.stack((values[byte & 0xF], values[byte >> 4]), dim=-1)  # (256, 2): low then high nibble
    block_scale = _scale_as_float(scale).to(compute)
    w = byte_to_values[codes.long()].view(*block_scale.shape, NVFP4_BLOCK) * block_scale[..., None]
    f = factor.to(device=device, dtype=compute)
    w = w * f.reshape(*f.shape, 1, 1, 1)  # factor covers the leading dimensions; add rows, blocks, in-block axes
    return w.reshape(*codes.shape[:-1], codes.shape[-1] * 2).to(dtype)


def _round_to_e4m3(x: torch.Tensor, positive: torch.Tensor, first_code: int) -> torch.Tensor:
    """Nearest finite positive e4m3 byte for each x (ties to the lower code, clamped to the table's range)."""
    n = positive.numel()
    i = torch.bucketize(x, positive)  # positive[i - 1] < x <= positive[i]
    lo = (i - 1).clamp(0, n - 1)
    hi = i.clamp(0, n - 1)
    pick_lo = (x - positive[lo]) <= (positive[hi] - x)
    index = torch.where(pick_lo, lo, hi)
    return (index + first_code).to(torch.uint8)


def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one matrix ``w`` (R, C), C divisible by 16, to ``(codes, scale, factor)``.

    ``codes`` uint8 (R, C/2), ``scale`` float8_e4m3fn (R, C/16), ``factor`` float32 0-dim. Each 16-column block tries
    the scales ``blockmax / 6`` and ``blockmax / 4`` (relative to ``factor``) and keeps the one with the smaller squared
    reconstruction error.
    """
    rows, columns = w.shape
    if columns % NVFP4_BLOCK:
        msg = f"columns ({columns}) must be divisible by {NVFP4_BLOCK}"
        raise ValueError(msg)
    device = w.device
    w = w.float()
    amax = w.abs().max()
    factor = torch.where(amax > 0, amax / (6.0 * 448.0), torch.ones_like(amax)).float()

    table = e4m3_table(device)
    # positive finite e4m3 values are bytes 0x01..0x7E, ascending
    first_code = 1
    positive = table[first_code:0x7F]
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    midpoints = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=device)

    blocks = w.view(rows, columns // NVFP4_BLOCK, NVFP4_BLOCK)
    blockmax = blocks.abs().amax(-1)
    best_err = best_scale = best_nibbles = None
    for divisor in (6.0, 4.0):
        scale_byte = _round_to_e4m3(blockmax / divisor / factor, positive, first_code)  # (R, B)
        scale_value = table[scale_byte.long()]
        unit = (scale_value * factor)[..., None]
        magnitude_index = torch.bucketize(blocks.abs() / unit, midpoints)  # ties go to the smaller magnitude
        decoded = magnitudes[magnitude_index] * torch.where(blocks < 0, -1.0, 1.0) * unit
        err = ((decoded - blocks) ** 2).sum(-1)
        sign = ((blocks < 0) & (magnitude_index > 0)).to(torch.uint8) * 8  # a zero magnitude is always nibble 0
        nibbles = magnitude_index.to(torch.uint8) | sign
        if best_err is None or best_scale is None or best_nibbles is None:
            best_err, best_scale, best_nibbles = err, scale_byte, nibbles
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale_byte, best_scale)
            best_nibbles = torch.where(better[..., None], nibbles, best_nibbles)
    if best_scale is None or best_nibbles is None:
        msg = "no scale candidates"
        raise RuntimeError(msg)
    nibbles = best_nibbles.reshape(rows, columns)
    codes = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return codes.contiguous(), best_scale.contiguous().view(torch.float8_e4m3fn), factor


def dequantize_mxfp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP8: ``values`` float8_e4m3fn (..., R, C), ``scale`` uint8 (..., R, C/32) -> float32 (..., R, C)."""
    block_scale = scale.float()
    w = values.float().view(*block_scale.shape, MXFP8_BLOCK) * torch.exp2(block_scale - 127)[..., None]
    return w.reshape(values.shape)
