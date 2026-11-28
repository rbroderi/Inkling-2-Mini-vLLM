"""Re-store Inkling-2-Mini's MXFP8 expert layers (20-23) as NVFP4 in a CONVERTED COPY of the checkpoint, for the vLLM
Ampere fork to serve (docs/inkling_memory_review.md item 7, section 9). CPU only. The original snapshot is only read.

Run in WSL with the vLLM venv (it has torch and safetensors):
  ~/vllm-venv/bin/python scripts/convert_layers_nvfp4.py --dry-run-layer 20 [--max-experts 8]
  ~/vllm-venv/bin/python scripts/convert_layers_nvfp4.py            # full copy, about 16.4 GiB
Then: INKLING_MODEL_DIR=$HOME/inkling_nvfp4_all scripts/serve_long.sh

What changes (the fork picks a layer's expert format from hf_quant_config.json, `is_mxfp8_layer`, and the model
declares 4 extra NVFP4 tensors per layer):
  per converted layer, w13 and w2:
    {p}_weight                          F8_E4M3 (128, N, K)    -> U8      (128, N, K/2)   two E2M1 codes per byte
    {p}_weight_scale                    U8      (128, N, K/32) -> F8_E4M3 (128, N, K/16)  block scale per 16 columns
    {p}_weight_amax                     new, F32 (128,)        per-expert amax
    {p}_weight_outer_scale_denominator  new, F32 (1,) = 2688  (6 x 448; same value as the stock NVFP4 layers)
  The fork's per-expert global scale is amax / denominator; quantize_nvfp4's factor is amax / 2688, so amax = factor x 2688.
  hf_quant_config.json: the 8 weight names move from the mxfp8 parameter group to the nvfp4 group (the mxfp8 group is
  dropped when empty). model.safetensors.index.json gets the 16 new keys and the new total_size.
Rows stay in checkpoint order (gate/up interleaved) as in the stock NVFP4 layers; the fork's loader does not reorder.
The checkpoint is ONE shard, so it is rewritten whole (unchanged tensors copied byte for byte); the other files are
symlinked to the original.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import torch

DENOMINATOR = 2688.0  # 6 x 448, read from the stock NVFP4 layers (outer_scale_denominator)
SHARD = "model-00001-of-00001.safetensors"
DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "F32": torch.float32}
DT_NAME = {torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8", torch.float32: "F32"}


def load_codec():
    path = Path(__file__).resolve().parents[1] / "inkling_qlora" / "inkling_nvfp4.py"
    spec = importlib.util.spec_from_file_location("inkling_nvfp4", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["inkling_nvfp4"] = module  # dataclasses / typing need it registered
    spec.loader.exec_module(module)
    return module


def find_snapshot(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    cache = Path.home() / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots"
    for s in sorted(cache.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (s / SHARD).exists():
            return s
    raise SystemExit(f"no snapshot with {SHARD} under {cache}")


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def convert_layer(codec, raw: dict[str, torch.Tensor], max_experts: int | None = None) -> dict[str, torch.Tensor]:
    """MXFP8 tensors of one layer (keys 'w13_weight', 'w13_weight_scale', 'w2_...') -> the 8 NVFP4 tensors."""
    out: dict[str, torch.Tensor] = {}
    for proj in ("w13", "w2"):
        values, scales = raw[f"{proj}_weight"], raw[f"{proj}_weight_scale"]
        n = values.shape[0] if max_experts is None else min(max_experts, values.shape[0])
        codes, blocks, factors = [], [], []
        for e in range(n):
            w = codec.dequantize_mxfp8(values[e], scales[e])
            c, s, f = codec.quantize_nvfp4(w)
            codes.append(c)
            blocks.append(s)
            factors.append(f)
        out[f"{proj}_weight"] = torch.stack(codes)
        out[f"{proj}_weight_scale"] = torch.stack(blocks)
        out[f"{proj}_weight_amax"] = torch.stack(factors).float() * DENOMINATOR
        out[f"{proj}_weight_outer_scale_denominator"] = torch.full((1,), DENOMINATOR, dtype=torch.float32)
    return out


def dry_run(codec, snapshot: Path, layer: int, max_experts: int | None) -> None:
    from safetensors import safe_open

    prefix = "model.llm.layers.{}.mlp.experts."
    with safe_open(snapshot / SHARD, "pt") as fh:
        ref = {
            k.removeprefix(prefix.format(0)): (tuple(fh.get_slice(k).get_shape()), fh.get_slice(k).get_dtype())
            for k in fh.keys()
            if k.startswith(prefix.format(0))
        }
        raw = {
            k.removeprefix(prefix.format(layer)): fh.get_tensor(k)
            for k in fh.keys()
            if k.startswith(prefix.format(layer))
        }
    assert not any("amax" in k for k in raw), f"layer {layer} is already NVFP4"
    print(f"layer {layer}: source tensors")
    for k, v in sorted(raw.items()):
        print(f"  {k:36s} {v.dtype!s:22s} {tuple(v.shape)}")
    t0 = time.time()
    new = convert_layer(codec, raw, max_experts)
    print(
        f"converted {max_experts or 128} experts per projection on the CPU in {time.time() - t0:.0f} s; compare with layer 0:"
    )
    bad = 0
    for k, v in sorted(new.items()):
        shape, dtype = ref[k]
        want = list(shape)
        if max_experts and want[0] == 128 and "denominator" not in k:
            want[0] = max_experts
        ok = list(v.shape) == want and DT_NAME[v.dtype] == dtype
        bad += not ok
        print(
            f"  {k:36s} {DT_NAME[v.dtype]:8s} {tuple(v.shape)!s:20s} layer 0: {dtype} {shape}  {'OK' if ok else 'MISMATCH'}"
        )
    print("round trip (relative L2 error of NVFP4 vs the MXFP8 source), first 3 experts:")
    for proj in ("w13", "w2"):
        errs = []
        for e in range(min(3, new[f"{proj}_weight"].shape[0])):
            src = codec.dequantize_mxfp8(raw[f"{proj}_weight"][e], raw[f"{proj}_weight_scale"][e])
            codes, bs = new[f"{proj}_weight"][e], new[f"{proj}_weight_scale"][e]
            factor = new[f"{proj}_weight_amax"][e] / new[f"{proj}_weight_outer_scale_denominator"][0]
            lo, hi = (codes & 0xF).long(), (codes >> 4).long()
            table = torch.tensor(codec.E2M1_VALUES)
            nib = torch.stack([table[lo], table[hi]], -1).reshape(codes.shape[0], -1)
            dec = nib.view(nib.shape[0], -1, 16) * (bs.float() * factor)[..., None]
            dec = dec.reshape(nib.shape)
            errs.append(((dec - src).norm() / src.norm()).item())
        print(f"  {proj}: " + ", ".join(f"{e:.3f}" for e in errs))
    print("RESULT:", "names, shapes and dtypes match the stock NVFP4 layers" if not bad else f"{bad} MISMATCHES")


def convert(codec, snapshot: Path, out: Path, layers: list[int]) -> None:
    from safetensors import safe_open

    src_path = snapshot / SHARD
    header, base = read_header(src_path)
    meta = header.pop("__metadata__", {"format": "pt"})
    prefix = "model.llm.layers.{}.mlp.experts."
    drop = {
        prefix.format(l) + s for l in layers for s in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }
    missing = drop - header.keys()
    assert not missing, missing
    # new tensors' shapes are known from the source shapes without converting
    new_meta: dict[str, tuple[str, list[int]]] = {}
    for l in layers:
        for proj in ("w13", "w2"):
            n, rows, k = header[prefix.format(l) + f"{proj}_weight"]["shape"]
            new_meta[prefix.format(l) + f"{proj}_weight"] = ("U8", [n, rows, k // 2])
            new_meta[prefix.format(l) + f"{proj}_weight_scale"] = ("F8_E4M3", [n, rows, k // 16])
            new_meta[prefix.format(l) + f"{proj}_weight_amax"] = ("F32", [n])
            new_meta[prefix.format(l) + f"{proj}_weight_outer_scale_denominator"] = ("F32", [1])
    sizes = {n: (d, s) for n, (d, s) in new_meta.items()}
    unit = {"U8": 1, "F8_E4M3": 1, "F32": 4}
    order = [
        n
        for n, _ in sorted(((n, v["data_offsets"][0]) for n, v in header.items() if n not in drop), key=lambda t: t[1])
    ]
    order += sorted(new_meta)
    new_header, offset = {"__metadata__": meta}, 0
    for n in order:
        if n in new_meta:
            dt, shape = new_meta[n]
            size = unit[dt] * int(torch.tensor(shape).prod())
        else:
            dt, shape = header[n]["dtype"], header[n]["shape"]
            size = header[n]["data_offsets"][1] - header[n]["data_offsets"][0]
        new_header[n] = {"dtype": dt, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    hb = json.dumps(new_header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    free = shutil.disk_usage(out.parent).free
    assert free > offset + 2 * 2**30, f"need {offset / 2**30:.1f} GiB, {free / 2**30:.1f} free"

    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (SHARD + ".tmp")
    t0 = time.time()
    with open(src_path, "rb") as src, open(tmp, "wb") as dst, safe_open(src_path, "pt") as fh:
        dst.write(struct.pack("<Q", len(hb)))
        dst.write(hb)
        pending: dict[str, torch.Tensor] = {}
        for n in order:
            if n in new_meta:
                l = int(n.split(".layers.")[1].split(".")[0])
                if l not in pending:
                    raw = {
                        k: fh.get_tensor(prefix.format(l) + k)
                        for k in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
                    }
                    pending = {l: convert_layer(codec, raw)}
                    print(f"layer {l} converted ({time.time() - t0:.0f} s)", flush=True)
                t = pending[l][n.rsplit(".", 1)[1]].contiguous()
                dt, shape = new_meta[n]
                assert DT_NAME[t.dtype] == dt and list(t.shape) == shape, (n, t.dtype, t.shape)
                dst.write(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8 else t.numpy().tobytes())
            else:
                a, b = header[n]["data_offsets"]
                src.seek(base + a)
                left = b - a
                while left:
                    chunk = src.read(min(left, 1 << 26))
                    dst.write(chunk)
                    left -= len(chunk)
    assert tmp.stat().st_size == 8 + len(hb) + offset, (tmp.stat().st_size, 8 + len(hb) + offset)
    tmp.rename(out / SHARD)

    # hf_quant_config.json and the index
    q = json.loads((snapshot / "hf_quant_config.json").read_text())
    groups = q["model"]["parameter_groups"]
    moved = {prefix.format(l) + f"{p}_weight" for l in layers for p in ("w13", "w2")}
    mx = next(g for g in groups if g["compute_dtype"] == "mxfp8")
    nv = next(g for g in groups if g["compute_dtype"] == "nvfp4")
    assert moved <= set(mx["tensors"]), "layers to convert are not all MXFP8 in hf_quant_config.json"
    mx["tensors"] = sorted(set(mx["tensors"]) - moved)
    nv["tensors"] = sorted(set(nv["tensors"]) | moved)
    q["model"]["parameter_groups"] = [g for g in groups if g["tensors"]]
    (out / "hf_quant_config.json").write_text(json.dumps(q, indent=2))
    idx = json.loads((snapshot / "model.safetensors.index.json").read_text())
    for n in drop:
        idx["weight_map"].pop(n)
    for n in new_meta:
        idx["weight_map"][n] = SHARD
    idx["metadata"]["total_size"] = offset
    (out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
    for p in snapshot.iterdir():  # everything else is shared with the original through symlinks
        if (
            p.name not in (SHARD, "hf_quant_config.json", "model.safetensors.index.json")
            and not (out / p.name).exists()
        ):
            (out / p.name).symlink_to(p.resolve())
    print(f"done: {out} ({(8 + len(hb) + offset) / 2**30:.2f} GiB shard, {time.time() - t0:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", help="original snapshot dir (default: newest with the weights in the HF cache)")
    ap.add_argument("--out", default=str(Path.home() / "inkling_nvfp4_all"))
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 21, 22, 23])
    ap.add_argument(
        "--dry-run-layer", type=int, help="convert this one layer in memory, check names/shapes, write nothing"
    )
    ap.add_argument("--max-experts", type=int, help="dry run only: convert just the first N experts")
    a = ap.parse_args()
    codec = load_codec()
    snapshot = find_snapshot(a.snapshot)
    out = Path(a.out)
    if a.dry_run_layer is not None:
        dry_run(codec, snapshot, a.dry_run_layer, a.max_experts)
        return
    if out.resolve() == snapshot.resolve() or (out / SHARD).exists():
        raise SystemExit(f"{out} already holds a converted copy or is the original; refusing")
    convert(codec, snapshot, out, a.layers)


if __name__ == "__main__":
    main()
