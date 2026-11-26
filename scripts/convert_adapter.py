"""Convert a PEFT Inkling adapter directory to vLLM's checkpoint names: convert_adapter.py SRC DST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkling_qlora.inkling_vllm_adapter import convert_adapter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    print(json.dumps(convert_adapter(args.src, args.dst), indent=2))


if __name__ == "__main__":
    main()
