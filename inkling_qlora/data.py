"""Training data: JSONL in TRL's conversational prompt/completion format.

One object per line::

    {"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "completion": [{"role": "assistant", "content": "..."}]}

The loss is computed on the completion only. Inkling's chat template is applied by TRL, with
``reasoning_effort="none"`` (see :data:`inkling_qlora.inkling.INKLING_CHAT_TEMPLATE_KWARGS`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """All records of a JSONL file, checked for the prompt/completion message lists."""
    return list(iter_records(path))


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("prompt", "completion"):
                if not isinstance(row.get(key), list):
                    raise ValueError(f"{path}:{n}: '{key}' must be a list of chat messages")
            yield row
