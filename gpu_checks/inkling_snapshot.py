"""Kept for the GPU scripts that call it; nothing to do any more.

On 2026-11-22 the local Hugging Face cache's ``refs/main`` for Inkling-2-Mini briefly pointed at a snapshot without
the weights. ``inkling.load_inkling_4bit`` and ``inkling._snapshot`` now pin ``inkling.INKLING_REVISION``
(overridable with the ``INKLING_REVISION`` environment variable), so ``refs/main`` no longer matters.
"""

from __future__ import annotations


def pin_full_snapshot(model_id: str = "thinkingmachines/Inkling-2-Mini") -> None:
    """No-op: the loader pins the revision itself (see the module docstring)."""
