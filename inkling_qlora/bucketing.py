"""Length-bucketed sampler: low padding without coupling updates to length."""

from __future__ import annotations

import random
from collections.abc import Iterator
from collections.abc import Sequence

from torch.utils.data import Sampler


class BucketedSampler(Sampler[int]):
    """Yield indices so that consecutive batches have similar lengths.

    Each epoch: shuffle all indices, cut into windows of ``batch_size * window_batches``, sort each window by
    length, cut into batches, then shuffle the full batches across windows. The one possibly short batch goes
    last so a ``DataLoader`` with the same ``batch_size`` and ``drop_last=False`` reproduces the batches.
    The order depends on ``seed`` and ``self.epoch``. ``__iter__`` does NOT advance the epoch, so repeated passes
    replay the same order; the trainer (or accelerate) must call ``set_epoch(epoch)``, which also makes a resume
    replay the right order. The short final batch, if any, is the longest examples of the last window, placed last.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, window_batches: int = 32, seed: int = 0):
        if batch_size < 1 or window_batches < 1:
            raise ValueError("batch_size and window_batches must be at least 1")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.window_batches = window_batches
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lengths)

    def batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed * 1_000_003 + epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        if self.batch_size == 1:
            return [[i] for i in order]
        win = self.batch_size * self.window_batches
        full: list[list[int]] = []
        short: list[int] = []
        for start in range(0, len(order), win):
            window = sorted(order[start : start + win], key=lambda i: (self.lengths[i], i))
            for b in range(0, len(window), self.batch_size):
                batch = window[b : b + self.batch_size]
                if len(batch) == self.batch_size:
                    full.append(batch)
                else:
                    short = batch
        rng.shuffle(full)
        return full + ([short] if short else [])

    def __iter__(self) -> Iterator[int]:
        for batch in self.batches(self.epoch):
            yield from batch


def padding_ratio(lengths: Sequence[int], order: Sequence[int], batch_size: int) -> float:
    """Padded tokens over real tokens when ``order`` is cut into batches of ``batch_size``."""
    padded = real = 0
    for s in range(0, len(order), batch_size):
        ls = [lengths[i] for i in order[s : s + batch_size]]
        padded += max(ls) * len(ls)
        real += sum(ls)
    return padded / real if real else 1.0
