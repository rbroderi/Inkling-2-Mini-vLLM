import random
import statistics

import numpy as np
import pytest
from torch.utils.data import DataLoader

from inkling_qlora.bucketing import BucketedSampler, padding_ratio


def lognormal(n=4000, seed=1):
    rng = random.Random(seed)
    sigma = 0.5
    mu = np.log(524) - sigma**2 / 2
    return [max(8, int(rng.lognormvariate(mu, sigma))) for _ in range(n)]


def test_each_index_once_epochs_differ_and_reproduce():
    lengths = lognormal(500)
    s = BucketedSampler(lengths, 2, seed=3)
    e0 = list(s)
    assert list(s) == e0
    s.set_epoch(1)
    e1 = list(s)
    assert sorted(e0) == list(range(500)) == sorted(e1)
    assert e0 != e1
    s2 = BucketedSampler(lengths, 2, seed=3)
    assert list(s2) == e0
    s2.set_epoch(1)
    assert list(s2) == e1
    assert len(s) == 500


def test_set_epoch_resume_and_dataloader_epochs():
    lengths = lognormal(203)
    ref = BucketedSampler(lengths, 4, seed=7)
    fresh = BucketedSampler(lengths, 4, seed=7)
    fresh.set_epoch(1)
    assert list(fresh) == [i for b in ref.batches(1) for i in b]
    assert list(fresh) == list(fresh)
    dl = DataLoader(list(range(203)), batch_size=4, sampler=ref)
    for e in (0, 1):
        ref.set_epoch(e)
        assert [b.tolist() for b in dl] == ref.batches(e)


def test_no_global_random_dependence():
    lengths = lognormal(300)
    random.seed(1)
    a = list(BucketedSampler(lengths, 4))
    random.seed(2)
    assert a == list(BucketedSampler(lengths, 4))


@pytest.mark.parametrize("n,bs,w", [(103, 4, 3), (100, 2, 5), (7, 3, 32), (64, 2, 32)])
def test_dataloader_batches_match(n, bs, w):
    lengths = lognormal(n, seed=n)
    s = BucketedSampler(lengths, bs, window_batches=w, seed=5)
    expected = s.batches(0)
    got = [b.tolist() for b in DataLoader(list(range(n)), batch_size=bs, sampler=s, drop_last=False)]
    assert got == expected
    assert all(len(b) == bs for b in got[:-1])


def test_short_batch_last_and_batches_shuffled():
    lengths = lognormal(1000)
    batches = BucketedSampler(lengths, 3, window_batches=4).batches(0)
    assert len(batches[-1]) == 1000 % 3
    assert all(len(b) == 3 for b in batches[:-1])
    means = [statistics.mean(lengths[i] for i in b) for b in batches[:-1]]
    # without the shuffle, batches within a window ascend, so most adjacent pairs would rise
    rises = sum(b > a for a, b in zip(means, means[1:], strict=False))
    assert 0.35 < rises / (len(means) - 1) < 0.65
    # within a batch lengths are sorted (bucketed), not by index
    assert all(lengths[b[0]] <= lengths[b[-1]] for b in batches[:-1])
    assert any(b != sorted(b) for b in batches)


def test_batch_size_one_is_permutation():
    s = BucketedSampler(list(range(50)), 1, seed=2)
    a = list(s)
    s.set_epoch(1)
    b = list(s)
    assert sorted(a) == list(range(50)) and a != b and a != list(range(50))


def test_padding_ratio_values():
    assert padding_ratio([10, 10, 20, 20], [0, 1, 2, 3], 2) == 1.0
    assert padding_ratio([10, 20, 10, 20], [0, 1, 2, 3], 2) == pytest.approx(80 / 60)


def test_padding_on_lognormal():
    lengths = lognormal()
    order = list(BucketedSampler(lengths, 2, seed=0))
    assert padding_ratio(lengths, order, 2) <= 1.03
    rand = list(range(len(lengths)))
    random.Random(0).shuffle(rand)
    assert padding_ratio(lengths, rand, 2) >= 1.08


def test_updates_not_length_sorted():
    lengths = lognormal()
    s = BucketedSampler(lengths, 2, seed=0)
    batches = s.batches(0)[:-1]
    m = np.array([np.mean([lengths[i] for i in b]) for b in batches])
    assert abs(np.corrcoef(m[:-1], m[1:])[0, 1]) < 0.2
    k = len(m) // 8 * 8
    runs = m[:k].reshape(-1, 8).mean(axis=1)
    assert runs.std() < 0.5 * m.std()
