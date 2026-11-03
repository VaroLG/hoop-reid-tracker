"""040: per-clip adaptation of embeddings from pieces and pieces on screen together."""

import numpy as np

from hooptrack.reid.adapt import adapt, cannot_link_pairs


def test_cannot_link_pairs_are_pieces_sharing_a_frame():
    frames = {1: np.arange(0, 10), 2: np.arange(5, 15), 3: np.arange(20, 30)}
    assert cannot_link_pairs(frames) == [(1, 2)]


def test_adaptation_separates_two_people_hidden_in_one_noisy_direction():
    # two people differ only along axis 0, which is small next to pose noise on the other axes; after adaptation
    # a new piece of person A is closer to A than to B
    rng = np.random.default_rng(0)
    D = 16

    def crops(sign, n=12):
        x = rng.normal(scale=1.0, size=(n, D))
        x[:, 0] = sign * 0.6 + 0.05 * rng.normal(size=n)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    feats, frames = {}, {}
    for k in range(20):                                   # 10 pairs of co-existing pieces A_k, B_k
        feats[2 * k], feats[2 * k + 1] = crops(+1), crops(-1)
        frames[2 * k] = frames[2 * k + 1] = np.arange(100 * k, 100 * k + 10)
    feats[99], frames[99] = crops(+1), np.arange(5000, 5010)
    out, scale = adapt(feats, frames, dim=2)
    d = lambda a, b: float(np.mean(1 - out[a] @ out[b].T))  # noqa: E731
    assert d(99, 0) < d(99, 1) and scale > 0
