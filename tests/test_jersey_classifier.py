"""033: digit-by-digit jersey-number reader (composition of the 111 answers)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hooptrack.reid.jersey_classifier import NUMBERS, encode, number_log_probs, read_piece  # noqa: E402


def test_encode():
    assert encode("nan") == (0, -100, -100)
    assert encode("7") == (1, -100, 7)
    assert encode("23") == (2, 2, 3)
    assert encode("00") == (2, 0, 0) and encode("0") == (1, -100, 0)     # "00" and "0" are different numbers


def test_composed_probabilities_sum_to_one_and_follow_the_digits():
    torch.manual_seed(0)
    lp = number_log_probs(torch.randn(4, 3), torch.randn(4, 10), torch.randn(4, 10))
    assert lp.shape == (4, 111) and torch.allclose(lp.exp().sum(1), torch.ones(4), atol=1e-5)
    ll, lt, lu = torch.tensor([[0., 0., 9.]]), torch.zeros(1, 10), torch.zeros(1, 10)
    lt[0, 2], lu[0, 3] = 9, 9
    assert NUMBERS[int(number_log_probs(ll, lt, lu).argmax())] == "23"


def test_read_piece_averages_the_crops():
    p = np.zeros((3, 111))
    p[:, NUMBERS.index("7")] = [0.9, 0.8, 0.1]
    p[:, 0] = [0.1, 0.2, 0.9]
    assert read_piece(p)[0] == "7"
