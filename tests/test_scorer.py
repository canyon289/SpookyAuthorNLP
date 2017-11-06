"""
Ensure that scorer works as expected
"""

from sklearn.metrics import log_loss
import pytest


def test_scorer():
    columns = ["a", "b", "c"]
    y_true = ["a", "b"]
    y_pred = [[1, 0, 0],
              [0, 1, 0]]

    assert log_loss(y_true, y_pred, labels=columns) == pytest.approx(0)

