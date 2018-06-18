"""
Ensure that scorer works as expected
"""

from sklearn.metrics import log_loss
import numpy as np
import pytest


def test_scorer():
    columns = ["a", "b", "c"]
    y_true = ["a", "b"]
    y_pred = [[1, 0, 0],
              [0, 1, 0]]

    assert log_loss(y_true, y_pred, labels=columns) == pytest.approx(0)


def test_numpy_softmax_conversion():
    """Check that getting max prediction is working"""
    scores = np.array([[100, 0.3231506,  0.2042883],
                      [0.40807605, 200,  0.31776983],
                      [300, 0.28616235, 0.23457035]])

    one_hot_score = np.zeros(scores.shape)
    one_hot_score[np.arange(3), np.argmax(scores, axis=1)] = 1
    print(one_hot_score)
    assert (one_hot_score == np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [1, 0, 0]])).all()
