"""
Test that models are returning expected values in trivial situations
"""

import pytest
import numpy as np
from sklearn.dummy import DummyClassifier


@pytest.fixture(scope='session')
def dummyclassifier():
    """Fit dummy classifier on bogus data"""
    train_x = np.array([1, 2, 3, 4]).reshape(-1, 1)
    train_y = [0, 1, 2, 2]
    test_x = np.array([5, 6, 7]).reshape(-1, 1)

    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(train_x, train_y)
    return clf, test_x


def test_constant_predictions(dummyclassifier):
    """Ensure that dummy predictor returns the same class label in single index"""
    clf, test_x = dummyclassifier
    pred = clf.predict(test_x)
    assert (pred == [2, 2, 2]).all()


def test_predictions_proba(dummyclassifier):
    """Ensure that dummy predictor returns the expected probability array
    per class label"""
    clf, test_x = dummyclassifier
    pred = clf.predict_proba(test_x)
    expected_preds = [[0, 0, 1] for i in range(3)]

    assert (pred == expected_preds).all()


def test_predictions_logproba(dummyclassifier):
    """Ensure that dummy predictor returns the expected probability 
    per class label"""
    clf, test_x = dummyclassifier
    pred = clf.predict_log_proba(test_x)
    expected_preds = [[-np.inf, -np.inf, 0] for _ in range(3)]
    assert (pred == expected_preds).all()
