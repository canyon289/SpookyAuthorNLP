"""
Tests that pipeline functionality works as expected
"""

from spookyauthor.models import pipeline
import numpy as np
import numpy.testing


def test_dummy_pipeline():
    """Tests that dummy pipeline works as expected"""
    train_x = np.array(["a", "b", "c"]).reshape(-1, 1)
    train_y = [1, 1, 2]

    test_x = np.array(["d", "e"]).reshape(-1, 1)
    clf = pipeline.dummypipeline.fit(train_x, train_y)

    test_y = clf.predict(test_x)

    numpy.testing.assert_array_equal(test_y, [1, 1])
