"""
Various programs that transform my input data
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class DummyTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    @staticmethod
    def transform(X):
        """Returns back a numpy array of zeroes with the exact same length as the input"""
        transformed_x = np.zeros([len(X), 1])
        return transformed_x

