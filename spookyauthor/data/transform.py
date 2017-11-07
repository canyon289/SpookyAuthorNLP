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


class Preprocessing:

    @staticmethod
    def split_cols(df):
        """Splits dataframe columns into constituent parts

        Parameters
        ----------
        df

        Returns
        -------
        ids
            Series of IDs
        text
            Series with text snippets
        authors
            Series of Author Labels or none if test set

        """

        ids = df["id"]
        text = df["text"]

        try:
            author = df["author"]
        except KeyError:
            author = None

        return ids, text, author
