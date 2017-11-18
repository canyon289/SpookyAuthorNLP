"""
Various programs that transform my input data
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer

class DummyTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    @staticmethod
    def transform(X):
        """Returns back a numpy array of zeroes with the exact same length as the input"""
        transformed_x = np.zeros([len(X), 1])
        return transformed_x


class TextTransformer:

    def __init__(self, features):
        """Takes column of text and returns back array of features. Features list is specified by features

        Parameters
        ----------
        features : iter
            Iterable of features strings that correspond to class

        Returns
        -------
            Numpy array
        """

        self.features = features
        return

    def fit(self, X, y):
        return self

    def transform(self, X):
        """Accepts Pandas Series and returns back a numpy array where each column is a feature"""

        if self.features is None:
            raise ValueError("Features must be a list of features that will be used in this model")

        features = pd.DataFrame()

        for feature in self.features:
            transformer = getattr(self, feature)
            col = transformer(X)

            features[feature] = col

        return features.values

    @staticmethod
    def word_count(X):
        """Count the number of words in each string"""
        return X.apply(lambda x: len(x.split(" ")))

    @staticmethod
    def sentence_length(X):
        """Get the length of the string for each author"""
        return X.apply(lambda x: len(x))


WNL = WordNetLemmatizer()


def lemmatizer(doc):
    """Lemmatizes a string"""
    words = [WNL.lemmatize(word) for word in doc.split(" ")]
    return words
