"""
Various programs that transform my input data
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB

class DummyTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    @staticmethod
    def transform(X):
        """Returns back a numpy array of zeroes with the exact same length as the input"""
        transformed_x = np.zeros([len(X), 1])
        return transformed_x


class NaiveBayesTransformer(MultinomialNB):
    """Makes Naive Bayes a transformer"""
    def transform(self, X):
        return self.predict_proba(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).predict_proba(X)

    def get_feature_names(self):
        """Gets the feature names of the Maive Bayes Transformer"""
        return ["nb_{0}".format(class_name) for class_name in self.classes_]


class PassthroughTransformer(BaseEstimator, TransformerMixin):
    """Takes Inputs and does nothing to them"""

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


class TextTransformer(BaseEstimator, TransformerMixin):

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
