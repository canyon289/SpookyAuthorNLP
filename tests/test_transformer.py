"""
Tests that transformer works as expected
"""
import pandas as pd
from spookyauthor.models.transform import TextTransformer
import numpy as np
import pytest
from keras.preprocessing.text import Tokenizer


@pytest.fixture
def text():
    """Return a series with text"""
    test = ["This is four words",
            "This one is five words"]

    return pd.Series(test)


def test_word_count(text):
    """Check that word count transformer works correctly on Pandas Series"""

    clf = TextTransformer(features=["word_count"])
    t = clf.transform(text)

    assert t[0] == 4
    assert t[1] == 5


def test_string_length(text):
    """Check that string length works for each sentence"""

    clf = TextTransformer(features=["sentence_length"])
    t = clf.transform(text)

    assert t[0] == 18


def test_two_transformers_at_once(text):
    """Ensure that I can use multiple sets of transformers in one go"""

    clf = TextTransformer(features=["word_count", "sentence_length"])
    t = clf.transform(text)

    assert (t[0] == [4, 18]).all()


def test_naive_bayes_feature_names():
    """Test that I can get back labels from the Naive Bayes Transformer"""
    from spookyauthor.models.transform import NaiveBayesTransformer

    x = [[1, 0],
         [0, 1]]
    y = ["Col1", "Col2"]
    transformer = NaiveBayesTransformer()
    features = transformer.fit(x, y)

    assert transformer.get_feature_names() == ["nb_Col1", "nb_Col2"]


def test_keras_to_sequences(text):
    """Turns (1,m) strings into (m, n_words) vectors
        Each index is an integer representing the word
    """
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    assert sequences[0][0] == sequences[1][0]

    # List lengths can be different
    assert len(sequences[0]) == 4
    assert len(sequences[1]) == 5


def test_keras_tokenizer(text):
    """Returns whether the word index exists, or tfidf, or count of word etc"""
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_matrix(text, mode='binary')
    assert sequences[0][0] == sequences[1][0]
