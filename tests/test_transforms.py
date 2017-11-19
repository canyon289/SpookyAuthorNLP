"""
Pytests for troublesome transformation steps
"""
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest


@pytest.fixture()
def data_target():
    from spookyauthor.data import make_dataset
    train, test = make_dataset.load_raw_data()

    # Preprocess  data
    from spookyauthor.data.make_dataset import Preprocessing
    id, text, authors = Preprocessing.split_cols(train)

    return id, text, authors


def test_svd_sign(data_target):
    """Test SVD to see what type of range is being generated from the decomposition"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from spookyauthor.models.transform import lemmatizer
    from sklearn.decomposition import TruncatedSVD

    _, text, _ = data_target

    TD = TfidfVectorizer(tokenizer=lemmatizer, stop_words='english', ngram_range=(1, 3))
    td_text = TD.fit_transform(text)

    SVD = TruncatedSVD(n_components=1000, algorithm='arpack')

    svd = SVD.fit_transform(td_text)
    assert svd is not None
