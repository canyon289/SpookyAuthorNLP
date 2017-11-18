"""
Test lemmatizer in conjunction with scikit learn to ensure I'm using the functionality coirrectly
"""

from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from spookyauthor.models.transform import lemmatizer



def test_lemmazation():
    """Test that Lemmaitzer works"""
    wnl = WordNetLemmatizer()
    assert wnl.lemmatize('dogs') == 'dog'


def test_lemmazation_with_countvectorizer():
    """Test Lemmatization with Stop Words removal"""
    strings = ["I like Dogs and", "I like Churches and"]
    count = CountVectorizer(tokenizer=lemmatizer, stop_words='english')
    count.fit(strings)

    assert {'church', 'dog', 'like'} == set(count.get_feature_names())

