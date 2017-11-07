"""
Various pipelines used in predictor
"""

from sklearn.pipeline import Pipeline
from spookyauthor.data import transform
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

"""
Dummy classifier that does nothing but predict the most common author
"""
dummypipeline = Pipeline([("DummyTransformer", transform.DummyTransformer()),
                         ("DummyPredictor", DummyClassifier(strategy='most_frequent'))

                     ])


basicxgb = Pipeline([("tfidf", TfidfVectorizer()),
                     ("XGBBaseModel", XGBClassifier(objective=' multi:softprob')),
                     ])
