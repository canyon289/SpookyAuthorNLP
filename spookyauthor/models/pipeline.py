"""
Various pipelines used in predictor
"""

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from spookyauthor.models import transform



# Dummy classifier that does nothing but predict the most common author

dummypipeline = Pipeline([("DummyTransformer", transform.DummyTransformer()),
                         ("DummyPredictor", DummyClassifier(strategy='most_frequent'))

                     ])

# Basic XGB classifier that just takes vectorization and outputs result
basicxgb = Pipeline([("tfidf", TfidfVectorizer()),
                     ("XGBBaseModel", XGBClassifier(objective=' multi:softprob')),
                     ])




