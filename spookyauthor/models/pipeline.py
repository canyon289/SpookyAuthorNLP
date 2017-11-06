"""
Various pipelines used in predictor
"""

from sklearn.pipeline import Pipeline
from spookyauthor.data import transform
from sklearn.dummy import DummyClassifier


dummypipeline = Pipeline([("DummyTransformer", transform.DummyTransformer()),
                         ("DummyPredictor", DummyClassifier(strategy='most_frequent'))
                            ])