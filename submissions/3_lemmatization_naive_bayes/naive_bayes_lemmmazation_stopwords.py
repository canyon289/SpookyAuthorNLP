import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

# Setup Logging
logging.basicConfig(filename='model_run.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting New Run \n")

CROSS_EVALUATE = True
FULL_EVALUATE = True
SUBMISSION = True

# Load data
from spookyauthor.data import make_dataset
train, test = make_dataset.load_raw_data()


# Preprocess  data
from spookyauthor.data.make_dataset import Preprocessing
id, text, authors = Preprocessing.split_cols(train)

from sklearn.preprocessing import LabelEncoder
Labels = LabelEncoder()
author_int = Labels.fit_transform(authors)

# Feature pipeline
from sklearn.pipeline import Pipeline
from spookyauthor.models.transform import TextTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spookyauthor.models.transform import lemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
features = Pipeline(
    [
    ("tfidf", CountVectorizer(tokenizer=lemmatizer, stop_words='english', ngram_range=(1, 3))),
    ]
)

# Add Predictor
from sklearn.naive_bayes import MultinomialNB
predict_pipeline = Pipeline(
    [
    ('union', features),
    ('naive_bayes', MultinomialNB()),
    ]
)

# Make Prediction and check score
if CROSS_EVALUATE is True:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    from sklearn.metrics import log_loss

    scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    scores = cross_val_score(predict_pipeline, text, author_int, scoring=scorer)

    logging.info("Cross Val Score is {0}".format(np.mean(scores)))


if FULL_EVALUATE is True:
    from spookyauthor.models.utilities import plot_confusion_matrix
    from sklearn.metrics import log_loss

    predict_pipeline.fit(text, author_int)
    y_pred = predict_pipeline.predict_proba(text)
    score = log_loss(authors, y_pred)

    logging.info("Full Evaluate Score was {0}".format(score))

    cm = confusion_matrix(author_int, np.argmax(y_pred, axis=1))
    print(cm)
    plot_confusion_matrix(cm, Labels.classes_)

if SUBMISSION is True:
    # Predict from test text

    predict_pipeline.fit(text, author_int)
    y_pred = predict_pipeline.predict(text)

    from spookyauthor.models.submission import generator

    filename = os.path.basename(__file__).split(".")[0] + '.csv'
    generator(test, predict_pipeline, filename)
