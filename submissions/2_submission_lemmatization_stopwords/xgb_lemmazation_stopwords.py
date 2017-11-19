import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
import numpy as np

SUBMISSION = False
FULL_EVALUATE = True

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
from sklearn.feature_extraction.text import TfidfVectorizer
from spookyauthor.models.transform import lemmatizer

features = FeatureUnion(
    [
    ('basic_preprocessing', TextTransformer(features=['word_count', 'sentence_length'])),
    ("tfidf", TfidfVectorizer(tokenizer=lemmatizer, stop_words='english'))
    ]
)

# Add Predictor
from xgboost import XGBClassifier
predict_pipeline = Pipeline(
    [
    ('union', features),
    ('xgb', XGBClassifier(n_estimators=1000, objective=' multi:softprob', eval_metric='mlogloss', silent=False)),
    ]
)

# Make Prediction and check score


if FULL_EVALUATE is True:
    from spookyauthor.models.utilities import plot_confusion_matrix
    from sklearn.metrics import log_loss

    predict_pipeline.fit(text, author_int)
    y_pred = predict_pipeline.predict_proba(text)

    score = log_loss(authors, y_pred)
    print("Score was {0}".format(score))


    cm = confusion_matrix(author_int, np.argmax(y_pred,axis=1))
    print(cm)
    plot_confusion_matrix(cm, Labels.classes_)

if SUBMISSION is True:
    # Predict from test text

    predict_pipeline.fit(text, author_int)
    y_pred = predict_pipeline.predict(text)

    from spookyauthor.models.submission import generator

    filename = os.path.basename(__file__).split(".")[0] + '.csv'
    generator(test, predict_pipeline, filename)
