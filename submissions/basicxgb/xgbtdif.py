"""
Experimental code for submissions to Kaggle
"""

# Load data
from spookyauthor.data import make_dataset
train, test = make_dataset.load_raw_data()

# Transform train data
from spookyauthor.data.transform import Preprocessing
id, text, authors = Preprocessing.split_cols(train)

# Fit pipeline
from spookyauthor.models import pipeline
clf = pipeline.basicxgb
clf.fit(text, authors)

# Predict from test text
from spookyauthor.models.predict_model import submission_generator
submission_generator(test, clf, "BasicXGB.csv")
