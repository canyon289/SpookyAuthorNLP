from sklearn.pipeline import Pipeline, FeatureUnion

# Load data
from spookyauthor.data import make_dataset
train, test = make_dataset.load_raw_data()

# Preprocess  data
from spookyauthor.data.make_dataset import Preprocessing
id, text, authors = Preprocessing.split_cols(train)

# Feature pipeline
from sklearn.pipeline import Pipeline
from spookyauthor.models.transform import TextTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

features = FeatureUnion(
    [
    ('basic_preprocessing', TextTransformer(features=['word_count', 'sentence_length'])),
    ("tfidf", TfidfVectorizer())
    ]
)

# Add Predictor
from xgboost import XGBClassifier
predict_pipeline = Pipeline(
    [
    ('union', features),
    ('xgb', XGBClassifier(objective=' multi:softprob', silent=False)),
    ]
)

predict_pipeline.fit(text, authors)

# Predict from test text
from spookyauthor.models.submission import generator
generator(test, predict_pipeline, "XGB_with_word_count.csv")