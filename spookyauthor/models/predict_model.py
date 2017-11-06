"""
All code for predictive models goes here
"""
import pandas as pd
from ..transform import preprocessing


def submission_generator(test_df, pipeline, filename):
    """Takes test dataframe and object that implements transform/predict methods
    and writes of predictions for submission to Kaggle
    
    Returns
    -------
    Dataframe of predictions
    """
    ids, text, _ = preprocessing.split_cols(test_df)

    preds = pipeline.predict_proba(text)

    df = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
    df["id"] = ids
    df = df.set_index("id")

    df.to_csv(filename)
    return df

