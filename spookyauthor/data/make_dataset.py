# -*- coding: utf-8 -*-
import os
import click
import logging
# from dotenv import find_dotenv, load_dotenv
import pandas as pd

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    return


def load_raw_data(input_filepath='raw'):
    """ Loads test and train datasets into memory
    
    Parameters
    ----------
    input_filepath

    Returns
    -------
    test_df 
    train_df
    """
    project_dir = os.path.join(os.path.dirname(__file__), "raw")
    train = pd.read_csv(os.path.join(project_dir, "train.csv"))
    test = pd.read_csv(os.path.join(project_dir, "test.csv"))
    return train, test


class Preprocessing:

    @staticmethod
    def split_cols(df):
        """Splits dataframe columns into constituent parts

        Parameters
        ----------
        df

        Returns
        -------
        ids
            Series of IDs
        text
            Series with text snippets
        authors
            Series of Author Labels or none if test set

        """

        ids = df["id"]
        text = df["text"]

        try:
            author = df["author"]
        except KeyError:
            author = None

        return ids, text, author


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()


