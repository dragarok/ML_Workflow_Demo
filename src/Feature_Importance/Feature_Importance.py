# -*- coding: utf-8 -*-
# Alok's python Feature Importance Script here

import yaml
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# TODO Uncomment this since it's slowing things down
# from featurewiz import featurewiz
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
# For training RF model

def read_config(fname="params.yaml"):
    """Function to read and return config params from yaml file

    Args:
        fname (str) : File name for yaml file
    Returns:
        dict : Dictionary with parameters from yaml file

    """
    with open(fname, "r") as fs:
        try:
            return yaml.safe_load(fs)['Feature_Selection']
        except yaml.YAMLError as exc:
            print(exc)
            return

def train_model(config):
    """This function trains a simple model for testing cml workflow
    to plot confusion matrix.

    Args:
        config (dict) : Config for feature selection
    Returns:
        df (pd.DataFrame) : DataFrame containing (actual_output, predicted_output)
    """
    pass


def select_k_best_features_sklearn(df, config):
    """This function selects k best features using feature selection from sklearn

    Args:
        config (dict) : Config for feature selection
    Returns:
        list: List of best features
    """
    features_df = df.drop(["Label"], axis=1)

    labels_df = df["Label"]

    k_best = (
        SelectKBest(chi2, k=config['n_features'])
        .fit(features_df, labels_df)
    )
    f_cols_idx = k_best.get_support(indices=True)
    all_cols = list(features_df.columns)
    f_cols_sel = [all_cols[i] for i in f_cols_idx]
    return f_cols_sel


def select_k_best_features_featurwiz(df, config):
    """This function selects k best features using feature selection from featurewiz

    Args:
        df (pd.DataFrame): Dataframe from csv values
        k (int): Number of features to select
    Returns:
        list: List of best features
    """
    out1, out2 = featurewiz(
        df,
        "Label",
        corr_limit=0.70,
        verbose=0,
        sep=",",
        header=0,
        test_data="",
        feature_engg="",
        category_encoders="",
    )
    # TODO Do we save all the ranks from featurewiz as well?
    if len(out1) > config['n_features']:
        out1 = out1[:config['n_features']]
    out = [(f,i+1) for i,f in enumerate(out1)]
    out_df = pd.DataFrame(out, columns=['Col_Name', 'Featurewiz_Rank'])
    return out_df


if __name__ == "__main__":
    # Read input data and drop unuseful column
    config = read_config()
    fpath = 'Full_Features.csv'
    df = pd.read_csv(fpath)
    if "Bar" in list(df.columns):
        df = df.drop("Bar", axis=1)

    # Run feature selection
    selected_cols = select_k_best_features_sklearn(df, config)
    # We need label as well for reduced feature used to train data later
    selected_cols.append('Label')
    reduced_features = df[selected_cols]
    # Ensure output directory exists
    os.makedirs('../2_Training_Workflow', exist_ok=True)
    reduced_features.to_csv("../2_Training_Workflow/Reduced_Features.csv", index=False)
    print("Saved features to Selected Features File")
