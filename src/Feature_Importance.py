# -*- coding: utf-8 -*-
# Alok's python Feature Importance Script here

import yaml
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# TODO Uncomment this since it's slowing things down
from featurewiz import featurewiz
import seaborn as sns
# import matplotlib.pyplot as plt
import os
import dvc.api
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


def get_df_from_dvc(config, read_dvc=False):
    in_fname_full = os.path.join(config['in_dir'], config['in_fname'])
    if read_dvc:
        fpath = dvc.api.get_url(in_fname_full)
    else:
        fpath = in_fname_full
    df = pd.read_csv(fpath)
    return df


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
    f_cols_list = list(features_df.columns)
    feat_scores_tuple = [(f, score) for f,score in zip(f_cols_list, k_best.scores_)]
    out_df = pd.DataFrame(feat_scores_tuple, columns=['Col_Name', 'Sklearn_Scores'])
    out_df['Sklearn_Rank'] = out_df['Sklearn_Scores'].rank(ascending=False, method='first')
    out_df = out_df[out_df['Sklearn_Scores'].notna()]
    out_df = out_df.drop(['Sklearn_Scores'], axis=1)
    return out_df


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
    df = get_df_from_dvc(config)
    if "Bar" in list(df.columns):
        df = df.drop("Bar", axis=1)

    # Run feature selection
    sklearn_out = select_k_best_features_sklearn(df, config)
    featurewiz_out = select_k_best_features_featurwiz(df, config)

    # Merge results into single dataframe and save to csv
    features_df = df.drop(["Label"], axis=1)
    cols_list = list(features_df.columns)
    out_df = pd.DataFrame(cols_list, columns=['Col_Name'])
    out_df_sm = pd.merge(out_df, sklearn_out, how='left')
    out_df_final = pd.merge(out_df_sm, featurewiz_out, how='left')
    out_df_final['Sklearn_Rank'] = pd.to_numeric(out_df_final['Sklearn_Rank'], downcast='integer')

    # Ensure output directory exists
    os.makedirs(config['out_dir'], exist_ok=True)

    csv_fname = os.path.join(config['out_dir'], "Selected_Features.csv")
    out_df_final.to_csv(csv_fname, index=False)
    print("Saved features to csv file")
