# -*- coding: utf-8 -*-
# Alok's python Feature Importance Script here

import yaml
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# TODO Uncomment this since it's slowing things down
# from featurewiz import featurewiz
import seaborn as sns
# import matplotlib.pyplot as plt
import os
import dvc.api
import tensorflow_cloud as tfc


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


def select_k_best_features_sklearn(config):
    """This function selects k best features using feature selection from sklearn

    Args:
        df (pd.DataFrame): Dataframe from csv values
        k (int): Number of features to select
    Returns:
        list: List of best features
    """
    df = get_df_from_dvc(config)
    features_df = df.drop(["Label"], axis=1)
    if "Bar" in list(features_df.columns):
        features_df = df.drop("Bar", axis=1)

    labels_df = df["Label"]

    f_cols_idx = (
        SelectKBest(chi2, k=config['n_features'])
        .fit(features_df, labels_df)
        .get_support(indices=True)
    )
    f_cols_list = list(features_df.columns)
    f_sel_cols = [f_cols_list[i-1] for i in f_cols_idx]
    out_df = pd.DataFrame(f_sel_cols)
    out_df = out_df.rename(columns={0: "Feature_cols"})

    # TODO Fix the path for file
    os.makedirs(config['out_dir'], exist_ok=True)
    csv_fname = os.path.join(config['out_dir'], "Feature_Importance_sklearn.csv")
    out_df.to_csv(csv_fname, index=False)
    print("Successfully wrote selected features to file")
    return


def select_k_best_features_featurwiz(config):
    """This function selects k best features using feature selection from featurewiz

    Args:
        df (pd.DataFrame): Dataframe from csv values
        k (int): Number of features to select
    Returns:
        list: List of best features
    """
    df = get_df_from_dvc(config)
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
    if len(out1) > config['n_features']:
        out1 = out1[:config['n_features']]
    # TODO What if the features selected are less than the desired quantity?
    out_df = pd.DataFrame(out1)
    out_df = out_df.rename(columns={0: "Feature_cols"})
    # TODO Fix the path for file
    csv_fname = os.path.join(config['out_dir'], "Feature_Importance_featurewiz.csv")
    out_df.to_csv(csv_fname, index=False)
    print("Successfully wrote selected features to file")
    return


if __name__ == "__main__":
    config = read_config()
    # select_k_best_features_featurwiz(config)
    select_k_best_features_sklearn(config)
