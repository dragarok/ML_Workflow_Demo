# -*- coding: utf-8 -*-
# Alok's python Feature Importance Script here

import yaml
import pandas as pd
import cudf
import cuml
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFECV
import numpy as np
from sklearn.svm import SVR
from featurewiz import featurewiz
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
from utils import reduce_memory_footprint

def read_config(fname="params.yaml"):
    """Function to read and return config params from yaml file

    Args:
        fname (str) : File name for yaml file
    Returns:
        dict : Dictionary with parameters from yaml file

    """
    with open(fname, "r") as fs:
        try:
            return yaml.safe_load(fs)
        except yaml.YAMLError as exc:
            print(exc)
            return

        
def compute_feature_importance(voting_clf, xgb_clf, kbest_feat_imp, weights):
    """ Function to compute feature importance given a voting classifier """

    feature_importance = dict()
    for est in voting_clf.estimators_:
        feature_importance[str(est)] = est.feature_importances_

    # xgboost taking different input forms from normal classifiers
    feature_importance['xgb'] = xgb_clf.feature_importances_

    # kbest feature selection
    feature_importance['selectk'] = kbest_feat_imp

    fe_scores = [0]*len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score*weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))
    return fe_scores


def select_k_best_features_voting(df, config):
    """This function selects k best features using feature selection using cuml

    Args:
        df (pd.DataFrame): Dataframe from csv values
        k (int): Number of features to select
    Returns:
        list: List of best features
    """

    features_df = df.drop(["Label"], axis=1)
    labels_df = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_df, test_size=0.05)

    le = preprocessing.LabelEncoder()
    le.fit(labels_df.to_gpu_array())

    y_train = le.transform(y_train.to_gpu_array())
    y_test = le.transform(y_test.to_gpu_array())

    rf_clf = RandomForestClassifier(n_estimators = 200)
    dc_clf = DecisionTreeClassifier()
    ab_clf = AdaBoostClassifier(n_estimators=200)
    estimators = [('AB', ab_clf), ('RF', rf_clf), ('DC', dc_clf)]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', verbose=True)
    voting_clf.fit(X_train.as_gpu_matrix(), y_train)

    # Train xgb classifier as well 
    xgb_clf = XGBClassifier(seed=41, gpu_id=0, tree_method='gpu_hist')
    xgb_clf.fit(X_train, y_train)

    k_best = (
        SelectKBest(chi2, k=config['n_features'])
        .fit(features_df.as_gpu_matrix(), labels_df.to_gpu_array())
    )
    x = np.nan_to_num(k_best.scores_)
    kbest_feat_imp = x / np.sum(x) 

    print("Done training voting classifier")

    df = cudf.DataFrame()
    df['Feature'] = features_df.columns
    df['Feature Importance'] = compute_feature_importance(voting_clf, xgb_clf, kbest_feat_imp,
                                                          [1, 1, 1, 1])
    df = df.sort_values('Feature Importance', ascending=False)
    df.drop('Feature Importance', inplace=True, axis=1)
    return df.head(config['n_features'])


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
    return out1


def save_dtree_viz(viz_df):
    """ Function to save a decision tree visualization"""
    features_df = viz_df.drop(["Label"], axis=1)
    labels_df = viz_df["Label"]
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(features_df.as_gpu_matrix(), labels_df.as_gpu_array())
    viz = dtreeviz(clf, features_df, labels_df, target_name='classifier',
                   feature_names=list(features_df.columns),
                   class_names=list(labels_df.unique().values_host))
    viz.save('decision_tree.svg')

if __name__ == "__main__":
    print("Start of the script")
    cuml.set_global_output_type('numpy')
    # Read input data and drop unuseful column
    config = read_config()
    feat_df = reduce_memory_footprint('Full_Features.csv')

    label_df = cudf.read_csv('Label.csv')
    label_df = label_df.drop('Visual_Label', axis=1)

    df = feat_df.merge(label_df, on=['Bar'])
    df = df.rename(columns = {'ML_Label': 'Label'})

    if "Bar" in list(df.columns):
        df = df.drop("Bar", axis=1)

    # Run feature selection featurewiz
    mode = config['mode']
    if mode == "featurewiz":
        # Run feature selection featurewiz
        selected_cols = select_k_best_features_featurwiz(df, config)
        sel_df = cudf.DataFrame(selected_cols, columns=['Features'])
    else:
        # Run feature selection using sklearn
        sel_df = select_k_best_features_voting(df, config)
        selected_cols = sel_df['Feature'].tolist()

    # Ensure output directory exists
    os.makedirs('../2_Training_Workflow', exist_ok=True)
    sel_df.to_csv("../2_Training_Workflow/Selected_Features.csv", index=False, header=False)
    print("\nSaved features to Selected Features File\n")

    # # For visualization with dtreeviz
    # dtree_config = config.copy()
    # dtree_config['n_features'] = config['vis_features']
    # viz_cols = select_k_best_features_sklearn(df, dtree_config)
    # viz_cols.append('Label')
    # viz_df = df[viz_cols]
    # save_dtree_viz(viz_df)
    
    # In other methods, we get features only.
    # We need label as well for reduced feature used to train data in other stages
    selected_cols.append('Label')
    reduced_features = df[selected_cols]
    reduced_features.to_csv("../2_Training_Workflow/Reduced_Features.csv", index=False)
    print("\nSaved Reduced DataFrame from Selected Features\n")
