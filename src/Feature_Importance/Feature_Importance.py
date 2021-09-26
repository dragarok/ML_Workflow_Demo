# -*- coding: utf-8 -*-
# Alok's python Feature Importance Script here

import yaml
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import sys
from git import Repo
# TODO Uncomment this since it's slowing things down
# from featurewiz import featurewiz
import os
import tensorflow_cloud as tfc
import subprocess


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
    # TODO Parametrize this
    gcp_bucket = 'tfc-cml'
    # Running in gcloud with url sent
    if len(sys.argv) == 2:
        git_url = sys.argv[1]
    else:
        # Get repo token to fetch github repo
        token = os.environ['REPO_TOKEN']
        with open('../../../src/params.yaml', 'r') as stream:
            config = yaml.safe_load(stream)
        branch_name = config['git']['git_branch_name']
        github_username = config['git']['git_username']
        repo_main_url = config['git']['git_repo']
        git_url = "https://" + github_username + ":" + token + "@" + repo_main_url
        # tfc.run(
        #     entry_point='../../../src/Feature_Importance/Feature_Importance.py',
        #     requirements_txt='../../../src/Feature_Importance/requirements.txt',
        #     entry_point_args=[git_url],
        #     chief_config=tfc.MachineConfig(
        #             cpu_cores=8,
        #             memory=30),
        #     docker_image_bucket_name=gcp_bucket,
        # )
        os.system('pyenv activate dvc_t3; python ../../../src/Feature_Importance/Feature_Importance.py ' + git_url)
        sys.exit()
        
    repo = Repo.clone_from(git_url, 'cloned_repo')
    # TODO Parametrize this
    repo.git.checkout('workflow_test')
    print("Cloned the repo")

    # Now pull the data into the repo
    pull_dvc_cmd = 'cd cloned_repo; dvc pull --run-cache'
    ret = subprocess.run([pull_dvc_cmd], capture_output=True, shell=True)
    if ret.returncode == 0:
        print("\nPulled the data\n")
        print(ret)
    else:
        sys.stderr.write("Error Pulling data from dvc\n\t")
        sys.stderr.write(str(ret))


    indir = 'ContinuousML/NQ_DR_4_10_20_Ideal_27/1_Label_And_Feature_Workflow/'
    outdir = 'ContinuousML/NQ_DR_4_10_20_Ideal_27/2_Training_Workflow'

    # Read input data and drop unuseful column
    fpath = os.path.join('cloned_repo', indir, 'Full_Features.csv')
    df = pd.read_csv(fpath)
    if "Bar" in list(df.columns):
        df = df.drop("Bar", axis=1)
    config = read_config()

    # Run feature selection
    selected_cols = select_k_best_features_sklearn(df, config)
    # We need label as well for reduced feature used to train data later
    selected_cols.append('Label')
    reduced_features = df[selected_cols]
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join('cloned_dir',outdir, "Reduced_Features.csv")
    # Now we have to save files into this repository and commit
    reduced_features.to_csv(outfile, index=False)
    print("Features have been reduced\n\n")

    # Dvc add file
    add_file_dvc_cmd = 'cd cloned_repo; dvc add ' + outfile
    ret = subprocess.run([add_file_dvc_cmd], capture_output=True, shell=True)
    print(ret)

    # Dvc push to push output to dvc remote
    push_model_dvc_cmd = 'cd cloned_repo; dvc push'
    ret = subprocess.run([push_model_dvc_cmd], capture_output=True, shell=True)
    print(ret)

    repo.git.add(update=True)
    COMMIT_MESSAGE = 'Saved features to selected features file'
    repo.index.commit(COMMIT_MESSAGE)
    origin = repo.remote(name='origin')
    origin.push()

    print("Script execution complete")
