#!/usr/bin/env python3

from utils import reduce_memory_footprint

def get_correlated_features(features_df):
    """ This function gives correlated features from a dataframe

    Args:
        features_df : pd.DataFrame()
    Returns:
        correlated_features: set() Set of correlated_features
    """
    corr_df = features_df.corr()

    corr_set = set()
    for feature_1 in corr_df.columns.tolist():
        a = corr_df[feature_1] > 0.99
        for feature_2 in a[a].index.values_host:
            if feature_2 != feature_1:
                corr_set.add(tuple(sorted((feature_1, feature_2))))

    correlated_features = set()
    for x,y in corr_set:
        correlated_features.add(y)

    print("\nNum of correlated features: ", len(correlated_features))

if __name__ == "main":
    df = reduce_memory_footprint('Full_Features.csv')
    # Remove bar columns for removing correlated features
    features_df = df.drop(['Bar'], axis=1)
    correlated_features = get_correlated_features(features_df)

    reduced_correlated_features = df.drop(correlated_features, axis=1)
    # Pandas conversion helps keep filesize small
    # reduced_feat_pandas = reduced_correlated_features.to_pandas()
    # reduced_feat_pandas.to_csv('Non_Correlated_Features.csv', index=False, float_format='%.2f')
    reduced_correlated_features.to_csv('Non_Correlated_Features.csv', index=False)
