#!/usr/bin/env python3
import cudf
import numpy as np
import pandas as pd

# TODO Sample size from parameters
def reduce_memory_footprint(filename, sample_size=5000):
    """ Reduce memory footprint for dataframe by changing datatype
    for individual columns since everything is 2 decimal places"""

    df_test = cudf.read_csv(filename, nrows=sample_size)

    int_cols = [c for c in df_test if df_test[c].dtype == "int64"]
    int_cols = {c: np.int8 for c in int_cols}
    int_cols['Bar'] = np.int64

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float16_cols = {c: np.float32 for c in float_cols}
    dtype_cols = dict(float16_cols, **int_cols)

    df = cudf.read_csv(filename, engine='c', dtype=dtype_cols)
    return df


def reduce_memory_usage(df):
    """ Reduce memory footprint for columns pandas dataframe
    NOTE: Needs reading the columns one by one though"""

    start_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_memory} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')

    end_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe after reduction {end_memory} MB")
    print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df
