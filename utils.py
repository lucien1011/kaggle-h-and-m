import os
import pandas as pd

def cudf_train_test_split(df):
    import cudf
    ts_split = cudf.to_datetime('2020-09-15')
    df['t_dat'] = cudf.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def pd_train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def train_test_split(df,gpu=False):
    if gpu:
        return cudf_train_test_split(df)
    else:
        return pd_train_test_split(df)

def print_memory_usage(obj):
    print('Used '+str(obj.memory_usage(deep=True).sum() / 1024 / 1024 / 1024)+" Gb")
    
def save_csv(df,out_dir,fname):
    os.makedirs(out_dir,exist_ok=True)
    df.to_csv(os.path.join(out_dir,fname),index=False)