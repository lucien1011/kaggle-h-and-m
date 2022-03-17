import os
import pandas as pd

def cudf_train_test_split(df,time='2020-09-15'):
    import cudf
    ts_split = cudf.to_datetime(time)
    df['t_dat'] = cudf.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def pd_train_test_split(df,time='2020-09-15'):
    ts_split = pd.to_datetime(time)
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def train_test_split(df,time='2020-09-15',gpu=False):
    if gpu:
        return cudf_train_test_split(df,time)
    else:
        return pd_train_test_split(df,time)
    
def train_val_test_split(df,test_time='2020-09-15',val_time='2020-09-08',gpu=False):
    trn,test = train_test_split(df,time=test_time,gpu=gpu)
    trn,val = train_test_split(trn,time=val_time,gpu=gpu)
    return trn,val,test
    
def x_y_group(data,features,target,only_x=False,verbose=False):
    x = data[features]
    if only_x: return x,None,None
    y = data[target]
    group = data.groupby('customer_id').size().to_frame('size')['size'].to_numpy()
    if verbose: print('shape (x,y,group): ',x.shape,y.shape,group.shape)
    return x,y,group

def print_memory_usage(obj):
    print('Used '+str(obj.memory_usage(deep=True).sum() / 1024 / 1024 / 1024)+" Gb")
    
def save_csv(df,out_dir,fname):
    os.makedirs(out_dir,exist_ok=True)
    df.to_csv(os.path.join(out_dir,fname),index=False)
    
def evaluate_score(pred_df,gt_df,k=12):
    from metric import mapk
    eval_df = gt_df.merge(pred_df,on='customer_id')
    print('map@'+str(k), mapk(eval_df['ground_truth'].tolist(),eval_df['prediction'].tolist()))