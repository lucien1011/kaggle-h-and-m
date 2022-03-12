import os
import pandas as pd
import pickle
from tqdm import tqdm

outdir = 'storage/preprocessing/220302_mlp_baseline/'

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

def calculate_normalized_vector(df,column,cats):
    value_counts = df[column].value_counts()
    value_counts = value_counts / value_counts.sum()
    for cat in cats:
        if cat not in value_counts:
            value_counts[cat] = 0.
    return value_counts

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def construct_map(df,name,outdir):
    df['count'] = 1
    m = df.groupby(['customer_id',name])['count'].count().to_dict()
    s = df.groupby(['customer_id'])['count'].count().to_dict()
    path = os.path.join(outdir,'map_'+name+'.p')
    pickle.dump(m,open(path,'wb'))
    path = os.path.join(outdir,'sum_'+name+'.p')
    pickle.dump(s,open(path,'wb'))

def run():
    trans = pd.read_csv('storage/transactions_train.csv')
    arts = pd.read_csv('storage/articles.csv')
    
    tran_trn_df,tran_val_df = train_test_split(trans)
    tran_trn_df = tran_trn_df.merge(arts,on='article_id')

    for c in ['garment_group_name',]:
        print_header(c)
        construct_map(tran_trn_df,c,outdir)
    
if __name__ == '__main__':
    run()
