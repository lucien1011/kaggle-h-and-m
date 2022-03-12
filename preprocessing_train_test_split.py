import os
import numpy as np
import pandas as pd
import pickle

conf = dict(
        tran_csv_path='storage/transactions_train.csv',
        art_csv_path='storage/articles.csv',
        cust_csv_path='storage/customers.csv',
        out_dir='storage/preprocessing/',
    )


def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def merge_df(tran_df,art_df,cust_df):
    out = tran_df.merge(art_df,on='article_id')
    out = out.merge(cust_df,on='customer_id')
    return out

def save_df(obj,path):
    obj.to_csv(path,index=False)

def run(conf):

    print_header('Read csv files')
    tran_df = pd.read_csv(conf['tran_csv_path'])
    art_df = pd.read_csv(conf['art_csv_path'])
    cust_df = pd.read_csv(conf['cust_csv_path'])
    
    merge_tran_df = merge_df(tran_df,art_df,cust_df)

    print_header('Train test split')
    tran_trn_df,tran_val_df = train_test_split(merge_tran_df)
    
    save_df(tran_trn_df,conf['out_dir']+'tran_trn_df.csv')
    save_df(tran_val_df,conf['out_dir']+'tran_val_df.csv')

if __name__ == '__main__':
    run(conf)
