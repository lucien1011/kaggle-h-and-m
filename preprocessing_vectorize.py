import os
import numpy as np
import pandas as pd
import pickle

conf = dict(
        tran_csv_path='storage/transactions_train.csv',
        art_csv_path='storage/articles.csv',
        cust_csv_path='storage/customers.csv',
    )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',action='store')
    parser.add_argument('--feature',action='store')
    return parser.parse_args()

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

def past_purchase_feature_engineering(art_df,cust_df,tran_df,selected_feature):
    df = tran_df.merge(art_df,on='article_id')
    df['count'] = 1
    norm = df.groupby(['customer_id'])['count'].count().to_dict()
    count = df.groupby(['customer_id',selected_feature])['count'].count().reset_index()
    count['norm'] = count['customer_id'].map(norm)
    count['count'] = count['count'] / count['norm']
    count = count.rename(columns={'count':selected_feature+'_count'})
    return count,norm

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def save(obj,path):
    pickle.dump(obj,open(path,'wb'))

def save_df(obj,path):
    obj.to_csv(path,index=False)

def run(conf,args):

    print_header('Read csv files')
    tran_df = pd.read_csv(conf['tran_csv_path'])
    art_df = pd.read_csv(conf['art_csv_path'])
    cust_df = pd.read_csv(conf['cust_csv_path'])

    print_header('Train test split')
    tran_trn_df,tran_val_df = train_test_split(tran_df)
    
    print_header('Feature engineering')
    count,norm = past_purchase_feature_engineering(art_df,cust_df,tran_trn_df,args.feature)
    save_df(count,os.path.join(args.out_dir,args.feature+'_count.csv'))
    save(norm,os.path.join(args.out_dir,'norm.p'))

if __name__ == '__main__':
    args = parse_arguments()
    run(conf,args)
