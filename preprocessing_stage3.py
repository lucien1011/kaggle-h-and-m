import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

conf = dict(
        tran_csv_path='storage/transactions_train.csv',
        art_csv_path='storage/articles.csv',
        cust_csv_path='storage/customers.csv',
        selected_feature_dir='storage/preprocessing/220306_baseline/',
        customer_features=[
                'customer_time_elpased_last_purchase',
                'customer_num_purchase',
                'customer_past_purchase_price_mean',
            ],
        article_features=[
                'num_purchase',
                'time_elpased_last_purchase',
                'price_mean',
            ],
        article_customer_features=[
                'product_group_name',
                'product_type_name',
                'graphical_appearance_name',
                'perceived_colour_value_name',
                'colour_group_code',
                'index_name',
                'index_group_name',
                'section_name',
                'department_name',
            ],
    )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',action='store')
    return parser.parse_args()

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

def print_subheader(message):
    print('*'*100)
    print(message)
    print('*'*100)

def article_feature_engineering(art_df,tran_df,current_time='2020-09-15',):
    out = art_df[['article_id']]

    art_group = tran_df.groupby('article_id')
    
    num_purchase = art_group['t_dat'].count().reset_index()
    num_purchase = num_purchase.rename(columns={'t_dat':'num_purchase'})
    num_purchase['num_purchase'] /= num_purchase['num_purchase'].max() 
    num_purchase = num_purchase[['article_id','num_purchase']]
    
    curr_time_stamp = pd.Timestamp(current_time)
    time_elapsed_last_purchase = art_group['t_dat'].max().reset_index()
    time_elapsed_last_purchase['time_elpased_last_purchase'] = time_elapsed_last_purchase['t_dat'].apply(lambda x: (curr_time_stamp-x).days)
    time_elapsed_last_purchase['time_elpased_last_purchase'] /= time_elapsed_last_purchase['time_elpased_last_purchase'].max()
    time_elapsed_last_purchase = time_elapsed_last_purchase[['article_id','time_elpased_last_purchase']]

    price_mean = art_group['price'].mean().reset_index()
    price_mean.rename(columns={'price':'price_mean'},inplace=True)
    price_mean = price_mean[['article_id','price_mean']]

    price_std = art_group['price'].std().reset_index()
    price_std.rename(columns={'price':'price_std'},inplace=True)
    price_std = price_std[['article_id','price_std']]

    out = out.merge(time_elapsed_last_purchase,on='article_id')
    out = out.merge(num_purchase,on='article_id')
    out = out.merge(price_mean,on='article_id')
    #out = out.merge(price_std,on='article_id')
    return out

def customer_feature_engineering(cust_df,tran_df,current_time='2020-09-15',):
    out = cust_df[['customer_id']]
    
    cust_group = tran_df.groupby('customer_id')

    num_purchase = cust_group['t_dat'].count().reset_index()
    num_purchase = num_purchase.rename(columns={'t_dat':'customer_num_purchase'})
    num_purchase['customer_num_purchase'] /= num_purchase['customer_num_purchase'].max() 
    num_purchase = num_purchase[['customer_id','customer_num_purchase']]
    
    curr_time_stamp = pd.Timestamp(current_time)
    time_elapsed_last_purchase = cust_group['t_dat'].max().reset_index()
    time_elapsed_last_purchase['customer_time_elpased_last_purchase'] = time_elapsed_last_purchase['t_dat'].apply(lambda x: (curr_time_stamp-x).days)
    time_elapsed_last_purchase['customer_time_elpased_last_purchase'] /= time_elapsed_last_purchase['customer_time_elpased_last_purchase'].max()
    time_elapsed_last_purchase = time_elapsed_last_purchase[['customer_id','customer_time_elpased_last_purchase']]

    price_mean = cust_group['price'].mean().reset_index()
    price_mean.rename(columns={'price':'customer_past_purchase_price_mean'},inplace=True)
    price_mean = price_mean[['customer_id','customer_past_purchase_price_mean']]

    price_std = cust_group['price'].std().reset_index()
    price_std.rename(columns={'price':'customer_past_purchase_price_std'},inplace=True)
    price_std = price_std[['customer_id','customer_past_purchase_price_std']]

    out = out.merge(time_elapsed_last_purchase,on='customer_id')
    out = out.merge(num_purchase,on='customer_id')
    out = out.merge(price_mean,on='customer_id')

    return out

def article_customer_feature_engineering(art_df,cust_df,tran_df,selected_features,postfix='_count'):
    df = tran_df.merge(art_df,on='article_id')
    norm = pickle.load(open(os.path.join(conf['selected_feature_dir'],'norm.p'),'rb'))
    for f in selected_features:
        print_subheader(f)
        count = pd.read_csv(os.path.join(conf['selected_feature_dir'],f+postfix+'.csv'))
        df = df.merge(count,on=['customer_id',f])
    return df[['customer_id','article_id']+[f+postfix for f in selected_features]]

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def construct_ground_true_df(df):
    gtdf = df.groupby('customer_id')['article_id'].apply(lambda x: x.tolist()).reset_index()
    gtdf.columns = ['customer_id','ground_truth']
    return gtdf

def construct_pos_df(tran_df,art_df,cust_df,art_cust_df,features):
    out = tran_df.merge(art_cust_df,on=['article_id','customer_id'])
    out = out.merge(art_df,on='article_id')
    out = out.merge(cust_df,on='customer_id')
    return out[['article_id','customer_id']+features]

def construct_neg_df(tran_df,art_df,cust_df,art_cust_df,features):
    out = tran_df['article_id'].sample(frac=1).to_frame().reset_index().drop(columns=['index'])
    out['customer_id'] = tran_df['customer_id']
    out = out.merge(art_cust_df,on=['article_id','customer_id'],how='outer')
    out = out.merge(art_df,on='article_id')
    out = out.merge(cust_df,on='customer_id')
    out = out.fillna(0.)
    return out[['article_id','customer_id']+features]

def construct_val_df(tran_df,art_df,cust_df,art_cust_df,features,val_dir):
    cust_ids = tran_df.customer_id.unique().tolist()
    for cust_id in tqdm(cust_ids):
        art_ids = art_df.article_id.unique().tolist()
        data = [(cust_id,art_id) for art_id in art_ids]
        out = pd.DataFrame(data)
        out.columns = ['customer_id','article_id']
        out = out.merge(art_cust_df[art_cust_df.customer_id==cust_id],on=['article_id','customer_id'])
        #out = out.merge(art_df,on='article_id')
        #out = out.merge(cust_df,on='customer_id')
        out = out.fillna(0.)
        out.to_csv(os.path.join(val_dir,cust_id+'.csv'),index=False)

def save(obj,path):
    obj.to_csv(path,index=False)

def run(conf,args):

    print_header('Read csv files')
    tran_df = pd.read_csv(conf['tran_csv_path'])
    art_df = pd.read_csv(conf['art_csv_path'])
    cust_df = pd.read_csv(conf['cust_csv_path'])

    print_header('Train test split')
    tran_trn_df,tran_val_df = train_test_split(tran_df)
    save(tran_val_df,os.path.join(args.out_dir,'tran_val_df.csv'))
    
    print_header('Construct ground truth df')
    gt_df = construct_ground_true_df(tran_val_df)
    
    print_header('Feature engineering')

    art_trn_df_path = os.path.join(args.out_dir,'art_trn_df.csv')
    if not os.path.exists(art_trn_df_path):
        art_trn_df = article_feature_engineering(art_df,tran_trn_df)
        save(art_trn_df,art_trn_df_path)
    else:
        art_trn_df = pd.read_csv(art_trn_df_path)
    
    cust_trn_df_path = os.path.join(args.out_dir,'cust_trn_df.csv')
    if not os.path.exists(cust_trn_df_path):
        cust_trn_df = customer_feature_engineering(cust_df,tran_trn_df)
        save(cust_trn_df,cust_trn_df_path)
    else:
        cust_trn_df = pd.read_csv(cust_trn_df_path)

    art_cust_trn_df_path = os.path.join(args.out_dir,'art_cust_trn_df.csv')
    if not os.path.exists(art_cust_trn_df_path):
        art_cust_trn_df = art_customer_feature_engineering(art_cust_df,tran_trn_df)
        save(art_cust_trn_df,art_cust_trn_df_path)
    else:
        art_cust_trn_df = pd.read_csv(art_cust_trn_df_path)

    #print_header('Construct positive dataframe')
    selected_features = [f+'_count' for f in conf['article_customer_features']] + conf['article_features'] + conf['customer_features']
    #pos_df = construct_pos_df(tran_trn_df,art_trn_df,cust_trn_df,art_cust_trn_df,selected_features)
    #save(pos_df,os.path.join(args.out_dir,'pos_df.csv'))
    
    #print_header('Construct negative dataframe')
    #neg_df = construct_neg_df(tran_trn_df,art_trn_df,cust_trn_df,art_cust_trn_df,selected_features)
    #save(neg_df,os.path.join(args.out_dir,'neg_df.csv'))

    print_header('Construct validation dataframe')
    construct_val_df(tran_val_df,art_trn_df,cust_trn_df,art_cust_trn_df,selected_features,'storage/preprocessing/220306_baseline/val_dir/')

if __name__ == '__main__':
    args = parse_arguments()
    run(conf,args)
