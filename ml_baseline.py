import numpy as np
import pandas as pd

from dataset import TransactionDataset 

conf = dict(
        tran_csv_path='storage/transactions_train.csv',
        art_csv_path='storage/articles.csv',
        cust_csv_path='storage/customers.csv',
        train_args = dict(
                feat_names=[
                    'product_type_name','graphical_appearance_name', 'colour_group_name', 'FN',  'club_member_status', 'Active',
                    'time_elpased_last_purchase','num_purchase', 'price_mean', 'price_std',
                    'age', 'fashion_news_frequency',
                    ],
                cat_names=[
                    'product_type_name','graphical_appearance_name', 'colour_group_name', 'FN',  'club_member_status', 'Active',
                    ],
                label_name='label',
                param=dict(
                    num_leaves=31,
                    objective='binary',
                    metric='auc',
                    ),
                num_round=10,
                )
        )

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

def article_feature_engineering(art_df,tran_df,current_time='2020-09-15',):
    out = art_df[['article_id','product_type_name','graphical_appearance_name','colour_group_name',]]
    out['product_type_name'] = out['product_type_name'].fillna(-1).astype('category').cat.codes
    out['graphical_appearance_name'] = out['graphical_appearance_name'].fillna(-1).astype('category').cat.codes
    out['colour_group_name'] = out['colour_group_name'].fillna(-1).astype('category').cat.codes

    art_group = tran_df.groupby('article_id')
    
    num_purchase = art_group['t_dat'].count().reset_index()
    num_purchase = num_purchase.rename(columns={'t_dat':'num_purchase'})
    num_purchase = num_purchase[['article_id','num_purchase']]
    
    curr_time_stamp = pd.Timestamp(current_time)
    time_elapsed_last_purchase = art_group['t_dat'].max().reset_index()
    time_elapsed_last_purchase['time_elpased_last_purchase'] = time_elapsed_last_purchase['t_dat'].apply(lambda x: (curr_time_stamp-x).days)
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
    out = out.merge(price_std,on='article_id')
    return out

def customer_feature_engineering(cust_df,tran_df):
    out = cust_df[['customer_id','age','FN','fashion_news_frequency','club_member_status','Active']]
    out['FN'] = out['FN'].fillna(-1).astype('category').cat.codes
    out['Active'] = out['Active'].fillna(-1).astype('category').cat.codes
    out['club_member_status'] = out['club_member_status'].fillna(-1).astype('category').cat.codes
    out['fashion_news_frequency'] = out['fashion_news_frequency'].fillna(-1).astype('category').cat.codes
    return out

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def train(mode,args):
    if mode == 'lightgbm':
        train_lightgbm(**args)
    else:
        raise NotImplementedError

def train_lightgbm(trn_df,val_df,feat_names,cat_names,label_name,param,num_round):
    import lightgbm as lgb

    trn_data = lgb.Dataset(trn_df[feat_names], label=trn_df[label_name], categorical_feature=cat_names)
    val_data = lgb.Dataset(val_df[feat_names], label=val_df[label_name], categorical_feature=cat_names)
    
    bst = lgb.train(param, trn_data, num_round, valid_sets=[trn_data,val_data])

def run(conf):

    print_header('Read csv files')
    tran_df = pd.read_csv(conf['tran_csv_path'])
    art_df = pd.read_csv(conf['art_csv_path'])
    cust_df = pd.read_csv(conf['cust_csv_path'])

    print_header('Train test split')
    tran_trn_df,tran_val_df = train_test_split(tran_df)
    
    print_header('Feature engineering')
    art_trn_df = article_feature_engineering(art_df,tran_trn_df)
    cust_trn_df = customer_feature_engineering(cust_df,tran_trn_df)

    print_header('Construct dataset')
    trn_dataset = TransactionDataset(tran_trn_df,art_trn_df,cust_trn_df)
    trn_df = trn_dataset.get_df() 
    val_dataset = TransactionDataset(tran_val_df,art_trn_df,cust_trn_df)
    val_df = val_dataset.get_df()

    print_header('Train')
    conf['train_args']['trn_df'] = trn_df
    conf['train_args']['val_df'] = val_df
    train('lightgbm',conf['train_args'])

if __name__ == '__main__':
    run(conf)
