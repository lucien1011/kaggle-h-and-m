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
                    'customer_time_elpased_last_purchase','customer_num_purchase', 'customer_past_purchase_price_mean', 'customer_past_purchase_price_std',
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
