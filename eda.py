import numpy as np
import pandas as pd
from tqdm import tqdm

from metric import calculate_score

conf = dict(
        trans_csv_path='storage/transactions_train.csv',
        item_pairs_path='storage/item_pairs.npy'
        )

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split ]
    valid_df = df[ df.t_dat > ts_split ]
    valid_gt_df = valid_df.groupby('customer_id').article_id.apply(list).reset_index()
    return train_df,valid_df,valid_gt_df

def first_purchase(df,k=12):
    first_purchase = df.groupby(['customer_id'])['t_dat'].min().reset_index()
    first_purchase.columns = ['customer_id','first_purchase_date']
    tmp = df.merge(first_purchase,on='customer_id')
    tmp = tmp[tmp['t_dat']==tmp['first_purchase_date']]
    return tmp.article_id.value_counts()[:k].index.tolist()

def pair_purchase(exist_cust_df,item_pairs):
    exist_cust_df['article_id2'] = exist_cust_df.article_id.map(item_pairs)
    exist_cust_df2 = exist_cust_df[['customer_id','article_id2']].copy()
    exist_cust_df2 = exist_cust_df2.loc[exist_cust_df2.article_id2.notnull()]
    exist_cust_df2 = exist_cust_df2.drop_duplicates(['customer_id','article_id2'])
    exist_cust_df2 = exist_cust_df2.rename({'article_id2':'article_id'},axis=1)

    exist_cust_df = exist_cust_df[['customer_id','article_id']]
    exist_cust_df = pd.concat([exist_cust_df,exist_cust_df2],axis=0,ignore_index=True)
    exist_cust_df.article_id = exist_cust_df.article_id.astype('int32')
    exist_cust_df = exist_cust_df.drop_duplicates(['customer_id','article_id'])

    exist_cust_df = pd.DataFrame( exist_cust_df.groupby('customer_id').article_id.apply(lambda x: x.tolist()).reset_index() )
    exist_cust_df = exist_cust_df.rename(columns={'article_id':'prediction'})
    return exist_cust_df

def construct_item_pairs(df):
    vc = df.article_id.value_counts()
    pairs = {}
    for j,i in enumerate(tqdm(vc.index.values)):
        USERS = df.loc[df.article_id==i.item(),'customer_id'].unique()
        vc2 = df.loc[(df.customer_id.isin(USERS))&(df.article_id!=i.item()),'article_id'].value_counts()
        pairs[i.item()] = vc2.index[0]
    return pairs

def construct_pred_df(trn_df,val_df,item_pairs):
    exist_cust_ids = trn_df.customer_id.unique().tolist()
    new_cust_df = val_df[val_df['customer_id'].isin(exist_cust_ids)]
    exist_cust_df = val_df[~val_df['customer_id'].isin(exist_cust_ids)]

    first_purchase_items = first_purchase(trn_df)
    new_cust_df = pd.DataFrame(new_cust_df.customer_id.unique().tolist())
    new_cust_df.columns = ['customer_id']
    new_cust_df['prediction'] = len(new_cust_df) * [first_purchase_items]
   
    exist_cust_df = pair_purchase(exist_cust_df,item_pairs)
    
    val_df = pd.concat([new_cust_df,exist_cust_df])
    return val_df

def construct_true_df(df):
    gtdf = df.groupby('customer_id')['article_id'].apply(lambda x: x.tolist()).reset_index()
    gtdf.columns = ['customer_id','ground_truth']
    return gtdf

def run(conf):
    
    print('Read csv files')
    trans = pd.read_csv(conf['trans_csv_path'])

    print('Train test split')
    trn_df,val_df,val_gt_df = train_test_split(trans)
    
    print('Construct item pairs')
    item_pairs = construct_item_pairs(trn_df) 
    
    print('Local cross validation')
    cv_df = construct_true_df(val_df)
    pred_df = construct_pred_df(trn_df,val_df,item_pairs)
    cv_df = cv_df.merge(pred_df,on='customer_id')
    calculate_score(cv_df['ground_truth'].tolist(),cv_df['prediction'].tolist())
    
if __name__ == '__main__':
    run(conf)
