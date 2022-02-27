import numpy as np
import pandas as pd

from metric import calculate_score

conf = dict(
        trans_csv_path='storage/transactions_train.csv',
        arts_csv_path='storage/articles.csv',
        custs_csv_path='storage/customers.csv',
        )

def customers_feature_engineering(df):
    out = df[['age','FN','fashion_news_frequency','club_member_status','Active']]
    out['FN'] = out['FN'].fillna(-1).astype('category').cat.codes
    out['Active'] = out['Active'].fillna(-1).astype('category').cat.codes
    out['club_member_status'] = out['club_member_status'].fillna(-1).astype('category').cat.codes
    out['fashion_news_frequency'] = out['fashion_news_frequency'].fillna(-1).astype('category').cat.codes
    return out

def articles_feature_engineering(df):
    out = df['product_type_name','graphical_appearance_name','colour_group_name',]
    out['product_type_name'] = out['product_type_name'].fillna(-1).astype('category').cat.codes
    out['graphical_appearance_name'] = out['graphical_appearance_name'].fillna(-1).astype('category').cat.codes
    out['colour_group_name'] = out['colour_group_name'].fillna(-1).astype('category').cat.codes
    return out

def transactions_feature_engineering(df):
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    return df 
    
def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split ]
    valid_df = df[ df.t_dat > ts_split ]
    valid_gt_df = valid_df.groupby('customer_id').article_id.apply(list).reset_index()
    return train_df,valid_df,valid_gt_df

def predict_topk_popular_articles(df,k=12):
    topk = df.loc[df.t_dat < pd.to_datetime('2020-09-16')]['article_id'].value_counts()[:k].index.tolist()
    return topk

def construct_true_df(df):
    gtdf = df.groupby('customer_id')['article_id'].apply(lambda x: x.tolist()).reset_index()
    gtdf.columns = ['customer_id','ground_truth']
    return gtdf

def construct_pred_df(trn_df,val_df):
    topk = predict_topk_popular_articles(trn_df)
    val_df['prediction'] = len(val_df) * [topk]
    return val_df

def run(conf):

    print('Read csv files')
    trans = pd.read_csv(conf['trans_csv_path'])

    print('Feature engineering')
    trans = transactions_feature_engineering(trans)
    trn_df,val_df,_ = train_test_split(trans)

    print('Local cross validation')
    cv_df = construct_true_df(val_df)
    cv_df = construct_pred_df(trn_df,cv_df)
    calculate_score(cv_df['ground_truth'].tolist(),cv_df['prediction'].tolist())

if __name__ == '__main__':
    run(conf)
