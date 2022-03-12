import numpy as np
import pandas as pd

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
        past_purchase_features=[
                'product_group_name',
                'colour_group_code',
            ],
        train_args=dict(
            device='cuda',
            bs=16,
            print_every=100,
            ),
    )

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

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
    #out = out.merge(price_std,on='customer_id')

    return out

def past_purchase_feature_engineering(art_df,cust_df,tran_df,selected_features):
    df = tran_df.merge(art_df,on='article_id')
    norm = pickle.load(open(os.path.join(conf['selected_feature_dir'],'norm.p'),'rb'))
    for f in selected_features:
        count = pickle.load(open(os.path.join(conf['selected_feature_dir'],f+'_count.p'),'rb'))
        df[f] = df.apply(lambda x: count[row['customer_id'],row[f]] / norm[row['customer_id'],axis=1)
    return df[['customer_id','article_id']+selected_features]

def train_test_split(df):
    ts_split = pd.to_datetime('2020-09-15')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    train_df = df[df.t_dat <= ts_split]
    valid_df = df[df.t_dat > ts_split]
    return train_df,valid_df

def construct_trn_dataset(tran_trn_df,cust_trn_df,art_trn_df,past_purchase_df,conf):
    from dataset import TorchDataset 
    return TorchDataset(
            tran_trn_df['customer_id'].tolist(),
            tran_trn_df['article_id'].tolist(),
            cust_trn_df,
            art_trn_df,
            past_purchase_df,
            conf['customer_features'],
            conf['article_features'],
            conf['past_purchase_features'],
            )

def construct_val_dataset(tran_val_df,cust_trn_df,art_trn_df,conf):
    from dataset import TorchDatasetValid
    val_dataset = TorchDatasetValid(
            tran_val_df['customer_id'].unique().tolist(),
            cust_trn_df,
            art_trn_df,
            conf['customer_features'],
            conf['article_features'],
            )
    return val_dataset

def construct_model():
    from model import MLP
    return MLP()

def train(trn_dataset,val_dataset,gt_df,args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model = construct_model()
    model.to(args['device'])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        
    loss_fn = nn.BCELoss()
    
    trn_loader = DataLoader(trn_dataset,batch_size=args['bs'],shuffle=True,collate_fn=trn_dataset.inbatch_sampling) 
    for idx, batch in enumerate(tqdm(trn_loader)):
        x = batch['x'].to(args['device'],dtype=torch.float)
        target = batch['target'].to(args['device'],dtype=torch.float)
        logit = model(x)
        loss = loss_fn(logit,target)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        if idx % args['print_every']==0 and idx != 0:
            tqdm.write(f"Training loss after {idx:04d} training steps: {loss.item()}")
            cross_validation(val_dataset,gt_df,model)

def cross_validation(val_dataset,gt_df,model,device='cuda',k=12,nsample=1000):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from metric import calculate_score
    val_loader = DataLoader(val_dataset,shuffle=True)
    pred_map = {}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            if idx > nsample: break
            x = batch['x'].to(device,dtype=torch.float)
            logit = model(x)
            for cid in batch['customer_id']:
                pred_map[batch['customer_id'][0]] = logit.cpu().numpy()
    cv = []
    for cid,pred in pred_map.items():
        pred_items = [val_dataset.article_ids[idx] for idx in np.argsort(pred)[:k]]
        cv.append((cid,pred_items))
    cv_df = pd.DataFrame(cv)
    cv_df.columns = ['customer_id','prediction']
    cv_df = cv_df.merge(gt_df,on='customer_id')
    cv_df.dropna(inplace=True)
    calculate_score(cv_df['ground_truth'].tolist(),cv_df['prediction'].tolist()) 

def construct_ground_true_df(df):
    gtdf = df.groupby('customer_id')['article_id'].apply(lambda x: x.tolist()).reset_index()
    gtdf.columns = ['customer_id','ground_truth']
    return gtdf

def run(conf):

    print_header('Read csv files')
    tran_df = pd.read_csv(conf['tran_csv_path'])
    art_df = pd.read_csv(conf['art_csv_path'])
    cust_df = pd.read_csv(conf['cust_csv_path'])

    print_header('Train test split')
    tran_trn_df,tran_val_df = train_test_split(tran_df)
    
    print_header('Construct ground truth df')
    gt_df = construct_ground_true_df(tran_val_df)
    
    print_header('Feature engineering')
    #art_trn_df = article_feature_engineering(art_df,tran_trn_df)
    #cust_trn_df = customer_feature_engineering(cust_df,tran_trn_df)
    past_purchase_trn_df = past_purchase_feature_engineering(art_df,cust_df,tran_trn_df,conf['past_purchase_features'])

    print_header('Construct dataset')
    trn_dataset = construct_trn_dataset(tran_trn_df,cust_trn_df,art_trn_df,past_purchase_df,conf)
    #val_dataset = construct_val_dataset(tran_val_df,cust_trn_df,art_trn_df,conf)

    #print_header('Train')
    #train(trn_dataset,val_dataset,gt_df,conf['train_args'])

if __name__ == '__main__':
    run(conf)
