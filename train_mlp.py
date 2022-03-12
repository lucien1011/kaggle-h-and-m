import os
import numpy as np
import pandas as pd

conf = dict(
        input_dir='storage/preprocessing/220306_baseline/',
        selected_features=[
            'product_group_name_count', 'product_type_name_count', 
            'graphical_appearance_name_count', 'perceived_colour_value_name_count', 'colour_group_code_count', 
            'index_name_count', 'index_group_name_count', 
            'section_name_count', 'department_name_count', 
            #'num_purchase', 'time_elpased_last_purchase', 'price_mean', 'customer_time_elpased_last_purchase', 'customer_num_purchase', 'customer_past_purchase_price_mean',
            ],
        train_args=dict(
            train_dir='storage/training/220306_baseline/',
            device='cuda',
            bs=128,
            print_every=100,
            ),
    )

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

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
    trn_loader = DataLoader(trn_dataset,batch_size=args['bs'],shuffle=True)
    tr_loss = 0.
    for idx, batch in enumerate(tqdm(trn_loader)):
        x = batch['x'].to(args['device'],dtype=torch.float)
        target = batch['target'].to(args['device'],dtype=torch.float)
        logit = model(x)
        loss = loss_fn(logit,target)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        if idx % args['print_every']==0 and idx != 0:
            tqdm.write(f"Training loss after {idx:04d} training steps: {tr_loss}")
            tr_loss = 0.
            #cross_validation(val_dataset,gt_df,model)
        else:
            tr_loss += loss.item() / args['print_every']
    torch.save(model.state_dict(),os.path.join(args['train_dir'],'model.pt'))

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

def construct_ground_truth_df(conf):
    df = pd.read_csv(conf['input_dir']+'tran_val_df.csv')
    gtdf = df.groupby('customer_id')['article_id'].apply(lambda x: x.tolist()).reset_index()
    gtdf.columns = ['customer_id','ground_truth']
    return gtdf

def construct_trn_dataset(conf):
    from dataset import BaselineDataset 
    pos_df = pd.read_csv(conf['input_dir']+'pos_df_lite.csv')
    pos_df['label'] = 1
    neg_df = pd.read_csv(conf['input_dir']+'neg_df_lite.csv')
    neg_df['label'] = 0
    trn_df = pd.concat([pos_df,neg_df]).fillna(0.)
    return BaselineDataset(trn_df,conf['selected_features'])

def construct_val_dataset(conf):
    from dataset import BaselineDatasetValid
    art_df = pd.read_csv(conf['input_dir']+'art_trn_df.csv')
    cust_df = pd.read_csv(conf['input_dir']+'cust_trn_df.csv')
    art_cust_df = pd.read_csv(conf['input_dir']+'art_cust_trn_df.csv')
    tran_val_df = pd.read_csv(conf['input_dir']+'tran_val_df.csv')
    return BaselineDatasetValid(tran_val_df.customer_id.unique().tolist(),cust_df,art_df,art_cust_df,conf['selected_features'])

def run(conf):

    print_header('Construct dataset')
    trn_dataset = construct_trn_dataset(conf)
    val_dataset = construct_val_dataset(conf)
    gtdf = construct_ground_truth_df(conf)

    print_header('Train')
    train(trn_dataset,val_dataset,gtdf,conf['train_args'])

if __name__ == '__main__':
    run(conf)
