import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TransactionDataset(object):
    def __init__(self,transaction_df,article_df,customer_df):
        self.transaction_df = transaction_df
        self.article_df = article_df
        self.customer_df = customer_df

    def get_pos_df(self):
        pos = self.transaction_df[['article_id','customer_id']]
        pos = pos.merge(self.article_df,on='article_id')
        pos = pos.merge(self.customer_df,on='customer_id')
        pos['label'] = 1
        return pos

    def get_neg_df(self):
        neg = self.transaction_df['article_id'].sample(frac=1).to_frame()
        neg['customer_id'] = self.transaction_df['customer_id']
        neg = neg.merge(self.article_df,on='article_id')
        neg = neg.merge(self.customer_df,on='customer_id')
        neg['label'] = 0
        return neg

    def get_df(self):
        pos_df = self.get_pos_df()
        neg_df = self.get_neg_df()
        return pd.concat([pos_df,neg_df])

class TorchDataset(Dataset):
    def __init__(self,
            customer_ids,article_ids,
            customer_df,article_df,
            article_customer_df,
            customer_features,article_features,
            article_customer_features,
            ):

        self.article_ids = article_ids
        self.article_df = article_df
        self.article_features = article_features
        
        self.customer_ids = customer_ids
        self.customer_df = customer_df
        self.customer_features = customer_features

        self.article_customer_df = article_customer_df
        self.article_customer_features = article_customer_features

        assert len(self.customer_ids) == len(self.article_ids)

    def __len__(self):
        return len(self.customer_ids)

    def __getitem__(self,index):

        customer_id = self.customer_ids[index]
        customer_arr = self.customer_df[self.customer_df['customer_id']==customer_id][self.customer_features].to_numpy().squeeze()

        article_id = self.article_ids[index]
        article_arr = self.article_df[self.article_df['article_id']==article_id][self.article_features].to_numpy().squeeze()

        article_customer_arr = self.article_customer_df[(self.article_customer_df['customer_id']==customer_id) & (self.article_customer_df['article_id']==article_id)].to_numpy().squeeze()
        
        return dict(
                index=index,
                customer_id=customer_id,
                customer_arr=torch.tensor(customer_arr),
                article_id=article_id,
                article_arr=torch.tensor(article_arr),
                article_customer_arr=torch.tensor(article_customer_arr),
                )
    
    @staticmethod
    def inbatch_sampling(batch):
        n = len(batch)
        customer_arr = torch.cat([batch[i]['customer_arr'].unsqueeze(0) for i in range(n)])
        article_arr = torch.cat([batch[i]['article_arr'].unsqueeze(0) for i in range(n)])
        article_customer_arr = torch.cat([batch[i]['article_customer_arr'].unsqueeze(0) for i in range(n)])

        neg_customer_arr = torch.cat([torch.index_select(customer_arr, 0, torch.tensor([i for i in range(n) if i != j])) for j in range(n)])
        neg_article_arr = torch.cat([torch.index_select(article_arr, 0, torch.tensor([i for i in range(n) if i != j])) for j in range(n)])
        neg_article_customer_arr = torch.cat([torch.index_select(article_customer_arr, 0, torch.tensor([i for i in range(n) if i != j])) for j in range(n)])

        pos_arr = torch.cat([customer_arr,article_arr,article_customer_arr],axis=1)
        pos_label = torch.ones(len(pos_arr))
        neg_arr = torch.cat([neg_customer_arr,neg_article_arr,neg_article_customer_arr],axis=1)
        neg_label = torch.zeros(len(neg_arr))

        return dict(
                x=torch.cat([pos_arr,neg_arr]),
                target=torch.cat([pos_label,neg_label]),
                )

class TorchDatasetValid(Dataset):
    def __init__(self,
            customer_ids,
            customer_df,article_df,past_purchase_df,
            customer_features,article_features,
            ):

        self.article_ids = list(article_df.article_id.unique())
        self.article_df = article_df
        self.article_features = article_features
        self.article_arr = self.article_df[self.article_features].to_numpy().squeeze()
        
        self.customer_ids = customer_ids
        self.customer_df = customer_df
        self.customer_features = customer_features
        
    def __len__(self):
        return len(self.customer_ids)

    def __getitem__(self,index):

        customer_id = self.customer_ids[index]
        customer_arr = self.customer_df[self.customer_df['customer_id']==customer_id][self.customer_features].to_numpy().squeeze()
        if len(customer_arr) == 0: customer_arr = self.customer_df[self.customer_features].mean(axis=0).to_numpy().squeeze()

        x = np.broadcast_to(customer_arr,(len(self.article_ids),customer_arr.shape[0]))
        x = np.concatenate([x,self.article_arr],axis=1)

        return dict(
                index=index,
                customer_id=customer_id,
                x=x,
                customer_arr=torch.tensor(customer_arr),
                article_arr=torch.tensor(self.article_arr),
                )

class BaselineDataset(Dataset):
    def __init__(self,data,selected_features,):
        self.data = data
        self.selected_features = selected_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        tmp = self.data.iloc[index]
        customer_id = tmp.customer_id
        article_id = tmp.article_id
        x = tmp[self.selected_features]
        target = tmp.label
        return dict(
                index=index,
                customer_id=customer_id,
                article_id=article_id,
                x=torch.tensor(x),
                target=torch.tensor(target),
                )

class BaselineDatasetValid(Dataset):
    def __init__(self,cust_ids,cust_df,art_df,art_cust_df,selected_features):
        self.cust_ids = cust_ids
        self.cust_df = cust_df
        self.art_df = art_df
        self.art_cust_df = art_cust_df
        self.selected_features = selected_features

    def __len__(self):
        return len(self.cust_ids)

    def __getitem__(self,index):
        customer_id = self.cust_ids[index]
        tmp = self.art_df.copy()
        tmp['customer_id'] = customer_id
        tmp = tmp.merge(self.art_cust_df,on=['customer_id','article_id'])
        x = tmp[self.selected_features].to_numpy()
        return dict(
                index=index,
                customer_id=customer_id,
                x=torch.tensor(x),
                )
