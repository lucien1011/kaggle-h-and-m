import pandas as pd

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
