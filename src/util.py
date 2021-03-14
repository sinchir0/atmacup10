import os, random

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_log_error
from sklearn.decomposition import PCA

def print_head_and_shape(df:pd.DataFrame(),name: str):
    print(name)
    print(df.shape)
    display(df.head())

def seed_everything(seed=42, use_torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if use_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_non_overlapping(train: pd.DataFrame, test: pd.DataFrame, col: str):
    """train/testにしか出てこない値を調べる"""
    only_in_train = set(train[col].unique()) - set(test[col].unique())
    only_in_test = set(test[col].unique()) - set(train[col].unique())
    non_overlapping = only_in_train.union(only_in_test)
    return non_overlapping

def replace_non_overlap_val_to_missing(train: pd.DataFrame, test: pd.DataFrame, columns: list):
    train_ = train.copy()
    test_ = test.copy()
    
    for column in columns:
        non_overlapping = get_non_overlapping(train, test, column)
        if train[column].dtype == np.dtype("O"):
            # dtypeがobjectなら欠損は'missing' クラスにする
            train_[column] = train[column].fillna("missing")
            test_[column] = test[column].fillna("missing")
            train_[column] = train_[column].map(lambda x: x if x not in non_overlapping else "other")
            test_[column] = test_[column].map(lambda x: x if x not in non_overlapping else "other")
        else:
            # dtypeがint/floatなら欠損は'-1'とする
            train_[column] = train[column].fillna(-1)
            test_[column] = test[column].fillna(-1)
            train_[column] = train_[column].map(lambda x: x if x not in non_overlapping else -1)
            test_[column] = test_[column].map(lambda x: x if x not in non_overlapping else -1)
                
    return train_, test_

class PseudoLabeling():
    def __init__(self, 
                 train_df: pd.DataFrame, 
                 test_df: pd.DataFrame, 
                 target_df: pd.DataFrame(), 
                 pred_df: pd.DataFrame(),
                 target_name: str
                ):
        self.train_df = train_df
        self.test_df = test_df
        self.target_df = target_df
        self.pred_df = pred_df
        self.target_name = target_name
        
    def replace_proba2bin(self):
        '''分類問題の場合はこれを通す'''
        replace_pred_df = self.pred_df.copy()
        for col in self.target_df.columns.tolist():
            tar_mean = self.target_df[col].mean()
            # target_dfの1,0の割合をthrで取得
            thr = replace_pred_df[col].quantile(1 - tar_mean)
            # trainと同じ割合になるようpred_dfをprobaからbinに変換
            replace_pred_df[col] = np.where(self.pred_df[col] > thr, 1, 0)
            
        return replace_pred_df
    
    def merge(self, pred_df=None):
        train_test = pd.concat([self.train_df, self.test_df], axis=0, sort=False).reset_index(drop=True)
        if pred_df:
            train_test_target = pd.concat([self.target_df, pred_df], axis=0, sort=False).reset_index(drop=True)
        else:
            train_test_target = pd.concat([self.target_df, self.pred_df], axis=0, sort=False).reset_index(drop=True)
        
        return train_test, train_test_target
    
    def make_pseudo_data(self):
        pseudo_data = self.test_df.copy()
        pseudo_data[self.target_name] = self.pred_df
        return pseudo_data
    
def inf2nan(sr: pd.Series()) -> pd.Series:
     return sr.replace([np.inf, -np.inf], np.nan)

def use_pca(df: pd.DataFrame(), comp_num: int) -> pd.DataFrame():
    pca = PCA(n_components = comp_num)
    pca_res = pca.fit_transform(df.values)
    pca_res_df = pd.DataFrame(pca_res, columns=[f'pca_{i}' for i in range(comp_num)])
    # all_emb_df_256 = pd.concat((all_emb_df['object_id'], emb_df_256), axis=1)
    return pca_res_df