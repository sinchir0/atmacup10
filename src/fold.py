import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def add_fold_kf(df: pd.DataFrame(), fold_num:int) -> pd.DataFrame():
    '''KFoldのfold_numberを追加したdfを返す'''

    result = df.copy()
    kf = KFold(n_splits=fold_num)
    for f, (_, v_idx) in enumerate(kf.split(df)):
        result.loc[v_idx, 'fold'] = int(f)

    result['fold'] = result['fold'].astype('int')

    return result

def add_fold_bin_skf(df: pd.DataFrame(), target_sr: pd.Series, fold_num: int, bins: list) -> pd.DataFrame():
    '''targetをbin化した数値に対して、StratifiedKFoldのfold_numberを追加したdfを返す'''
    
    result = df.copy()
    
    target_sr_bin = pd.cut(target_sr, bins, labels=False, include_lowest=True)
    
    skf = StratifiedKFold(n_splits=fold_num)
    for fold, (_, val_idx) in enumerate(skf.split(df,target_sr_bin)):
        result.loc[val_idx, 'fold'] = int(fold)

    result['fold'] = result['fold'].astype('int')
    
    return result

def add_fold_skf(df: pd.DataFrame(), target_sr:pd.Series, fold_num:int) -> pd.DataFrame():
    '''StratifiedKFoldのfold_numberを追加したdfを返す'''
    
    result = df.copy()

    skf = StratifiedKFold(n_splits=fold_num)
    for fold, (_, val_idx) in enumerate(skf.split(df,target_sr)):
        result.loc[val_idx, 'fold'] = int(fold)

    result['fold'] = result['fold'].astype('int')

    return result

def add_fold_gkf(df: pd.DataFrame(), target_sr: pd.Series, fold_num: int, shuffle=True, seed=33) -> pd.DataFrame():
    '''GroupKFoldのfold_numberを追加したdfを返す'''
    
    result = df.copy()
    
    unique_target = target_sr.unique()

    kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for fold, (trn_gr_idx, val_gr_idx) in enumerate(kf.split(unique_target)):
        trn_gr, var_gr = unique_target[trn_gr_idx], unique_target[val_gr_idx]
        
        val_idx = target_sr.isin(var_gr)
        result.loc[val_idx, 'fold'] = fold
        
    result['fold'] = result['fold'].astype('int')
    
    return result

def add_fold_tss(df: pd.DataFrame(), time_sr:pd.Series, fold_num=5) -> pd.DataFrame():
    '''TimeSeriesSplitのfold_numberを追加したdfを返す'''
    
    result = df.copy()
    
    # Xヶ月毎
    fold1time = '20XX-XX-XX'
    fold2time = '20XX-XX-XX'
    fold3time = '20XX-XX-XX'
    fold4time = '20XX-XX-XX'

    for fold in range(fold_num):
        if fold == 0:
            val_idx = df[time_sr < fold1time].index
        elif fold == 1:
            val_idx = df[(fold1time <= time_sr ) & (time_sr < fold2time)].index
        elif fold == 2:
            # train_idx = train[train['imp_at'] < fold3time].index
            val_idx = df[(fold2time <= time_sr ) & (time_sr < fold3time)].index
        elif fold == 3:
            # train_idx = train[train['imp_at'] < fold4time].index
            val_idx = df[(fold3time <= time_sr ) & (time_sr < fold4time)].index
        elif fold == 4:
            # train_idx = train[train['imp_at'] < fold5time].index
            val_idx = df[fold4time <= time_sr].index
        
        # foldを追加
        result.loc[val_idx, 'fold'] = int(fold)
        
    result['fold'] = result['fold'].astype('int')
    
    return result

def add_fold_mskf(df, target_df, fold_num: int):
    '''MultilabelStratifiedKFoldのfold_numberを追加したdfを返す'''
    result = df.copy()

    mskf = MultilabelStratifiedKFold(n_splits=fold_num)

    for f, (_, v_idx) in enumerate(mskf.split(X=df, y=target_df)):
        result.loc[v_idx, 'fold'] = f

    result['fold'] = result['fold'].astype(int)

    return result

def add_fold_org(df: pd.DataFrame(), fold_num:int) -> pd.DataFrame():
    return NotImplementedError
