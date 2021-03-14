import pandas as pd
import numpy as np

from fold import add_fold_skf, add_fold_tss
from lgbm import LGBM

def add_target(
    train_feat_df: pd.DataFrame(),
    train_target_df:pd.DataFrame(),
    target: int
    ) -> pd.DataFrame():
    """pd.DataFrame()に予測するターゲットの列を追加する"""

    target_sr = train_target_df.reset_index(drop=False)[target].rename(f'tar_{target}')
    return pd.concat([train_feat_df,target_sr], axis=1)

def make_data_per_one_tar(
    train_feat_df: pd.DataFrame(),
    train_target_df: pd.DataFrame(), 
    target: int,
    fold_num: int,
    # time_sr: pd.Series() #for tss
    ) -> pd.DataFrame():
    """pd.DataFrame()にfoldの列と, 予測するターゲットの列を追加する"""

    # add kf skf fold
    train_feat_df_per_tar = add_fold_skf(
        df=train_feat_df, 
        target_sr=train_target_df[target], 
        fold_num=fold_num
        )

#     train_feat_df_per_tar = add_fold_tss(
#         df=train_feat_df, 
#         time_sr=time_sr,
#         fold_num=fold_num
#         )

    # add target
    train_lgbm = add_target(train_feat_df_per_tar, train_target_df, target)

    return train_lgbm

def make_many_oof(OUTPUT_DIR:str, TARGET_IDS: list) -> pd.DataFrame():
    oof = pd.DataFrame()
    for tar_col in TARGET_IDS:
        oof[tar_col] = np.load(f'{OUTPUT_DIR}/oof/oof_tar_{tar_col}.npy')
    oof.to_csv(f'{OUTPUT_DIR}/oof/oof.csv')

    return oof

def make_oof(model_oof, OUTPUT_DIR:str, TARGET: str) -> pd.DataFrame():
    oof = pd.DataFrame()
    oof[TARGET] = model_oof
    oof.to_csv(f'{OUTPUT_DIR}/oof/oof.csv')

    return oof

def make_oof_add_pre(model_oof, OUTPUT_DIR:str, TARGET: str, prefix: str) -> pd.DataFrame():
    oof = pd.DataFrame()
    oof[TARGET] = model_oof
    oof.to_csv(f'{OUTPUT_DIR}/oof/oof_{prefix}.csv')

    return oof

def make_sub_at_multi_column(sample_sub: pd.DataFrame(), TRIAL_NAME:str, OUTPUT_DIR:str, cv_score:float) -> pd.DataFrame():
    sub = pd.DataFrame()
    for tar_col in sample_sub:
        sub[tar_col] = np.load(f'{OUTPUT_DIR}/pred/pred_tar_{tar_col}.npy')
    sub.to_csv(f'{OUTPUT_DIR}/pred/pred_{TRIAL_NAME}_{cv_score:.4f}.csv',index=False)
    print(f'QUick Sub')
    print(f'sh sub.sh Users/atma38/my_pipeline/{OUTPUT_DIR[6:]}/pred/pred_{TRIAL_NAME}_{cv_score:.4f}.csv')

    return sub

def make_sub(sample_sub: pd.DataFrame(), TARGET_NAME: str, TRIAL_NAME:str, OUTPUT_DIR:str, cv_score:float) -> pd.DataFrame():
    sub = pd.DataFrame()
    sub[TARGET_NAME] = np.load(f'{OUTPUT_DIR}/pred/pred_{TARGET_NAME}.npy')
    sub.to_csv(f'{OUTPUT_DIR}/pred/pred_{TARGET_NAME}_{cv_score:.4f}.csv',index=False)
    print(f'QUick Sub')
    print(f'sh sub.sh {OUTPUT_DIR[6:]}/pred/pred_{TARGET_NAME}_{cv_score:.4f}.csv')

    return sub

def make_sub_for_split_model(sample_sub: pd.DataFrame(), pred, TARGET_NAME: str, TRIAL_NAME:str, OUTPUT_DIR:str, cv_score:float) -> pd.DataFrame():
    sub = pd.DataFrame()
    sub[TARGET_NAME] = pred
    sub.to_csv(f'{OUTPUT_DIR}/pred/pred_{TARGET_NAME}_{cv_score:.4f}.csv',index=False)
    print(f'QUick Sub')
    print(f'sh sub.sh {OUTPUT_DIR[6:]}/pred/pred_{TARGET_NAME}_{cv_score:.4f}.csv')

    return sub