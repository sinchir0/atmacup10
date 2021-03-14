import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_log_error

import wandb

def calc_macro_auc(train_target_df:pd.DataFrame(), oof_df: pd.DataFrame(), order_list:list) -> float:
    y_true = train_target_df.reset_index(drop=True)[order_list].values
    y_pred = oof_df.values
    score = roc_auc_score(y_true, y_pred, average='macro')
    print(f'macro_auc : {score}')
    wandb.log({"macro_auc": score})
    return score

def calc_rmlse(true, pred) -> float:
    return (mean_squared_log_error(true, pred) ** .5)