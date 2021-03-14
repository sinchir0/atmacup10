import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score

def make_dist_oof_pred(oof_df:pd.DataFrame(), sub_df:pd.DataFrame(), OUTPUT_DIR:str):
    sns.distplot(oof_df.values, color='red', label='oof')
    sns.distplot(sub_df.values, color='black', label='pred')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/ood_pred_dist.png")

def make_roc_auc_curve(train_target_df:pd.DataFrame(), oof_df:pd.DataFrame(),OUTPUT_DIR:str, TARGET_IDS:str, TARGET_CATEGORIES:str):
    cat_id2name = dict(zip(TARGET_IDS, TARGET_CATEGORIES))

    fig, ax = plt.subplots(figsize=(8, 8))

    for c in TARGET_IDS:
        fpr, tpr, _ = roc_curve(train_target_df[c], oof_df[c])
        ax.plot(fpr, tpr, label=cat_id2name[c])
    ax.legend()
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), '--', color='grey')
    plt.savefig(f"{OUTPUT_DIR}/roc_auc_curve.png")