import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
# LightGBMTunerを回して、最適なパラメータを1回分取り出す仕組みはあってもいいかも？
# import optuna.integration.lightgbm as lgb
from sklearn.metrics import roc_auc_score

import wandb

from score import calc_rmlse
from util import seed_everything

def get_score(true, pred):
    return calc_rmlse(true, pred)

class LGBM(object):

    def __init__(self, 
                 train: pd.DataFrame(), 
                 test: pd.DataFrame(), 
                 target: str, 
                 use_col: list, 
                 cat_col: list, 
                 cfg: dict, 
                 OUTPUT_DIR: str, 
                 DO_FIT=True, 
                 DO_SEED_AVE=False, 
                 DEBUG=False,
                 use_weight=False,
                 is_target_log=False,
                 use_pseudo_labeling=False,
                 pseudo_data=False
                ):
        self.train = train
        self.test = test
        self.target = f"{target}"
        self.use_col = use_col
        self.cat_col = cat_col
        self.params = cfg['params']
        self.fold_num = cfg['fold_num']
        self.seed = cfg['seed_num']
        self.output_dir = OUTPUT_DIR
        self.verbose_eval = cfg['verbose_eval']
        self.seed_ave_num = cfg['seed_ave_num']
        self.debug = DEBUG
        self.score_name = cfg['score_name']
        self.weight = use_weight
        self.is_target_log = is_target_log
        self.use_pseudo_labeling = use_pseudo_labeling
        self.pseudo_data = pseudo_data
        
        if self.debug:
            print('DEBUG is True')
            self.params['num_boost_round'] = 1
        if DO_FIT:
            self.oof, self.test_preds_all, self.models, self.score = self.fit(self.seed)
        if DO_SEED_AVE:
            self.oof, self.test_preds_all, self.models, self.score = self.seed_ave(self.seed_ave_num)

    def fit(self, seed):
        self.params.update({'random_state': seed})
        for col in self.cat_col:
            self.train[col] = self.train[col].astype('category')
            self.test[col] = self.test[col].astype('category')

        oof = np.zeros((len(self.train),))
        test_preds_all = np.zeros((len(self.test),))
        models = []
        for fold in range(self.fold_num):

            ### kf,skf
            valid_idx = self.train[(self.train.fold == fold)].index
            
            train_tmp = self.train[(self.train.fold != fold)].reset_index(drop=True)
            
            if self.use_pseudo_labeling:
                pseudo_data = self.pseudo_data[(self.pseudo_data.fold != fold)]
                
                print(pseudo_data.shape)
                
                train_tmp = pd.concat([train_tmp, pseudo_data],axis=0).reset_index(drop=True)
                for col in self.cat_col:
                    train_tmp[col] = train_tmp[col].astype('category')
                print(train_tmp.shape)
            
            # add tmp for 52_clip_high_likes_to_1000
            # train_tmp[self.target] = train_tmp[self.target].clip(0, 1000)
            # print(train_tmp[self.target])
            
            # add tmp for 53_delete_high_likes_in_folds
            #print('for 53_delete_high_likes_in_folds')
            #print(train_tmp.shape)
            #train_tmp = train_tmp[train_tmp[self.target] < 1200]
            #print(train_tmp.shape)
            
            valid_tmp = self.train[(self.train.fold == fold)].reset_index(drop=True)
            
            ### tss
            # if fold == self.fold_num - 1: # 4
            #    break
                
            # valid_idx = self.train[(self.train.fold > fold)].index
            #### use only last fold
            # valid_idx = self.train[(self.train.fold == 4)].index
            # print(valid_idx)
            
            # train_tmp = self.train[(self.train.fold <= fold)].reset_index(drop=True)
            # valid_tmp = self.train[(self.train.fold > fold)].reset_index(drop=True)
            ### use only last fold
            # valid_tmp = self.train[(self.train.fold == 4)].reset_index(drop=True)

            print(f"--------------FOLDS : {fold} --------------")

            X_train, y_train = train_tmp[self.use_col], train_tmp[self.target]
            X_valid, y_valid = valid_tmp[self.use_col], valid_tmp[self.target]
            
            if self.is_target_log:
                y_train = np.log1p(y_train)
                y_valid = np.log1p(y_valid)

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_valid,y_valid,reference=lgb_train)
                
            if self.weight:
                lgb_train = lgb.Dataset(
                    X_train,
                    y_train,
                    weight=train_tmp['weight']
                )
                lgb_eval = lgb.Dataset(
                    X_valid,
                    y_valid,
                    reference=lgb_train,
                    weight=valid_tmp['weight']
                )

            model = lgb.train(
                self.params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=self.verbose_eval,
                categorical_feature=self.cat_col,
            )
            
            if self.is_target_log:
                # 元に戻す
                oof[valid_idx] = np.expm1(
                    model.predict(X_valid, num_iteration=(model.best_iteration))
                )
            else:
                oof[valid_idx] = model.predict(X_valid, num_iteration=(model.best_iteration))
                
            # 0以下の数値の補正
            oof[oof < 0] = 0

            models.append(model)

            test_preds_all += model.predict((self.test[self.use_col]), num_iteration=(model.best_iteration)) / self.fold_num
            
            # score
            if self.is_target_log:
                y_train = np.expm1(y_train)
                y_valid = np.expm1(y_valid)
            
            fold_score = get_score(y_valid, oof[valid_idx])
            print(f"Fold {fold} {self.score_name}: {fold_score:.4f}")
            wandb.log({f"{self.target}_fold{fold}_{self.score_name}": fold_score})
            
            # save
            name = f"{self.target}_fold{fold}"
            os.makedirs(f"{self.output_dir}/model", exist_ok=True)
            model.save_model(f"{self.output_dir}/model/{name}")

        # 0以下の数値の補正
            test_preds_all[test_preds_all < 0] = 0
            
        # log変換を元に戻す
        if self.is_target_log:
            test_preds_all = np.expm1(test_preds_all)
            
        # make oof
        os.makedirs(f"{self.output_dir}/oof", exist_ok=True)
        np.save(f"{self.output_dir}/oof/oof_{self.target}", oof)

        # make pred
        os.makedirs(f"{self.output_dir}/pred", exist_ok=True)
        np.save(f"{self.output_dir}/pred/pred_{self.target}", test_preds_all)

        # calc score
        
        ### kf,skf
        score = get_score(self.train[self.target], oof)
        
        ### tss
        # tar_auc = roc_auc_score(self.train[self.target].values[(self.train.fold == 0).sum():], oof[(self.train.fold == 0).sum():])
        # use only last fold
        # tar_auc = roc_auc_score(self.train[self.target].values[(self.train.fold != 4).sum():], oof[(self.train.fold != 4).sum():])
        
        print(f"ALL {self.target} {self.score_name}: {score: .4f}")
        
        wandb.log({f"{self.target}_all_{self.score_name}": score})

        return oof, test_preds_all, models, score

    def seed_ave(self, seed_ave_num):
        oof_sa = np.zeros((len(self.train),))
        test_preds_all_sa = np.zeros((len(self.test),))
        models_sa = []

        for seed in seed_ave_num:
            print(f"############# SEED : {seed} #############")
            oof_, test_preds_all_, models_ = self.fit(seed)
            oof_sa += oof_ / len(seed_ave_num)
            test_preds_all_sa += test_preds_all_ / len(seed_ave_num)
            models_sa.append(models_)
        
        return oof_sa, test_preds_all_sa, models_sa

    def feature_importance(self, PLOT=True):
        """lightGBM の model 配列の feature importance を plot する
        CVごとのブレを boxen plot として表現します.

        args:
            models:
                List of lightGBM models
            feat_train_df:
                学習時に使った DataFrame
        """
        feature_importance_df = pd.DataFrame()

        for fold, model in enumerate(self.models):
            _df = pd.DataFrame()
            _df[f"fold_{fold}"] = model.feature_importance(importance_type='gain')
            _df = _df.T
            _df.columns = self.use_col
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0)

        order = _df.mean().sort_values(ascending=False).index.tolist()

        if PLOT:
            fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.4), len(order) * 0.5))
            sns.boxplot(data=feature_importance_df, orient='h', order=order, ax=ax, palette='viridis')
            ax.grid()
            fig.tight_layout()
            os.makedirs(f"{self.output_dir}/imp", exist_ok=True)
            fig.savefig(f"{self.output_dir}/imp/{self.target}_importance.png")

        return feature_importance_df, order

# def lgbm_per_one_tar(
#     train_lgbm: pd.DataFrame(),
#     test_feat_df: pd.DataFrame(),
#     target: str,
#     use_col: list,
#     cat_col: list,
#     cfg: dict,
#     OUTPUT_DIR: str,
#     change_dict: dict, 
#     DO_FIT=True, 
#     DO_SEED_AVE=False, 
#     DEBUG=False
#     ):
    
#     lgbm = LGBM(train=train_lgbm,
#       test=test_feat_df,
#       target=target,
#       use_col=use_col,
#       cat_col=cat_col,
#       cfg=cfg,
#       OUTPUT_DIR=OUTPUT_DIR,
#       DO_FIT=DO_FIT,
#       DO_SEED_AVE=DO_SEED_AVE,
#       DEBUG=DEBUG)

#     feature_importance_df, order = lgbm.feature_importance(change_dict=change_dict)