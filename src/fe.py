import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def count_null(df: pd.DataFrame) -> pd.DataFrame():
    df['count_null'] = df.isnull().sum(axis=1)
    return df

def fill_nan(df: pd.DataFrame) -> pd.DataFrame():
    for col in df.columns.tolist():
        # dtypeがobjectなら欠損は'missing' クラスにする
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        # dtypeがint/floatなら欠損は'-1'とする
        else:
            df[col] = df[col].fillna(-1)      
    return df

def label_encode(train: pd.DataFrame, test: pd.DataFrame, use_col: list) -> pd.DataFrame():
    le = LabelEncoder()
    all_df = pd.concat([train, test], axis=0).reset_index(drop=True)

    for col in use_col:
        le.fit(all_df[col])
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    return train, test

def target_encoding(train: pd.DataFrame(),test: pd.DataFrame(), target: str, target_enc_list: list, fold_num: int):
    for c in target_enc_list:
        print(c)
        # targetを付加
        data_tmp = pd.DataFrame({c: train[c], 'target': train[target]})
        # 変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train.shape[0])

        ###test

        # testはtrain全体でのtarget meanを付与
        target_mean = data_tmp.groupby(c)['target'].mean()
        test[c+'_target_enc'] = test[c].map(target_mean)
        test[c+'_target_enc'] = test[c+'_target_enc'].astype('float')

        ###train

        # 学習データからバリデーションデータを分ける
        for fold in range(fold_num):
            tr_idx = train[train.fold != fold].index
            va_idx = train[train.fold == fold].index

            # 学習データについて、各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()

            # バリデーションデータについて、変換後の値を一時配列に格納
            tmp[va_idx] = train[c].iloc[va_idx].map(target_mean)

        # 変換後のデータで元の変数を置換
        train[c+'_target_enc'] = tmp
        train[c+'_target_enc'] = train[c+'_target_enc'].astype('float')
    
    return train, test

# def target_encoding(train: pd.DataFrame(),test: pd.DataFrame(), target: str, target_enc_list: list, fold_num: int):
#     for c in target_enc_list:
#         print(c)
#         # targetを付加
#         data_tmp = pd.DataFrame({c: train[c], 'target': train[target]})
#         # 変換後の値を格納する配列を準備
#         tmp = np.repeat(np.nan, train.shape[0])

#         ###train

#         # 学習データからバリデーションデータを分ける
#         for fold in range(fold_num):
#             tr_idx = train[train.fold != fold].index
#             va_idx = train[train.fold == fold].index

#             # 学習データについて、各カテゴリにおける目的変数の平均を計算
#             target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()

#             # バリデーションデータについて、変換後の値を一時配列に格納
#             tmp[va_idx] = train[c].iloc[va_idx].map(target_mean)

#         # 変換後のデータで元の変数を置換
#         train[c+'_target_enc'] = tmp
#         train[c+'_target_enc'] = train[c+'_target_enc'].astype('float')

#         ###test
#         # testはtrain全体でのtarget meanを付与
#         target_mean = data_tmp.groupby(c)['target'].mean()
#         test[c+'_target_enc'] = test[c].map(target_mean)
#         test[c+'_target_enc'] = test[c+'_target_enc'] + np.random.randn(len(test),) / 100
#         test[c+'_target_enc'] = test[c+'_target_enc'].astype('float')
    
#     return train, test