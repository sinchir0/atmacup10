import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer

def rankgauss_scaler(train: pd.DataFrame(), test: pd.DataFrame(), target_col: list, n_quantiles: int) -> pd.DataFrame():

    for col in target_col:
        transformer = QuantileTransformer(n_quantiles=n_quantiles,random_state=33, output_distribution="normal")
        
        vec_len_trn = len(train[col].values)
        vec_len_tst = len(test[col].values)
        
        raw_vec_trn = train[col].values.reshape(vec_len_trn, 1)
        raw_vec_tst = test[col].values.reshape(vec_len_tst, 1)
        
        transformer.fit(raw_vec_trn)

        train[col] = transformer.transform(raw_vec_trn).reshape(1, vec_len_trn)[0]
        test[col] = transformer.transform(raw_vec_tst).reshape(1, vec_len_tst)[0]
        
    return train, test