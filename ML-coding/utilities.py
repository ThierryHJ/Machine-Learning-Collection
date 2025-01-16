import numpy as np
import pandas as pd
import math

def one_hot_encoding(df, col):
    '''perfrom one hot encoding for a specific categorical column of a dataframe'''
    new_col = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,new_col], axis=1).drop([col], axis=1)
    return df

def fill_na_mean(df, col):
    '''fill na with column mean'''
    df[col].fillna(df[col].mean(), inplace=True)
    
    
def bootstrap(X, y, replacement=True):
    '''perform bootstrap given feature matrix and target array'''
    n_samples = X.shape[0]
    random_idx = np.random.choice(range(n_samples), 
                                  n_samples, 
                                  replace=True)
    new_X, new_y = X[random_idx, :], y[random_idx]
    
    return new_X, new_y


