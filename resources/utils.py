import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def drop_columns(X,columns):
    return X.drop(columns,axis=1)

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):  
        return self
    
    def transform(self, X):  
        X_ = X.copy()
        return drop_columns(X_, self.columns_to_drop)