import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LeadLagger(BaseEstimator, TransformerMixin):
    def __init__(self, n_lags=10, n_leads=0, freq='M', start = "1960", cols = None):
        self.n_lags=n_lags
        self.n_leads=n_leads
        self.freq=freq
        self.start = start
        self.cols = cols
        
    def fit(self, X, y=None):
        self.lags=list(range(-1*self.n_leads, self.n_lags+1))
        self.lags.remove(0)
        self.cols = X.columns #if self.cols is None else self.cols
        #print "leads and lags : ", self.lags
        return self
        
    def transform(self, X, y=None): #this could be optimised with an apply instead of looping on columns
        if self.n_lags==0 and self.n_leads==0:
            return X
        else:
            if not isinstance(X, pd.DataFrame):
                X=pd.DataFrame(X)
            lagged=[X]
            #lagged.append(X)
            for lag in self.lags:
                lag = lag*3  if self.freq=='Q' else lag
                _=X[self.cols].shift(lag)
                sign = "-" if np.sign(lag)==1 else "+"
                if lag!=0:
                    _=_.rename(columns={col:"{}, M{}{}".format(col,sign, np.abs(lag)) for col in self.cols})
                lagged.append(_)
            X_transform=pd.concat(lagged, axis=1)
            #X_transform = X_transform[X_transform.index>=self.start]
            return X_transform

class TimeAlignment(BaseEstimator, TransformerMixin):
    """
    Time Alignement aims at creating a compact information set, that is a data frame where all information is available
    at index time. In other words, TimeAlignement transforms raw data in a dataset that I can cut at time T using simple
    indexing so that all remaining data would be available to a forecaster standing in T.
    """
    def __init__(self,today, freq='MS', bonus=0):
        self.freq=freq
        self.today= today
        self.rd={}
        self.bonus=bonus
        pass

    def fit(self, X, y=None):
        
        return self

    def transform(self, X, y=None): 
        X_reindex =X.copy()
        X_reindex.index = X_reindex.index.map(lambda t: t.replace(day=1))
        X_=pd.DataFrame(data=X_reindex, index=pd.date_range(start=X.first_valid_index(), 
            end=self.today.replace(day=1), freq=self.freq))
        for col in X_.columns:
            self.rd[col] = self.get_rd(X_[col])
            if self.rd[col] > 0:
                X_[col] = X_[col].shift(self.rd[col])
        return X_
    
    def reverse_transformation(self, X, y=None, start="1960"):
        _X=pd.DataFrame(data=X, index=pd.date_range(start=start, end=self.today, freq=X.index.freq), columns=X.columns)
        for col in _X.columns:
            if self.rd[col]>0:
                _X[col]=_X[col].shift(-1*self.rd[col])
        return _X
        
    def get_rd(self, ser):
        return max(0, ser[ser.last_valid_index():].isnull().sum()-self.bonus)

        




