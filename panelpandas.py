import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FixedEffects(BaseEstimator, TransformerMixin):
    def __init__(self, level = 0, prefix = "Country"):
        self.level = level
        self.prefix = prefix
        
    def fit(self, X, y=None):
        return(self)

    def get_cfe(self, X, y=None):
        self.cfe = pd.get_dummies(X.index.get_level_values(self.level), prefix = self.prefix)
        self.cfe.index = X.index
        return(self)

    def transform(self, X, y=None): 
        self.get_cfe(X = X, y=y)
        return pd.concat([X, self.cfe], axis = 1)

class CountryEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, n_components = 3, level = 0):
        self.level = level
        self.n_components = n_components
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None): 
        X['country'] = X.index.get_level_values(self.level)
        df_mean = X.groupby("country").transform(np.mean)
        fe = PCA(self.n_components, whiten=True).fit_transform(df_mean)
        fe = pd.DataFrame(fe, index = df_mean.index, columns = ["fe {}".format(i) for i in range(self.n_components)])
        X = pd.concat([X, fe], axis = 1).drop('country', axis = 1)
        return (X)

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

        



class CreateGrowthRates(BaseEstimator, TransformerMixin):
    def __init__(self, n_lags = 1, transformation = "growth rate", keep_both=True):
        self.keep_both = keep_both
        self.transformation = transformation
        self.n_lags = n_lags

    @staticmethod
    def growth_rate_(x, n_lags):
        x_lag=x.shift(n_lags).replace({0.0:0.000001})#to avoid dividing by zero
        x_g=x.divide(x_lag)-1.0
        return(x_g)

    @staticmethod
    def log_diff_(x, n_lags):
        logx = np.log(x)
        x_lag = logx.shift(n_lags)
        x_g = logx - x_lag
        return(x_g)

    @staticmethod
    def delta_(x, n_lags):
        x_lag=x.shift(n_lags)
        x_g=x - x_lag
        return(x_g)

    @staticmethod
    def transform_level(x, n_lags, transformation):
        if transformation == "growth rate":
            return(CreateGrowthRates.growth_rate_(x, n_lags))
        elif transformation == "log diff":
            return(CreateGrowthRates.log_diff_(x, n_lags))
        elif transformation == "delta":
            return(CreateGrowthRates.delta_(x, n_lags))
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_gr=X.apply(lambda ser: CreateGrowthRates.transform_level(x = ser, n_lags = self.n_lags, transformation = self.transformation))
        if self.keep_both:
            X_gr.columns=[col+", growth rate" for col in X_gr.columns]
            _X=pd.concat([X_gr, X], axis=1)
            return _X
        else:
            return X_gr
    


class FirstDifference(BaseEstimator, TransformerMixin):
    def __init__(self, keep_both=True):
        self.keep_both = keep_both
        
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_gr=X.apply(FirstDifference.first_diff_)
        X_gr.columns=[col+", growth rate" for col in X_gr.columns]
        if self.keep_both:
            _X=pd.concat([X_gr, X], axis=1)
            return _X
        else:
            return X_gr
    
    @staticmethod
    def first_diff_(x):
        x_g=x - x.shift(1)
        return(x_g)

class MakeStationary(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold=threshold
        
    def fit(self, X, y=None):
        self.stationary=X.apply(lambda ser : MakeStationary._is_stationary(self.threshold, ser))
        #print self.stationary
        return self
        
    def transform(self, X, y=None):
        X_stationary=X.loc[:,self.stationary]
        
        X_nonstat=X.loc[:,~self.stationary]
        X_nonstat=X_nonstat.apply(MakeStationary.growth_rate_)
        X_nonstat.columns=[col+", growth rate" for col in X_nonstat.columns]
        _X=pd.concat([X_stationary, X_nonstat], axis=1)

        return _X

    @staticmethod
    def _is_stationary(threshold, ser):
        try:
            return adfuller(ser.dropna())[1]<=threshold
        except ZeroDivisionError:
            return False
    
    @staticmethod
    def growth_rate_(x):
        x_lag=x.shift(1).replace({0:0.01})#to avoid dividing by zero
        x_g=x.divide(x_lag)-1.0
        return(x_g)



class ScalerDF(BaseEstimator, TransformerMixin):
    """
    Applies scaling with scikit learn scaler to pandas data frame while conserving columns and index.
    """
    def __init__ (self, scaler = StandardScaler(), verbose=0):
        self.scaler = scaler
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.verbose>0:
            print ("Fitting ScalerDF")
        self.scaler.fit(X)
        return(self)

    def transform(self, X, y =None):
        X_ = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame) :
            return pd.DataFrame(X_, columns = X.columns, index = X.index)
        elif isinstance(X, pd.Series):
            return X_
        elif isinstance(X, np.ndarray):
            return X_
        else:
            raise (ValueError)
            
    def inverse_transform(self, X, copy=None):
        X_ = self.scaler.inverse_transform(X)
        return(pd.DataFrame(X_, columns = X.columns, index = X.index))


class CountryTransformer(TransformerMixin, BaseEstimator):
    "Applies any transformer that keeps pd.DF type to a panel dataset on a country-by-country basis"
    def __init__(self, transformer = ScalerDF(), country_index = 0):
        self.country_index = country_index
        self.transformer = transformer

    def fit(self, X, y = None):
        self.countries = list(set(X.index.get_level_values(self.country_index)))
        return(self)

    def transform(self, X, y = None):
        transformed_country = []
        for country in self.countries:
            df = X.xs(country, level = self.country_index)
            transformed_country.append(self.transformer.fit_transform(df))
        transformed = pd.concat(transformed_country, keys = self.countries)
        transformed.index.names = X.index.names
        return(transformed)