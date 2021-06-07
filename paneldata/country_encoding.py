#country-encoding

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

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