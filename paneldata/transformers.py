import pandas as pd
import numpy as np




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