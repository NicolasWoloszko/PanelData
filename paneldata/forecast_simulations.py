#forecast simulations

import numpy as np
import pandas as pd
from sklearn.model_selection._split import (_BaseKFold)


import warnings
import numbers
import time
from traceback import format_exception_only

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils._joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder

import pandas as pd

from sklearn.model_selection._validation import _fit_and_predict

def cross_val_predict(estimator, X, y=None, groups=None, cv='warn',
                      n_jobs=None, verbose=0, fit_params=None,
                      pre_dispatch='2*n_jobs', method='predict'):

    """
    Minor modifications and simplications brought to the sklearn function in order to allow
    for application with non-partition CV scheme. 
    """

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))


    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    predictions = np.concatenate(predictions)

    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])
    test_index = [y.index[_] for _ in test_indices]
    #print(predictions)

    if y.ndim == 1:
        return pd.Series(predictions, index = test_index)
    elif y.ndim>1:
        return pd.DataFrame(predictions, index = test_index)


class PanelForwardLookingCrossVal(_BaseKFold):

    def __init__(self, n_splits=10, index_year=1, verbose = 0):

        self.n_splits=n_splits
        self.index_year=index_year
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        
        y_full = y.dropna()
        years=sorted(set(y_full.index.get_level_values(self.index_year)))
        assert self.n_splits<len(set(years)), "n_splits must be less than number of years"
        
        n_samples=len(X)
        indices = np.arange(n_samples)
        start=years[len(years)-self.n_splits]

        for i, year in enumerate(years) :
            if year>=start:
                if self.verbose>0:
                    print(year)
                if y.ndim ==1:    
                    train_indexes=np.where((X.index.get_level_values(self.index_year)<year) & (~np.isnan(y)))

                    ### Release delays in monthly series for GBR and CAN
                    ## UK
                    remove = np.where((X.index.get_level_values(self.index_year)<year) & (~np.isnan(y)) & (X.index.get_level_values(0) == "United Kingdom")) 
                    remove = remove[0][-2:]
                    train_indexes = np.setdiff1d(train_indexes,remove)

                    ## Canada
                    remove = np.where((X.index.get_level_values(self.index_year)<year) & (~np.isnan(y)) & (X.index.get_level_values(0) == "Canada")) 
                    remove = remove[0][-3:]
                    train_indexes = np.setdiff1d(train_indexes,remove)

                elif y.ndim>1:
                    train_indexes=np.where((X.index.get_level_values(self.index_year)<year) & (~np.isnan(y).any(axis = 1)))
                
                # if i<len(years)-1:
                #     test_indexes=np.where((X.index.get_level_values(self.index_year)<=year) & (X.index.get_level_values(self.index_year)>years[i-1]))
                # else:
                #     test_indexes=np.where((X.index.get_level_values(self.index_year)>years[i-1]))
                if i<len(years)-1:
                    test_indexes=np.where(X.index.get_level_values(self.index_year)==year)
                else:
                    test_indexes=np.where((X.index.get_level_values(self.index_year)>=year))

                yield (indices[train_indexes], indices[test_indexes])



class PanelForwardLookingCrossVal2(_BaseKFold):

    def __init__(self, n_splits=10, index_year=1, verbose = 0):

        self.n_splits=n_splits
        self.index_year=index_year
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        
        y_full = y.dropna()
        years=sorted(set(y_full.index.get_level_values(self.index_year)))
        years = [year for year in years if year.month in [1, 4, 7, 10]]
        assert self.n_splits<len(set(years)), "n_splits must be less than number of years"
        
        n_samples=len(X)
        indices = np.arange(n_samples)
        start=years[len(years)-self.n_splits]

        for i, year in enumerate(years) :
            if year>=start:
                if self.verbose>0:
                    print(year)

                if y.ndim ==1:    
                    train_indexes=np.where(
                        (X.index.get_level_values(self.index_year)<year) & 
                        (~np.isnan(y)))
                elif y.ndim>1:
                    train_indexes=np.where(
                        (X.index.get_level_values(self.index_year)<year) & 
                        (~np.isnan(y).any(axis = 1)))

                if i<len(years)-1:
                    test_indexes=np.where(
                        (X.index.get_level_values(self.index_year)>=year) & 
                        (X.index.get_level_values(self.index_year)<years[i+1])
                        )
                else:
                    test_indexes=np.where((X.index.get_level_values(self.index_year)>=year))

                yield (indices[train_indexes], indices[test_indexes])




class OneCountryOneModelCV(_BaseKFold):
    """ 
    This CV is used to run one model by country using only each country's data to train the model
    
    n_splits is the number of predicted quarter starting from the max date and going back in time
    """

    def __init__(self, n_splits=10, index_year=1, index_country=0):
        self.n_splits = n_splits
        self.index_year = index_year
        self.index_country = index_country

    def split(self, X, y=None, groups=None):
        y_full = y.dropna()
        years=sorted(set(y_full.index.get_level_values(self.index_year)))
        countries = sorted(set(X.index.get_level_values(self.index_country)))

        assert self.n_splits<len(set(years)), "n_splits must be less than number of years"
        
        n_samples=len(X)
        indices = np.arange(n_samples)
        start=years[len(years)-self.n_splits]
        for country in countries:
            print(country)
            for i, year in enumerate(years):
                if year>=start:
                    train_indexes=np.where((X.index.get_level_values(self.index_country)==country) & 
                        (X.index.get_level_values(self.index_year)<year) & 
                        (~np.isnan(y)))

                    if i<len(years)-1:
                        test_indexes=np.where((X.index.get_level_values(self.index_country)==country) & 
                            (X.index.get_level_values(self.index_year)<=year) & 
                            (X.index.get_level_values(self.index_year)>years[i-1]))
                    else:
                        test_indexes=np.where((X.index.get_level_values(self.index_country)==country) & 
                            (X.index.get_level_values(self.index_year)>years[i-1]))

                    if len(test_indexes)>0:
                        #print(indices[train_indexes], indices[test_indexes])
                        yield (indices[train_indexes], indices[test_indexes])







