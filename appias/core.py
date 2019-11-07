""" Appias Core """
import os
from warnings import warn

import pandas as pd
import numpy as np
import seaborn as sns
from loguru import logger
from tqdm.auto import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from .frame import AppiasDataFrame
from .util import reduce_memory_usage


class appias:
    """ Some glue for a few of the standard steps when exploring a dataset and building a model.
    Params:
        df: Appias DataFrame
        response: str
            Column name of response
        models: dict (optional)
            sklearn compliant models
        logging: boolean (optional): Whether to log actions to 'appias_logs' folder
    """
    def __init__(self, df, response=None, models=None, logging=False):
        self.df = self._ensure_appias_df(df)
        self.response = response
        self.models = models
        self.logging = logging

        self.y = self.df[self.response]
        self.X = self.df.drop(self.response, axis=1)

        if self.logging:
            self._setup_logging()

    @staticmethod
    def _ensure_appias_df(df):
        if isinstance(df, AppiasDataFrame):
            return df
        elif isinstance(df, pd.DataFrame):
            return AppiasDataFrame(df)
        else:
            raise TypeError('Expecting a DataFrame for df.')

    def reduce_memory_usage(self, verbose=0):
        """ Reduces memory used by DataFrame. """
        self.df = reduce_memory_usage(self.df, verbose=verbose)

    @staticmethod
    def _setup_logging():
        logger.remove()
        sink = os.path.join(os.getcwd(), 'appias_logs/{time}.log')
        logger.add(sink, format="{time} | {message}")

    def _log(self, msg):
        if self.logging:
            logger.info(msg)

    def plot_hist_features(self, **kwargs):
        """ Plots histogram of features. """
        for col in self.X:
            self.X[col].plot_hist(**kwargs)

    def pairplot(self, **kwargs):
        """ Plots pairplot of features via seaborn. """
        sns.pairplot(self.X, **kwargs)

    def impute_features(self, na_strategy=np.median, inf_strategy=max):
        """ Imputes infinite and nan values for all features.
        Params:
            na_strategy: callable (optional): Defaults to np.median
            inf_strategy: callable (optional): Defaults to max
        Returns:
            DataFrame
        """
        for col in self.X:
            self.X[col] = self.X[col].impute(na_strategy=na_strategy, inf_strategy=inf_strategy)

    def describe_features(self):
        """ Describe features with enhanced pd.DataFrame.describe
        Returns:
            pd.DataFrame: Summary statistics
        """
        ldesc = [s.describe() for _, s in self.X.iteritems()]
        # set a convenient order for rows
        names = []
        ldesc_indexes = sorted((x.index for x in ldesc), key=len)
        for idxnames in ldesc_indexes:
            for name in idxnames:
                if name not in names:
                    names.append(name)

        d = pd.concat([x.reindex(names, copy=False) for x in ldesc], axis=1, sort=False)
        d.columns = self.X.columns.copy()

        return d

    def remove_outliers_features(self, **kwargs):
        """ Removes outliers via median absolute deviation for all features. """
        for col in self.X:
            self.X[col] = self.X[col].remove_outliers(**kwargs)

    def verify(self):
        """ A few data quality checks to ensure data is okish enough for ml.
        """
        if not self.df.index.is_unique:
            warn("Index is not unique, resetting.", UserWarning)
            self.df = self.df.reset_index(drop=True)

        constants = self.X.apply(lambda x: x.type() == 'constant')
        if constants.any():
            cols = self.X.columns[constants]
            warn("Dropped columns with constants: {}".format(cols.values), UserWarning)
            self.df = self.df.drop(columns=cols, axis=1)

        duplicated = self.X.columns.duplicated()
        if duplicated.any():
            raise ValueError("Duplicate columns: {}".format(self.X.columns[duplicated]))

        object_columns = self.X.select_dtypes(include='O').columns
        if object_columns.any():
            raise ValueError("Object columns: {}".format(object_columns.values))

    def fit(self, transform=False):
        """ Fits models.
        Returns:
            dict
        """
        for _, model in self.models.items():
            if transform:
                model = self._make_pipeline(model)

            model.fit(self.X, self.y)

        return self.models

    def predict(self):
        """ Predicts using trained models.
        Returns:
            dict
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(self.X)
        return predictions

    def fit_predict(self, **kwargs):
        """ Fit and predict models. """
        return self.fit(**kwargs), self.predict()

    def score(self, sample_weight=None):
        """ Returns score of the prediction for each model.
        Params:
            sample_weight: array-like (optional)
        Returns:
            dict
        """
        scores = {}
        for name, model in self.models.items():
            scores[name] = model.score(self.X, self.y, sample_weight)

        self._log(scores)

        return scores

    def cross_validate(self,
                       cv=5,
                       scoring=('r2', 'neg_mean_squared_error'),
                       return_train_score=True,
                       n_jobs=-1,
                       transform_response=False,
                       **kwargs):
        """ Evaluate metric(s) by cross-validation and also record fit/score times.
        Params:
            cv: int (optional)
            scoring: array-like (optional)
            return_train_score: boolean (optional)
            n_jobs: int (optional)
            transform_response: boolean (optional): Whether to transform the response during cv.
        Returns:
            tuple of dicts
        """
        cvs = {}
        for name, model in tqdm(self.models.items()):
            if transform_response:
                model = self._make_pipeline(model)

            cvs[name] = cross_validate(
                model,
                X=self.X,
                y=self.y,
                cv=cv,
                scoring=scoring,
                return_train_score=return_train_score,
                n_jobs=n_jobs,
                **kwargs
            )

        self._log(cvs)
        return cvs

    @staticmethod
    def cross_validate_averages(cv_results):
        """ Averages cross validation results from sklearn's cross_validate
        Params:
            cv_results: dict: Results from cross_validate
        """
        cv_avgs = {}
        for model, results in cv_results.items():
            cv_avgs[model] = {k: np.mean(v) for k, v in results.items()}
        return cv_avgs

    def _make_pipeline(self, model):
        if self.y.transformer is None:
            raise ValueError("Transformer is not set, please set with AppiasSeries.transform.")

        return make_pipeline(self.y.transformer, model)
