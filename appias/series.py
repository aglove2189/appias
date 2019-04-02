""" Series """
from functools import partial

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer


class AppiasSeries(pd.Series):
    """ AppiasSeries """
    _metadata = ['transformer']

    @property
    def _constructor(self):
        return AppiasSeries

    def plot_hist(self, **kwargs):
        """ Plots histogram. """
        sns.distplot(self, **kwargs)
        plt.show()

    @staticmethod
    def _transform(kind, **kwargs):
        """ Internal method for setting up a transformer. """
        if callable(kind):
            return kind(**kwargs)

        transformers = {'log': partial(FunctionTransformer,
                                       func=np.log1p,
                                       inverse_func=np.expm1,
                                       validate=True
                                       ),
                        'exp': partial(FunctionTransformer,
                                       func=np.expm1,
                                       inverse_fun=np.log1p,
                                       validate=True
                                       )
                        }

        if kind not in transformers.keys():
            msg = "'{}' is not a transformer I know, I can do: {} or any sklearn preprocessor."
            raise ValueError(msg.format(kind, transformers.keys()))

        return transformers[kind](**kwargs)

    def transform(self, kind, **kwargs):
        """ Transforms a Series and sets self.transformer to transformer used.
        Params:
            kind: sklearn callable or str
        Returns:
            Series
        """
        self.transformer = self._transform(kind=kind, **kwargs)

        transformed_result = self.transformer.fit_transform(self.values.reshape(-1, 1))

        return self._constructor(transformed_result.reshape(1, -1)[0])

    def inf_to_na(self):
        """ Replaces infinite values with nan values for a Series.
        Returns:
            Series
        """
        return self.replace([np.inf, -np.inf], np.nan)

    def impute_na(self, strategy=np.mean):
        """ Imputes nan values for a Series.
        Params:
            strategy: callable (optional)
        Returns:
            Series
        Raises:
            ValueError: if 'strategy' is not a callable aggregator
        """
        if not callable(strategy):
            raise ValueError("'strategy' should be a callable aggregator")

        return self.replace(np.nan, strategy(self))

    def impute_inf(self, strategy=max):
        """ Imputes infinite values for a Series.
        Params:
            strategy: callable (optional)
        Returns:
            Series
        Raises:
            ValueError: if 'strategy' is not a callable aggregator
        """
        if not callable(strategy):
            raise ValueError("'strategy' should be a callable aggregator")

        return self.replace([np.inf, -np.inf], strategy(self.inf_to_na()))

    def impute(self, na_strategy=np.mean, inf_strategy=max):
        """ Imputes infinite and nan values for all features.
        Params:
            na_strategy: callable (optional): Defaults to np.mean
            inf_strategy: callable (optional): Defaults to max
        Returns:
            DataFrame
        """
        return self.impute_inf(strategy=inf_strategy).impute_na(strategy=na_strategy)

    def describe(self, **kwargs):
        """ Adds 'len', 'median', 'distinct', and 'na' to Pandas describe
        Params:
            series: Series
        Returns:
            Seires: Summary statistics
        """
        stats = {'len': len(self), # number of observations in the Series (including na and inf)
                 'median': self.median(),
                 'distinct': self.nunique(dropna=False),
                 'constant': self.nunique(dropna=False) == 1,
                 'na': sum(self.isna())
                }

        if pd.core.dtypes.common.is_numeric_dtype(self):
            stats['infinite'] = sum(np.isinf(self))

        stats = pd.Series(list(stats.values()), index=list(stats.keys()), name=self.name)

        return stats.append(super(AppiasSeries, self).describe(**kwargs))

    def remove_outliers(self, threshhold=3.5):
        """ Removes outliers via median absolute deviation.
        Returns:
            Series
        """
        diff = np.sqrt((self - self.median())**2)
        modified_z_score = 0.6745 * diff / diff.median()

        return self.loc[modified_z_score < threshhold]
