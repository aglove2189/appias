""" DataFrame """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .series import AppiasSeries


class AppiasDataFrame(pd.DataFrame):
    """ AppiasDataFrame """
    @property
    def _constructor(self):
        return AppiasDataFrame

    @property
    def _constructor_sliced(self):
        return AppiasSeries

    def plot_hist(self, **kwargs):
        """ Plots histogram. """
        for _, s in self.iteritems():
            sns.distplot(s, **kwargs)
            plt.show()

    def inf_to_na(self):
        """ Replaces infinite values with nan values.
        Returns:
            DataFrame
        """
        return self.replace([np.inf, -np.inf], np.nan)
