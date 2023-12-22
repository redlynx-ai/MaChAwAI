"""
Utility classes and functions.
"""

import pandas as pd

class DFStats():
    """
    Compute statistics of a collection of DataFrame objects.
    """
    def __init__(self, dfs: 'list[pd.DataFrame]', columns: list = None) -> None:
        assert len(dfs) > 0
        self.dfs = dfs
        if columns == None:
            columns = self.dfs[0].columns
        self.columns = columns
        self.stats = list(map(lambda df: df[self.columns].describe(), self.dfs))

    def getMin(self):
        """
        Return the minimum values.
        """
        min_vals = map(lambda s: s.loc['min'].values, self.stats)
        min_df = pd.DataFrame(min_vals, columns=self.columns)
        return min_df.min()

    def getMax(self):
        """
        Return the maximum values.
        """
        max_vals = map(lambda s: s.loc['max'].values, self.stats)
        max_df = pd.DataFrame(max_vals, columns=self.columns)
        return max_df.max()
    
    def getMean(self):
        """
        Return the mean values.
        """
        mean_vals = map(lambda s: s.loc['mean'].values, self.stats)
        mean_df = pd.DataFrame(mean_vals, columns=self.columns)
        return mean_df.mean()

    def getStats(self):
        """
        Return the full statistics DataFrame.
        """
        return pd.DataFrame({"min": self.getMin(),
                            "max": self.getMax(),
                            "mean": self.getMean()}).transpose()