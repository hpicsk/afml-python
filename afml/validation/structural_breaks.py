import numpy as np
import pandas as pd
from typing import Tuple, Optional
import statsmodels.api as sm

class StructuralBreaks:
    """
    Methods for detecting structural breaks in time series data, as described
    in Chapter 17 of "Advances in Financial Machine Learning".
    """

    @staticmethod
    def get_chow_type_stat(series: pd.Series, f_value: float) -> float:
        """
        Performs a Chow-type test for a structural break at every point in the series.
        This function computes the F-statistic for a single breakpoint test.
        The book suggests iterating this over the series to find the max F-statistic.

        Args:
            series: The time series to test.
            f_value: The point at which to split the series for the test.

        Returns:
            The F-statistic for the given breakpoint.
        """
        series1 = series[:int(f_value * len(series))]
        series2 = series[int(f_value * len(series)):]
        
        if len(series1) < 2 or len(series2) < 2:
            return 0.0

        # Fit models for the two parts and the whole series
        X1 = sm.add_constant(np.arange(len(series1)))
        model1 = sm.OLS(series1, X1).fit()
        
        X2 = sm.add_constant(np.arange(len(series2)))
        model2 = sm.OLS(series2, X2).fit()
        
        X_full = sm.add_constant(np.arange(len(series)))
        model_full = sm.OLS(series, X_full).fit()
        
        # Chow test statistic
        rss_full = model_full.ssr
        rss_1 = model1.ssr
        rss_2 = model2.ssr
        
        k = 2 # Number of parameters (const, trend)
        n1 = len(series1)
        n2 = len(series2)
        
        if (n1 + n2 - 2 * k) == 0:
            return 0.0
            
        chow_stat = ((rss_full - (rss_1 + rss_2)) / k) / ((rss_1 + rss_2) / (n1 + n2 - 2 * k))
        
        return chow_stat

    @staticmethod
    def get_cusum_stat(series: pd.Series, threshold: float) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Computes the CUSUM statistic for detecting a shift in the mean.

        Args:
            series: The time series of returns or residuals.
            threshold: The threshold for flagging a break.

        Returns:
            A tuple of (CUSUM statistics series, thresholds series).
        """
        s_plus = pd.Series(0.0, index=series.index)
        s_minus = pd.Series(0.0, index=series.index)
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.DataFrame({'S+': s_plus, 'S-': s_minus}), pd.Series(threshold, index=series.index)

        for i in range(1, len(series)):
            s_plus.iloc[i] = max(0, s_plus.iloc[i-1] + series.iloc[i] - mean)
            s_minus.iloc[i] = min(0, s_minus.iloc[i-1] + series.iloc[i] - mean)
            
        cusum_stat = pd.DataFrame({'S+': s_plus, 'S-': s_minus}) / std
        
        thresholds = pd.Series(threshold, index=series.index)
        
        return cusum_stat, thresholds

    @staticmethod
    def get_cusum_vol_stat(returns: pd.Series, threshold: float) -> Tuple[pd.Series, pd.Series]:
        """
        Computes the CUSUM statistic on squared returns for detecting a shift in volatility.

        Args:
            returns: The time series of returns.
            threshold: The threshold for flagging a break.

        Returns:
            A tuple of (CUSUM volatility statistics series, thresholds series).
        """
        squared_returns = returns**2
        return StructuralBreaks.get_cusum_stat(squared_returns, threshold)
