import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, List
from statsmodels.tsa.stattools import adfuller
import warnings
from scipy.stats import pearsonr

class FractionalDifferentiation:
    """
    Implementation of fractional differentiation.
    """
    
    @staticmethod
    def get_weights(d: float, size: int) -> np.ndarray:
        weights = [1.0]
        for k in range(1, size):
            weights.append(weights[-1] * (d - k + 1) / k)
        return np.array(weights)
    
    @staticmethod
    def frac_diff_expanding_window(series: pd.Series, d: float, 
                                  min_periods: int = 1) -> pd.Series:
        n = len(series)
        if n == 0:
            return pd.Series(dtype='float64')
        weights = FractionalDifferentiation.get_weights(d, n)
        differenced = pd.Series(index=series.index, dtype='float64')
        for i in range(n):
            if i < min_periods - 1:
                differenced.iloc[i] = np.nan
                continue
            window = series.iloc[:i+1]
            differenced.iloc[i] = np.dot(weights[:len(window)], window.values[::-1])
        return differenced
    
    @staticmethod
    def frac_diff_fixed_window(series: pd.Series, d: float, 
                             window_size: Optional[int] = None, 
                             threshold: float = 1e-5) -> pd.Series:
        n = len(series)
        if window_size is None:
            weights = [1.0]
            k = 1
            while True:
                weight = weights[-1] * (d - k + 1) / k
                if abs(weight) < threshold:
                    break
                weights.append(weight)
                k += 1
            window_size = len(weights)
        else:
            weights = FractionalDifferentiation.get_weights(d, window_size)
        differenced = np.zeros(n)
        for i in range(window_size - 1, n):
            current_window = series.iloc[i - window_size + 1:i+1]
            differenced[i] = np.sum(weights * current_window.values[::-1])
        differenced_series = pd.Series(differenced, index=series.index)
        return differenced_series.iloc[window_size-1:]
    
    @staticmethod
    def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 1000) -> np.ndarray:
        weights = [1.0]
        k = 1
        while True:
            weight = weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold or k >= max_size:
                break
            weights.append(weight)
            k += 1
        return np.array(weights)
    
    @staticmethod
    def frac_diff_ffd(series: pd.Series, d: float, 
                    threshold: float = 1e-5, max_size: int = 1000) -> pd.Series:
        weights = FractionalDifferentiation.get_weights_ffd(d, threshold, max_size)
        width = len(weights)
        if width == 0:
            return pd.Series(dtype='float64', index=series.index)
        differenced_values = np.convolve(series.values, weights, mode='full')[:len(series)]
        differenced_series = pd.Series(differenced_values, index=series.index)
        differenced_series.iloc[:width-1] = np.nan
        return differenced_series


class StationarityTests:
    """
    Stationarity tests and methods for finding the optimal differentiation parameter.
    """
    
    @staticmethod
    def adf_test(series: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                adf_result = adfuller(series.dropna(), autolag='AIC')
            p_value = adf_result[1]
        except ValueError:
            p_value = 1.0  # Assume non-stationary if adfuller fails

        is_stationary = p_value < significance
        return is_stationary, p_value

    @staticmethod
    def measure_memory_preservation(original: pd.Series, differentiated: pd.Series) -> float:
        common_idx = original.index.intersection(differentiated.dropna().index)
        if len(common_idx) < 2:
            return 0.0
        corr, _ = pearsonr(original[common_idx], differentiated[common_idx])
        return corr

    @staticmethod
    def find_optimal_d(series: pd.Series, d_range: np.ndarray = np.linspace(0, 1, 21),
                     threshold: float = 1e-5, significance: float = 0.05) -> Tuple[float, pd.DataFrame]:
        results = []
        for d in d_range:
            diff_series = FractionalDifferentiation.frac_diff_ffd(series, d, threshold)
            is_stationary, p_value = StationarityTests.adf_test(diff_series, significance)
            corr = StationarityTests.measure_memory_preservation(series, diff_series)
            results.append({
                'd': d,
                'p_value': p_value,
                'correlation': corr,
                'is_stationary': is_stationary
            })
        results_df = pd.DataFrame(results)
        optimal_d_row = results_df[results_df['is_stationary']].sort_values('correlation', ascending=False).head(1)
        optimal_d = optimal_d_row['d'].iloc[0] if not optimal_d_row.empty else None
        return optimal_d, results_df

    @staticmethod
    def plot_optimization_results(results_df: pd.DataFrame, optimal_d: float, 
                                significance: float = 0.05) -> None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(results_df['d'], results_df['p_value'], 'b-', label='ADF p-value')
        ax1.axhline(y=significance, color='r', linestyle='--', label=f'Significance ({significance})')
        ax1.set_xlabel('d value')
        ax1.set_ylabel('p-value', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(results_df['d'], results_df['correlation'], 'g-', label='Correlation')
        ax2.set_ylabel('Correlation', color='g')
        ax2.tick_params('y', colors='g')
        if optimal_d is not None:
            plt.axvline(x=optimal_d, color='k', linestyle=':', label=f'Optimal d = {optimal_d:.2f}')
        plt.title('Optimal d Search')
        fig.tight_layout()
        plt.legend()
        plt.show() 