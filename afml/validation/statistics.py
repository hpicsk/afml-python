import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from scipy.stats import norm, t, rankdata, spearmanr, probplot, ttest_1samp
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings

class PerformanceStatistics:
    """
    A collection of methods for calculating various backtest and performance statistics,
    drawing from concepts in "Advances in Financial Machine Learning" by Marcos López de Prado.
    This class combines metrics from Chapters 7 and 9.
    """

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, annualization_factor: float = 252) -> float:
        """
        Calculate the annualized Sharpe ratio from a series of returns.

        Parameters:
        - returns: Array of periodic returns.
        - annualization_factor: Factor to annualize the Sharpe ratio (e.g., 252 for daily).

        Returns:
        - Annualized Sharpe ratio.
        """
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)

    @staticmethod
    def probability_of_backtest_overfitting(
        is_performance: np.ndarray, oos_performance: np.ndarray
    ) -> float:
        """
        Calculates the Probability of Backtest Overfitting (PBO).
        PBO measures the likelihood that a strategy configuration that performed best
        in-sample will underperform the median out-of-sample performance.

        Parameters:
        - is_performance: Array of in-sample performance metrics (e.g., Sharpe ratios).
        - oos_performance: Array of out-of-sample performance metrics.

        Returns:
        - The PBO score, a float between 0 and 1.
        """
        if len(is_performance) != len(oos_performance):
            raise ValueError("In-sample and out-of-sample arrays must have the same length.")
        
        best_is_idx = np.argmax(is_performance)
        best_is_oos_perf = oos_performance[best_is_idx]
        
        median_oos_perf = np.median(oos_performance)
        
        # Count how many times the best IS configuration underperforms the median OOS
        underperform_count = np.sum(oos_performance < best_is_oos_perf)
        total_count = len(oos_performance)
        
        # PBO is the rank of the best IS strategy's OOS performance
        pbo = rankdata(oos_performance)[best_is_idx] / total_count
        return pbo

    @staticmethod
    def deflated_sharpe_ratio(
        sharpe_ratio: float,
        n_trials: int,
        n_obs: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        annualization_factor: float = 252.0
    ) -> float:
        """
        Calculates the Deflated Sharpe Ratio (DSR), which adjusts the Sharpe ratio for
        selection bias under multiple trials and for non-normal returns.

        Parameters:
        - sharpe_ratio: The observed annualized Sharpe ratio.
        - n_trials: The number of independent strategy trials.
        - n_obs: The number of observations (e.g., days) used to calculate the Sharpe ratio.
        - skew: Skewness of the returns series.
        - kurtosis: Kurtosis of the returns series (not excess kurtosis).
        - annualization_factor: The factor used to annualize the Sharpe ratio.

        Returns:
        - The DSR as a p-value, indicating the probability that the SR is not a statistical fluke.
        """
        sr_deannualized = sharpe_ratio / np.sqrt(annualization_factor)
        
        var_sr = (1 - skew * sr_deannualized + (kurtosis - 1) / 4 * sr_deannualized**2)
        
        if var_sr <= 0:
            warnings.warn("Variance of Sharpe ratio is non-positive. Returning 0.0 for DSR.")
            return 0.0

        if n_trials == 1:
            e_max_z = 0.0
        else:
            gamma = np.euler_gamma
            e_max_z = (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))

        z_score = (sr_deannualized * np.sqrt(n_obs - 1) - e_max_z) / np.sqrt(var_sr)
        dsr = norm.cdf(z_score)
        
        return dsr

    @staticmethod
    def minimum_track_record_length(
        target_sharpe: float,
        observed_sharpe: Optional[float] = None,
        prob: float = 0.95,
        skew: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Calculates the Minimum Track Record Length (MinTRL) required to be statistically
        confident that a strategy's true Sharpe ratio exceeds a target Sharpe ratio.

        Parameters:
        - target_sharpe: The target Sharpe ratio to beat.
        - observed_sharpe: The observed annualized Sharpe ratio of the strategy.
        - prob: The desired confidence level.
        - skew: Skewness of the returns.
        - kurtosis: Kurtosis of the returns.

        Returns:
        - The minimum number of years required for the track record.
        """
        if observed_sharpe is None or observed_sharpe <= target_sharpe:
            return np.nan

        term1 = (1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
        term2 = (norm.ppf(prob) / (observed_sharpe - target_sharpe))**2
        
        min_trl_years = term1 * term2
        return min_trl_years

    @staticmethod
    def drawdown_and_time_under_water(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates the drawdown series and time under water series.

        Parameters:
        - returns: A pandas Series of returns.

        Returns:
        - A tuple containing:
          - Drawdown series.
          - Time under water series (number of periods since last peak).
        """
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        
        drawdown = (cum_returns - running_max) / running_max
        
        time_under_water = pd.Series(index=returns.index, dtype=int)
        in_drawdown = False
        peak_idx = -1
        for i in range(len(drawdown)):
            if drawdown[i] < 0 and not in_drawdown:
                in_drawdown = True
                peak_idx = i -1 if i > 0 else 0
            
            if in_drawdown:
                time_under_water[i] = i - peak_idx
            
            if drawdown[i] == 0:
                in_drawdown = False
                time_under_water[i] = 0
                
        return drawdown, time_under_water.fillna(0)

    @staticmethod
    def compute_all_metrics(returns: pd.Series, annualization_factor: int = 252) -> Dict[str, float]:
        """
        Computes a dictionary of all key performance metrics for a returns series.

        Parameters:
        - returns: A pandas Series of returns.
        - annualization_factor: The factor for annualizing metrics.

        Returns:
        - A dictionary of performance metrics.
        """
        cum_returns = (1 + returns).cumprod()
        drawdown, tuw = PerformanceStatistics.drawdown_and_time_under_water(returns)
        
        metrics = {
            'sharpe_ratio': PerformanceStatistics.sharpe_ratio(returns.to_numpy(), annualization_factor),
            'cagr': (cum_returns.iloc[-1])**(annualization_factor / len(returns)) - 1,
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown.mean(),
            'max_time_under_water': tuw.max(),
            'avg_time_under_water': tuw.mean(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis() + 3,  # Report full kurtosis
            'calmar_ratio': (cum_returns.iloc[-1] - 1) / abs(drawdown.min()),
            'sortino_ratio': np.mean(returns) / returns[returns < 0].std() * np.sqrt(annualization_factor) if returns[returns < 0].std() > 0 else 0.0
        }
        return metrics

    @staticmethod
    def multiple_testing_correction(p_values: np.ndarray, method: str = 'fdr_bh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple testing correction to p-values.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Array of p-values
        method : str, optional
            Method for multiple testing correction
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (Corrected p-values, Rejection mask)
        """
        rejected, p_corrected, _, _ = multipletests(p_values, method=method, is_sorted=False, returnsorted=False)
        return p_corrected, rejected 

    @staticmethod
    def sharpe_ratio_from_positions(
        returns: np.ndarray, positions: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: float = 252.0
    ) -> float:
        """Calculates Sharpe Ratio from returns and positions."""
        strategy_returns = returns * positions
        return PerformanceStatistics.sharpe_ratio(strategy_returns, annualization_factor)

    @staticmethod
    def drawdown_from_positions(
        returns: np.ndarray, positions: np.ndarray
    ) -> Tuple[float, float]:
        """Calculates max drawdown and duration from returns and positions."""
        strategy_returns = returns * positions
        cum_returns = np.cumprod(1 + strategy_returns) - 1
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns + 1) / (running_max + 1) - 1
        
        max_drawdown = np.min(drawdowns)
        
        drawdown_durations = np.zeros_like(drawdowns)
        duration = 0
        for i in range(1, len(drawdowns)):
            if drawdowns[i] < 0:
                duration += 1
            else:
                duration = 0
            drawdown_durations[i] = duration
        
        max_duration = np.max(drawdown_durations)
        
        return max_drawdown, max_duration

    @staticmethod
    def profit_factor(returns: np.ndarray, positions: np.ndarray) -> float:
        """Calculates profit factor (gross profits / gross losses)."""
        strategy_returns = returns * positions
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss

    @staticmethod
    def cagr(returns: np.ndarray, positions: np.ndarray, years: float) -> float:
        """Calculates Compound Annual Growth Rate (CAGR)."""
        strategy_returns = returns * positions
        cumulative_return = np.prod(1 + strategy_returns) - 1
        return (1 + cumulative_return) ** (1 / years) - 1

    @staticmethod
    def pbo_test_spearman(in_sample_sharpes: List[float], out_of_sample_sharpes: List[float]) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO) based on Spearman rank correlation.
        """
        if len(in_sample_sharpes) != len(out_of_sample_sharpes):
            raise ValueError("In-sample and out-of-sample Sharpe lists must have the same length.")
        if len(in_sample_sharpes) < 2:
            return 0.5
        
        rank_corr, _ = spearmanr(in_sample_sharpes, out_of_sample_sharpes)
        if np.isnan(rank_corr):
            return 0.5
        
        return 0.5 * (1 - rank_corr)

    @staticmethod
    def haircut_sharpe_ratio(sharpe_ratio: float, n_trials: int, n_obs: int, p_value: float = 0.05) -> float:
        """
        Apply haircut to Sharpe ratio to account for selection bias.
        """
        critical_value = norm.ppf(1 - p_value / (2 * n_trials))
        se = 1 / np.sqrt(n_obs)
        haircut = sharpe_ratio - critical_value * se
        return max(0, haircut) 