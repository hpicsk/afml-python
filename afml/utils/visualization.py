import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from ..validation.statistics import PerformanceStatistics
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_1samp, norm

class VisualizationTools:
    """
    A collection of static methods for plotting common backtesting and portfolio analysis results.
    (Consolidates plotting functions from chap12 and chap14)
    """
    
    @staticmethod
    def plot_portfolio_performance(results: pd.DataFrame,
                                   benchmark: Optional[pd.Series] = None):
        """
        Plots the main performance charts: cumulative returns and drawdown.

        Args:
            results: The DataFrame from a backtest run, must contain 'portfolio_value' and 'returns'.
            benchmark: An optional series of benchmark returns for comparison.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Cumulative Returns ---
        cum_returns = (1 + results['returns']).cumprod()
        ax1.plot(cum_returns.index, cum_returns, label='Portfolio', color='blue', lw=2)
        
        if benchmark is not None:
            common_idx = results.index.intersection(benchmark.index)
            bench_cum_returns = (1 + benchmark[common_idx]).cumprod()
            ax1.plot(bench_cum_returns.index, bench_cum_returns, label='Benchmark', color='gray', linestyle='--')
        
        ax1.set_title('Cumulative Performance')
        ax1.set_ylabel('Growth of $1')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # --- Drawdown ---
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_efficient_frontier(frontier_df: pd.DataFrame):
        """
        Plots the efficient frontier.

        Args:
            frontier_df: DataFrame containing 'return', 'volatility', and 'sharpe' columns.
        """
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(frontier_df['volatility'], frontier_df['return'], c=frontier_df['sharpe'], cmap='viridis')
        
        # Highlight Max Sharpe and Min Volatility portfolios
        max_sharpe_port = frontier_df.loc[frontier_df['sharpe'].idxmax()]
        min_vol_port = frontier_df.loc[frontier_df['volatility'].idxmin()]
        plt.scatter(max_sharpe_port['volatility'], max_sharpe_port['return'], marker='*', color='red', s=200, label='Max Sharpe Ratio')
        plt.scatter(min_vol_port['volatility'], min_vol_port['return'], marker='P', color='blue', s=200, label='Min Volatility')

        plt.title('Efficient Frontier')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.colorbar(sc, label='Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_weight_history(weight_df: pd.DataFrame, top_n: int = 10):
        """
        Plots the portfolio's asset allocation over time as a stacked area chart.

        Args:
            weight_df: DataFrame of historical weights.
            top_n: Number of top assets to show individually. Others are grouped.
        """
        mean_weights = weight_df.abs().mean().sort_values(ascending=False)
        top_assets = mean_weights.head(top_n).index
        
        plot_df = weight_df[top_assets].copy()
        if len(mean_weights) > top_n:
            plot_df['Others'] = weight_df.drop(columns=top_assets).sum(axis=1)

        fig, ax = plt.subplots(figsize=(12, 7))
        plot_df.plot.area(stacked=True, colormap='viridis', ax=ax)
        
        ax.set_title('Portfolio Allocation Over Time')
        ax.set_ylabel('Weight')
        ax.set_xlabel('Date')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_approximation_errors(error_results_df: pd.DataFrame):
        """
        Visualizes the trade-off between subsample size and backtest error.

        Args:
            error_results_df: The DataFrame output from `EfficiencyAnalyzer.analyze_approximation_error`.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Subsample Ratio')
        ax1.set_ylabel('Mean CV Score', color=color)
        ax1.plot(error_results_df['ratio'], error_results_df['mean_score'], 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Absolute Error', color=color)
        ax2.plot(error_results_df['ratio'], error_results_df['absolute_error'], 's--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Backtest Score and Error vs. Subsample Size')
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_performance_degradation(is_performance: np.ndarray, 
                                   oos_performance: np.ndarray,
                                   title: str = 'Performance Degradation',
                                   ylabel: str = 'Sharpe Ratio') -> plt.Figure:
        """
        Plot performance degradation from in-sample to out-of-sample.
        """
        if len(is_performance) != len(oos_performance):
            raise ValueError("In-sample and out-of-sample arrays must be the same length")
        
        data = pd.DataFrame({
            'IS': is_performance,
            'OOS': oos_performance,
            'Degradation': is_performance - oos_performance
        }).sort_values('IS', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        x = np.arange(len(data))
        ax1.plot(x, data['IS'], 'bo-', label='In-Sample', markersize=8, alpha=0.7)
        ax1.plot(x, data['OOS'], 'ro-', label='Out-of-Sample', markersize=8, alpha=0.7)
        
        X = sm.add_constant(data['IS'])
        model = sm.OLS(data['OOS'], X).fit()
        ax1.plot(x, model.predict(X), 'g--', label=f'Regression (slope={model.params.iloc[1]:.2f})', linewidth=2)
        
        rank_corr, _ = spearmanr(data['IS'], data['OOS'])
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title(f'{title}\nRank Correlation: {rank_corr:.2f}', fontsize=15)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_xlabel('Strategy Rank (sorted by In-Sample Performance)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        
        colors = ['g' if x <= 0 else 'r' for x in data['Degradation']]
        ax2.bar(x, data['Degradation'], color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Degradation\n(IS - OOS)', fontsize=12)
        ax2.set_xlabel('Strategy Rank (sorted by In-Sample Performance)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        stats_text = (f"Mean Degradation: {data['Degradation'].mean():.4f}\n"
                      f"Median Degradation: {data['Degradation'].median():.4f}\n"
                      f"% Strategies with Degradation: {np.mean(data['Degradation'] > 0)*100:.1f}%")
        ax2.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_pbo_analysis(is_performance: np.ndarray, 
                        oos_performance: np.ndarray,
                        n_samples: int = 1000,
                        title: str = 'Probability of Backtest Overfitting Analysis') -> plt.Figure:
        """
        Plot Probability of Backtest Overfitting (PBO) analysis.
        """
        if len(is_performance) != len(oos_performance):
            raise ValueError("In-sample and out-of-sample arrays must be the same length")
        
        pbo = PerformanceStatistics.probability_of_backtest_overfitting(is_performance, oos_performance)
        
        pbo_samples = []
        n_strategies = len(is_performance)
        for _ in range(n_samples):
            indices = np.random.choice(n_strategies, n_strategies, replace=True)
            pbo_samples.append(PerformanceStatistics.probability_of_backtest_overfitting(is_performance[indices], oos_performance[indices]))
        
        pbo_ci = np.percentile(pbo_samples, [2.5, 97.5])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.histplot(pbo_samples, ax=ax1, kde=True, bins=30, color='skyblue')
        ax1.axvline(x=pbo, color='red', linestyle='--', linewidth=2, label=f'PBO = {pbo:.4f}')
        ax1.axvline(x=pbo_ci[0], color='green', linestyle='--', linewidth=2, label=f'95% CI: [{pbo_ci[0]:.4f}, {pbo_ci[1]:.4f}]')
        ax1.axvline(x=pbo_ci[1], color='green', linestyle='--', linewidth=2)
        ax1.set_title(f'PBO Bootstrap Distribution', fontsize=12)
        ax1.set_xlabel('Probability of Backtest Overfitting', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        ax2.scatter(is_performance, oos_performance, color='blue', alpha=0.7)
        min_val, max_val = min(np.min(is_performance), np.min(oos_performance)), max(np.max(is_performance), np.max(oos_performance))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='45° line')
        
        X = sm.add_constant(is_performance)
        model = sm.OLS(oos_performance, X).fit()
        ax2.plot(is_performance, model.predict(X), 'r-', linewidth=2, label=f'Regression (slope={model.params.iloc[1]:.2f})')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('In-Sample vs. Out-of-Sample Performance', fontsize=12)
        ax2.set_xlabel('In-Sample Performance', fontsize=10)
        ax2.set_ylabel('Out-of-Sample Performance', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        return fig

    @staticmethod
    def plot_minimum_track_record_length(sharpe_ratio: float,
                                       skew: float = 0.0,
                                       kurtosis: float = 3.0,
                                       target_sharpe_ratios: Optional[List[float]] = None,
                                       title: str = 'Minimum Track Record Length Analysis') -> plt.Figure:
        """
        Plot Minimum Track Record Length (MinTRL) for different target Sharpe ratios.
        """
        if target_sharpe_ratios is None:
            target_sharpe_ratios = [0, 0.25, 0.5, 0.75, 1.0]
        
        target_sharpe_ratios = [sr for sr in target_sharpe_ratios if sr < sharpe_ratio]
        
        if not target_sharpe_ratios:
            raise ValueError("No valid target Sharpe ratios (must be less than observed Sharpe ratio)")
        
        conf_levels = np.linspace(0.5, 0.99, 20)
        min_trls = np.zeros((len(target_sharpe_ratios), len(conf_levels)))
        
        for i, target_sr in enumerate(target_sharpe_ratios):
            for j, conf in enumerate(conf_levels):
                min_trls[i, j] = PerformanceStatistics.minimum_track_record_length(target_sr, sharpe_ratio, conf, skew, kurtosis)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, target_sr in enumerate(target_sharpe_ratios):
            ax.plot(conf_levels * 100, min_trls[i], linewidth=2, label=f'Target SR = {target_sr:.2f}')
        
        time_periods = [(0.25, '3m'), (0.5, '6m'), (1, '1y'), (2, '2y'), (3, '3y'), (5, '5y')]
        for years, label in time_periods:
            ax.axhline(y=years, color='gray', linestyle='--', alpha=0.5)
            ax.text(50, years, label, va='center', ha='center', backgroundcolor='white', fontsize=8, alpha=0.7)
        
        ax.set_title(f'Minimum Track Record Length for SR={sharpe_ratio:.2f}', fontsize=12)
        ax.set_xlabel('Confidence Level (%)', fontsize=10)
        ax.set_ylabel('Track Record Length (years)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, title='Target Sharpe Ratio')
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_deflated_sharpe_ratio(sharpe_ratio: float,
                                 n_trials_range: List[int],
                                 n_obs: int,
                                 annualization_factor: float = 252,
                                 skew: float = 0.0,
                                 kurtosis: float = 3.0,
                                 title: str = 'Deflated Sharpe Ratio Analysis') -> plt.Figure:
        """
        Plot Deflated Sharpe Ratio (DSR) for different numbers of trials.
        """
        dsrs = [PerformanceStatistics.deflated_sharpe_ratio(sharpe_ratio, n, n_obs, skew, kurtosis, annualization_factor) for n in n_trials_range]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(n_trials_range, dsrs, 'b-', linewidth=2)
        ax.axhline(y=sharpe_ratio, color='green', linestyle='--', alpha=0.7, label=f'Original SR = {sharpe_ratio:.2f}')
        ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='p-value=0.05')
        ax.axhline(y=0.0, color='red', linestyle='-', alpha=0.5, label='SR = 0 (no skill)')
        
        ax.set_title('Deflated Sharpe Ratio vs. Number of Trials', fontsize=12)
        ax.set_xlabel('Number of Trials (strategy variations tested)', fontsize=10)
        ax.set_ylabel('Deflated Sharpe Ratio (p-value)', fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_stochastic_dominance(returns1: np.ndarray, returns2: np.ndarray,
                                label1: str = 'Strategy 1',
                                label2: str = 'Strategy 2',
                                title: str = 'Stochastic Dominance Analysis') -> plt.Figure:
        """
        Plot stochastic dominance analysis.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        sns.histplot(returns1, kde=True, ax=ax1, color='blue', alpha=0.5, label=label1)
        sns.histplot(returns2, kde=True, ax=ax1, color='red', alpha=0.5, label=label2)
        ax1.set_title('Return Distributions', fontsize=12)
        ax1.legend()

        x_combined = np.sort(np.concatenate([returns1, returns2]))
        cdf1 = np.searchsorted(np.sort(returns1), x_combined, side='right') / len(returns1)
        cdf2 = np.searchsorted(np.sort(returns2), x_combined, side='right') / len(returns2)
        ax2.step(x_combined, cdf1, 'b-', label=label1)
        ax2.step(x_combined, cdf2, 'r-', label=label2)
        ax2.set_title('Cumulative Distribution Functions (First-Order SD)', fontsize=12)
        ax2.legend()
        
        fig.suptitle(title, fontsize=15)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cusum(cusum_stats: pd.DataFrame, thresholds: pd.Series, title: str = 'CUSUM Test'):
        """
        Plots the results of a CUSUM test.

        Args:
            cusum_stats: DataFrame with CUSUM statistics (e.g., 'S+' and 'S-').
            thresholds: Series of threshold values.
            title: The title for the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(cusum_stats.index, cusum_stats['S+'], label='S+', color='blue')
        ax.plot(cusum_stats.index, cusum_stats['S-'], label='S-', color='green')
        ax.plot(thresholds.index, thresholds, label='Threshold', color='red', linestyle='--')
        ax.plot(thresholds.index, -thresholds, color='red', linestyle='--')
        
        ax.set_title(title)
        ax.set_ylabel('CUSUM Statistic')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig