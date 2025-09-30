import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from afml.validation.statistics import PerformanceStatistics
from afml.utils.visualization import VisualizationTools
from afml.data.simulation import BacktestSimulator


# Example usage
def example_with_synthetic_data():
    """
    Example with synthetic data to demonstrate backtest statistics techniques.
    """
    np.random.seed(42)
    
    print("1. Demonstrating Backtest Overfitting")
    overfitting_results = BacktestSimulator.demonstrate_backtest_overfitting(
        n_strategies=100, n_periods=504, is_fraction=0.5, random_state=42
    )
    
    print(f"PBO: {overfitting_results['pbo']:.4f}")
    print(f"DSR: {overfitting_results['dsr']:.4f}")
    
    plt.figure(overfitting_results['pbo_fig'].number)
    plt.show()
    plt.figure(overfitting_results['degradation_fig'].number)
    plt.show()

    print("\n2. Minimum Track Record Length Analysis")
    sharpe_ratio = 1.0
    mintrl_fig = VisualizationTools.plot_minimum_track_record_length(
        sharpe_ratio, target_sharpe_ratios=[0, 0.25, 0.5, 0.75]
    )
    plt.figure(mintrl_fig.number)
    plt.show()

    print("\n3. Deflated Sharpe Ratio Analysis")
    dsr_fig = VisualizationTools.plot_deflated_sharpe_ratio(
        sharpe_ratio, n_trials_range=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        n_obs=252
    )
    plt.figure(dsr_fig.number)
    plt.show()

    print("\n4. Drawdown Analysis")
    n_periods = 252
    returns = np.random.normal(0.001, 0.01, n_periods)
    returns[50:100] = np.random.normal(-0.002, 0.012, 50)
    dd_fig = VisualizationTools.plot_portfolio_performance(pd.DataFrame({'returns': returns}))
    plt.figure(dd_fig.number)
    plt.show()
    
    print("\n5. Stochastic Dominance Analysis")
    returns1 = np.random.normal(0.001, 0.01, n_periods)
    returns2 = np.random.normal(0.0005, 0.012, n_periods)
    sd_fig = VisualizationTools.plot_stochastic_dominance(returns1, returns2, 'Strategy 1', 'Strategy 2')
    plt.figure(sd_fig.number)
    plt.show()

    return overfitting_results

if __name__ == "__main__":
    results = example_with_synthetic_data()