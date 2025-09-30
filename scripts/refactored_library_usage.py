import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Import from the refactored library
from afml.core.backtester import VectorizedBacktester
from afml.portfolio.optimizer import PortfolioOptimizer
from afml.portfolio.analysis import PortfolioAnalytics
from afml.validation.dangers import DangerDetector
from afml.validation.robustness import RobustnessChecker
from afml.data.simulation import SyntheticData
from afml.utils.visualization import VisualizationTools
from afml.portfolio import strategies


def main():
    """
    An example script demonstrating the use of the refactored backtesting library.
    """
    print("--- Refactored Backtesting Library Demonstration ---")

    # 1. Data Generation
    print("\n[1] Generating synthetic asset prices...")
    data_generator = SyntheticData(seed=42)
    prices = data_generator.generate_prices(n_assets=20, n_days=1260) # ~5 years
    returns = prices.pct_change().dropna()

    # 2. Portfolio Optimization
    print("\n[2] Performing portfolio optimization...")
    # Use the last 2 years of data for optimization
    optimization_returns = returns.iloc[-504:]
    optimizer = PortfolioOptimizer(returns=optimization_returns, risk_model='ledoit_wolf')
    
    # Get weights for different strategies
    min_var_weights = optimizer.minimum_variance(max_position_size=0.15)
    max_sharpe_weights = optimizer.maximum_sharpe(max_position_size=0.15)
    
    print(f"Min Variance Portfolio (Top 5): \n{min_var_weights.nlargest(5)}")
    print(f"\nMax Sharpe Portfolio (Top 5): \n{max_sharpe_weights.nlargest(5)}")

    # 3. Backtesting a Strategy
    print("\n[3] Running a vectorized backtest with a momentum strategy...")
    backtester = VectorizedBacktester(prices=prices, initial_capital=100_000)
    
    # Run the backtest using one of the predefined strategies
    results_df = backtester.run(strategy=strategies.momentum_strategy, rebalance_freq='QE')
    
    print("\nBacktest Performance Metrics:")
    for metric, value in backtester.metrics.items():
        print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")

    # 4. In-depth Performance Analysis
    print("\n[4] Analyzing portfolio performance...")
    analytics = PortfolioAnalytics(returns=returns, weights=backtester.weight_history)
    summary = analytics.performance_summary()
    print("\nDetailed Performance Summary:")
    for metric, value in summary.items():
        if value is not None:
            print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")

    # 5. Validation and Robustness Checks
    print("\n[5] Performing validation and robustness checks...")
    
    # Create a simple binary target for demonstration
    X = returns.shift(1).dropna()
    y = (returns.iloc[1:] > 0).astype(int).iloc[:, 0] # Predict direction of first asset
    X = X.loc[y.index]
    
    train_size = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Check for data leakage
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    DangerDetector.detect_data_leakage(model, X_train, y_train, X_test, y_test)
    
    # Check for parameter stability
    param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100]}
    RobustnessChecker.parameter_stability_test(RandomForestClassifier, param_grid, X, y, n_periods=4)
    
    # 6. Visualization
    print("\n[6] Generating visualizations...")
    VisualizationTools.plot_portfolio_performance(results_df)
    VisualizationTools.plot_weight_history(backtester.weight_history)
    
    # Generate and plot an efficient frontier
    frontier_df = optimizer.efficient_frontier(n_points=50)
    VisualizationTools.plot_efficient_frontier(frontier_df)


if __name__ == "__main__":
    main() 