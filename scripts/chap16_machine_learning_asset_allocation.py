import matplotlib.pyplot as plt

from afml.data.simulation import SyntheticData
from afml.core.backtester import VectorizedBacktester
from afml.portfolio import strategies, analysis
from afml.utils.visualization import VisualizationTools

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Example script demonstrating portfolio optimization and backtesting
    using the refactored library.
    """
    logger.info("--- Portfolio Optimization Demonstration ---")

    # 1. Data Generation
    logger.info("\n[1] Generating synthetic price data...")
    data_generator = SyntheticData(seed=42)
    prices = data_generator.generate_prices(n_assets=20, n_days=750)

    # 2. Backtesting a Strategy
    logger.info("\n[2] Backtesting a Minimum Variance strategy...")
    backtester = VectorizedBacktester(prices=prices, initial_capital=1_000_000)
    
    # Run the backtest using one of the predefined strategies
    results = backtester.run(
        strategy=strategies.minimum_variance_strategy,
        rebalance_freq='ME',
        lookback_periods=252
    )

    print("\nBacktest Performance Metrics:")
    for metric, value in backtester.metrics.items():
        print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")

    # 3. Performance Analysis
    logger.info("\n[3] Analyzing portfolio performance...")
    # The 'results' DataFrame from the backtester can be used for more detailed analysis
    # For example, let's create an analytics object
    portfolio_analyzer = analysis.PortfolioAnalytics(
        returns=prices.pct_change(), # Asset returns
        weights=backtester.weight_history # Weights from the backtest
    )
    
    # Get a performance summary
    summary = portfolio_analyzer.performance_summary()
    print("\nPortfolio Analytics Summary:")
    print(f"  - Total Return: {summary['total_return']:.4f}")
    print(f"  - Sharpe Ratio: {summary['sharpe_ratio']:.2f}")

    # 4. Visualization
    logger.info("\n[4] Visualizing results...")
    
    # Use the library's visualization tools
    fig1 = VisualizationTools.plot_portfolio_performance(results)
    fig1.suptitle("Minimum Variance Strategy Performance")
    
    fig2 = VisualizationTools.plot_weight_history(backtester.weight_history)
    fig2.suptitle("Minimum Variance Weight Allocation")

    plt.show()


if __name__ == "__main__":
    main()