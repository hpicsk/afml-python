import matplotlib.pyplot as plt

from afml.data.simulation import SyntheticData
from afml.core.backtester import VectorizedBacktester
from afml.portfolio import strategies, analysis

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Example script demonstrating strategy risk analysis, including
    risk contributions and concentration, from Chapter 15.
    """
    logger.info("--- Strategy Risk Analysis Demonstration (Chapter 15) ---")

    # 1. Data Generation
    logger.info("\n[1] Generating synthetic price data...")
    data_generator = SyntheticData(seed=42)
    prices = data_generator.generate_prices(n_assets=25, n_days=1000)

    # 2. Backtesting a Strategy to get weights
    logger.info("\n[2] Backtesting an Equal Weight strategy to generate weight history...")
    backtester = VectorizedBacktester(prices=prices, initial_capital=1_000_000)
    
    backtester.run(
        strategy=strategies.equal_weight_strategy,
        rebalance_freq='QE', # Quarterly rebalance
        lookback_periods=1 # No lookback needed for equal weight
    )

    # 3. Risk Analysis
    logger.info("\n[3] Analyzing portfolio risk contributions and concentration...")
    portfolio_analyzer = analysis.PortfolioAnalytics(
        returns=prices.pct_change(),
        weights=backtester.weight_history
    )
    
    # Calculate rolling risk contributions
    risk_contributions = portfolio_analyzer.calculate_risk_contributions(lookback=252)
    
    # Calculate risk concentration (HHI)
    risk_concentration = portfolio_analyzer.calculate_risk_concentration(risk_contributions)

    # 4. Visualization
    logger.info("\n[4] Visualizing results...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot risk contributions (as a stacked area chart)
    risk_contributions.plot.area(ax=ax1, stacked=True, colormap='viridis')
    ax1.set_title('Rolling Asset Risk Contribution')
    ax1.set_ylabel('Percentage of Total Risk')
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Plot risk concentration (HHI)
    risk_concentration.plot(ax=ax2, color='red')
    ax2.set_title('Portfolio Risk Concentration (Herfindahl-Hirschman Index)')
    ax2.set_ylabel('HHI')
    ax2.set_xlabel('Date')
    
    # Theoretical HHI for equal risk contribution
    n_assets = len(prices.columns)
    equal_hhi = 1 / n_assets
    ax2.axhline(equal_hhi, color='gray', linestyle='--', label=f'Equal Risk HHI (1/N = {equal_hhi:.3f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
