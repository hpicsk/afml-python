import pandas as pd
from .optimizer import PortfolioOptimizer

# This file contains example strategy functions that can be passed to a backtester.
# A strategy function must have the following signature:
# strategy(hist_prices, hist_returns, current_date) -> pd.Series
# It should return a Series of target weights for each asset.

def equal_weight_strategy(hist_prices: pd.DataFrame, hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> pd.Series:
    """A simple equal weight strategy."""
    active_assets = hist_prices.iloc[-1].dropna().index
    return pd.Series(1.0 / len(active_assets), index=active_assets)

def minimum_variance_strategy(hist_prices: pd.DataFrame, hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> pd.Series:
    """A minimum variance portfolio strategy."""
    optimizer = PortfolioOptimizer(returns=hist_returns, risk_model='ledoit_wolf')
    return optimizer.minimum_variance(max_position_size=0.2)

def momentum_strategy(hist_prices: pd.DataFrame, hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> pd.Series:
    """
    A simple cross-sectional momentum strategy.
    Goes long the top quartile of assets based on 12-month-minus-1-month momentum.
    """
    # if len(hist_returns) < 252:
    #     return equal_weight_strategy(hist_prices, hist_returns, current_date)
    if len(hist_returns) < 252:
        raise ValueError("Not enough data to calculate momentum strategy. Need at least 252 days of returns.")

    # 12m-1m momentum
    momentum_period = hist_returns.iloc[-252:-21]
    momentum_scores = (1 + momentum_period).prod() - 1

    top_quartile = momentum_scores.nlargest(len(momentum_scores) // 4)
    
    weights = pd.Series(0.0, index=hist_prices.columns)
    if not top_quartile.empty:
        weights[top_quartile.index] = 1.0 / len(top_quartile)
        
    return weights

def max_sharpe_strategy(hist_prices: pd.DataFrame, hist_returns: pd.DataFrame, current_date: pd.Timestamp) -> pd.Series:
    """A maximum Sharpe ratio portfolio strategy."""
    optimizer = PortfolioOptimizer(returns=hist_returns, risk_model='ledoit_wolf')
    return optimizer.maximum_sharpe(max_position_size=0.2) 