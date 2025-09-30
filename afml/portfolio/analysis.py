import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Dict, Any
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """
    Provides tools for in-depth analysis of a portfolio's historical performance,
    risk exposures, and factor attributions.
    (Derived from chap14.PortfolioAnalytics)
    """

    def __init__(self, returns: pd.DataFrame, weights: pd.DataFrame):
        """
        Initialize the PortfolioAnalytics.

        Args:
            returns: DataFrame of asset returns.
            weights: DataFrame of portfolio weights over time.
        """
        common_dates = returns.index.intersection(weights.index)
        self.asset_returns = returns.loc[common_dates]
        self.weights = weights.loc[common_dates]

        self.portfolio_returns = (self.asset_returns * self.weights.shift(1)).sum(axis=1).dropna()
        logger.info(f"Initialized PortfolioAnalytics for {len(common_dates)} periods.")

    def performance_summary(self, risk_free_rate: float = 0.0,
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculates a comprehensive summary of performance metrics.

        Args:
            risk_free_rate: Annualized risk-free rate.
            benchmark_returns: Optional series of benchmark returns for relative metrics.

        Returns:
            A dictionary of performance metrics.
        """
        returns = self.portfolio_returns
        if returns.empty:
            return {metric: 0.0 for metric in ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']}

        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()

        summary = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
        }

        # Benchmark-relative metrics
        if benchmark_returns is not None:
            common_idx = returns.index.intersection(benchmark_returns.index)
            active_returns = returns[common_idx] - benchmark_returns[common_idx]
            
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            X = sm.add_constant(benchmark_returns[common_idx])
            model = sm.OLS(returns[common_idx], X).fit()
            beta, alpha = model.params.get('const', 0.0), model.params.get(benchmark_returns.name, 0.0)

            summary.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha * 252 # Annualize
            })

        return summary

    def calculate_risk_contributions(self, lookback: int = 252) -> pd.DataFrame:
        """
        Calculates the contribution of each asset to total portfolio risk over time.

        Args:
            lookback: The rolling window period for covariance estimation.

        Returns:
            A DataFrame of percentage risk contributions for each asset.
        """
        risk_contrib_history = pd.DataFrame(index=self.weights.index, columns=self.weights.columns)
        
        for date in tqdm(self.weights.index[lookback:], desc="Calculating Risk Contributions"):
            current_weights = self.weights.loc[date]
            if current_weights.sum() == 0:
                continue

            hist_returns = self.asset_returns.loc[:date].iloc[-lookback:]
            cov_matrix = hist_returns.cov()

            portfolio_vol = np.sqrt(current_weights.T @ cov_matrix @ current_weights)
            if portfolio_vol > 0:
                # Marginal Contribution to Risk (MCR)
                mcr = (cov_matrix @ current_weights) / portfolio_vol
                # Risk Contribution (RC = weight * MCR)
                rc = current_weights * mcr
                # Percentage Risk Contribution
                prc = rc / portfolio_vol
                risk_contrib_history.loc[date] = prc
                
        return risk_contrib_history.dropna(how='all')

    def calculate_risk_concentration(self, risk_contributions: pd.DataFrame) -> pd.Series:
        """
        Calculates the Herfindahl-Hirschman Index (HHI) of risk contributions
        to measure risk concentration.

        Args:
            risk_contributions: DataFrame of percentage risk contributions from
                                `calculate_risk_contributions`.

        Returns:
            A Series of HHI values over time. A higher value means more concentrated risk.
        """
        # HHI is the sum of the squares of the contributions
        hhi = (risk_contributions**2).sum(axis=1)
        logger.info("Calculated risk concentration (HHI).")
        return hhi

    def factor_attribution(self, factor_returns: pd.DataFrame, rolling_window: int = 63) -> pd.DataFrame:
        """
        Performs rolling factor regression to attribute portfolio returns to known factors.

        Args:
            factor_returns: DataFrame of factor returns.
            rolling_window: The window size for the rolling regressions.

        Returns:
            A DataFrame showing the contribution of each factor and alpha to the portfolio's return.
        """
        common_idx = self.portfolio_returns.index.intersection(factor_returns.index)
        portfolio_rets = self.portfolio_returns[common_idx]
        factor_rets = factor_returns[common_idx]
        
        exposures = pd.DataFrame(index=common_idx, columns=list(factor_rets.columns) + ['alpha'])
        
        for i in range(rolling_window, len(common_idx)):
            window = slice(i - rolling_window, i)
            X = sm.add_constant(factor_rets.iloc[window])
            y = portfolio_rets.iloc[window]
            model = sm.OLS(y, X).fit()
            exposures.iloc[i] = model.params
            
        exposures = exposures.dropna(how='all')
        
        # Calculate return contributions
        # Contribution = Exposure(t-1) * Factor Return(t)
        contributions = exposures.shift(1).mul(factor_rets).dropna(how='all')
        contributions['alpha'] = exposures['alpha'].shift(1) / 252 # Daily alpha contribution
        contributions['unexplained'] = portfolio_rets - contributions.sum(axis=1)
        contributions['portfolio_return'] = portfolio_rets
        
        return contributions 