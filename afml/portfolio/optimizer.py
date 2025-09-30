import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import ledoit_wolf
from typing import Optional, List

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Provides a suite of portfolio optimization methods.
    (Derived from chap14.PortfolioOptimizer)
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 risk_model: str = 'ledoit_wolf',
                 risk_free_rate: float = 0.0):
        """
        Initialize the PortfolioOptimizer.

        Args:
            returns: DataFrame of asset returns.
            risk_model: The risk model for covariance estimation ('empirical' or 'ledoit_wolf').
            risk_free_rate: Annualized risk-free rate.
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate

        self.expected_returns = self._estimate_expected_returns()
        self.covariance_matrix = self._estimate_covariance(risk_model)
        logger.info(f"Initialized PortfolioOptimizer with {self.n_assets} assets using '{risk_model}' risk model.")

    def _estimate_expected_returns(self) -> pd.Series:
        """Estimates expected returns (annualized simple mean)."""
        return self.returns.mean() * 252

    def _estimate_covariance(self, risk_model: str) -> pd.DataFrame:
        """Estimates the covariance matrix (annualized)."""
        if risk_model == 'ledoit_wolf':
            cov_raw, _ = ledoit_wolf(self.returns)
            cov = pd.DataFrame(cov_raw, index=self.assets, columns=self.assets)
        elif risk_model == 'empirical':
            cov = self.returns.cov()
        else:
            raise ValueError(f"Unknown risk model: {risk_model}")
        return cov * 252

    def minimum_variance(self,
                         max_position_size: float = 1.0,
                         target_return: Optional[float] = None) -> pd.Series:
        """
        Finds the minimum variance portfolio weights.

        Args:
            max_position_size: Maximum weight for any single asset.
            target_return: Optional target return for the portfolio.

        Returns:
            A Series of optimal portfolio weights.
        """
        w = cp.Variable(self.n_assets)
        objective = cp.Minimize(cp.quad_form(w, self.covariance_matrix.values))
        constraints = [cp.sum(w) == 1, w >= 0, w <= max_position_size]
        if target_return is not None:
            constraints.append(self.expected_returns.values @ w >= target_return)

        try:
            problem = cp.Problem(objective, constraints)
            problem.solve()
            if problem.status == 'optimal':
                return pd.Series(w.value, index=self.assets)
            else:
                logger.warning(f"Min variance optimization failed: {problem.status}. Returning equal weights.")
        except Exception as e:
            logger.error(f"Min variance optimization error: {e}. Returning equal weights.")

        return pd.Series(1/self.n_assets, index=self.assets)

    def maximum_sharpe(self, max_position_size: float = 1.0) -> pd.Series:
        """
        Finds the maximum Sharpe ratio portfolio weights (tangency portfolio).

        Args:
            max_position_size: Maximum weight for any single asset.

        Returns:
            A Series of optimal portfolio weights.
        """
        w = cp.Variable(self.n_assets)
        gamma = cp.Parameter(nonneg=True, value=1.0)  # Risk aversion
        excess_returns = self.expected_returns - self.risk_free_rate

        objective = cp.Maximize(excess_returns.values @ w - gamma * cp.quad_form(w, self.covariance_matrix.values))
        constraints = [cp.sum(w) == 1, w >= 0, w <= max_position_size]

        try:
            problem = cp.Problem(objective, constraints)
            problem.solve()
            if problem.status == 'optimal':
                return pd.Series(w.value, index=self.assets)
            else:
                logger.warning(f"Max Sharpe optimization failed: {problem.status}. Returning equal weights.")
        except Exception as e:
            logger.error(f"Max Sharpe optimization error: {e}. Returning equal weights.")

        return pd.Series(1/self.n_assets, index=self.assets)

    def risk_parity(self) -> pd.Series:
        """
        Calculates risk parity weights using an iterative algorithm.
        This method aims to have each asset contribute equally to the total portfolio risk.
        """
        w = np.ones(self.n_assets) / self.n_assets
        cov_matrix_np = self.covariance_matrix.values
        tolerance = 1e-6
        max_iter = 100

        for _ in range(max_iter):
            w_prev = w.copy()
            port_var = w.T @ cov_matrix_np @ w
            marginal_risk_contrib = cov_matrix_np @ w
            risk_contrib = w * marginal_risk_contrib / np.sqrt(port_var)

            # Update weights to move towards equal risk contribution
            w = risk_contrib**-1
            w /= w.sum() # Normalize

            if np.linalg.norm(w - w_prev) < tolerance:
                break
        
        return pd.Series(w, index=self.assets)


    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generates the efficient frontier by solving for minimum variance
        at different levels of target return.

        Args:
            n_points: The number of points to calculate on the frontier.

        Returns:
            A DataFrame containing the returns, volatilities, and Sharpe ratios for
            portfolios on the efficient frontier.
        """
        min_ret_weights = self.minimum_variance()
        min_ret = (min_ret_weights @ self.expected_returns).item()

        max_ret_weights = self.maximum_sharpe()
        max_ret = (max_ret_weights @ self.expected_returns).item()

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_volatilities = []

        for target in target_returns:
            weights = self.minimum_variance(target_return=target)
            vol = np.sqrt(weights.T @ self.covariance_matrix @ weights)
            frontier_volatilities.append(vol)

        frontier_df = pd.DataFrame({
            'return': target_returns,
            'volatility': frontier_volatilities
        })
        frontier_df['sharpe'] = (frontier_df['return'] - self.risk_free_rate) / frontier_df['volatility']
        return frontier_df 