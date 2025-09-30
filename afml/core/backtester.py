import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, Optional, List, Dict, Any, Tuple

from ..utils.performance import PerformanceMonitor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorizedBacktester:
    """
    A powerful backtesting engine that combines a vectorized approach for speed
    with a detailed simulation of portfolio effects like transaction costs and market impact.

    This class merges the high-level structure of chap14.PortfolioBacktester with the
    detailed execution logic of chap14.PortfolioSimulator.
    """

    def __init__(self,
                 prices: pd.DataFrame,
                 initial_capital: float = 1_000_000.0,
                 commission_rate: float = 0.001,
                 market_impact_model: Optional[Callable] = None):
        """
        Initialize the VectorizedBacktester.

        Args:
            prices: DataFrame of asset prices with dates as index and assets as columns.
            initial_capital: Initial capital for the backtest.
            commission_rate: Transaction cost as a fraction of trade value.
            market_impact_model: A function to calculate market impact.
                                 Signature: func(trade_volume, avg_daily_volume, price) -> cost_fraction
        """
        self.prices = prices.copy()
        self.returns = prices.pct_change().fillna(0).infer_objects(copy=False)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.market_impact_model = market_impact_model or self._default_market_impact
        self.monitor = PerformanceMonitor()

        # Initialize tracking variables
        self._reset_state()

        logger.info(f"Initialized VectorizedBacktester with {len(prices.columns)} assets.")

    def _reset_state(self):
        """Resets the backtester's state for a new run."""
        self.current_weights = pd.Series(0.0, index=self.prices.columns)
        self.position_history = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        self.weight_history = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        self.portfolio_value_history = pd.Series(self.initial_capital, index=self.prices.index)
        self.turnover_history = pd.Series(0.0, index=self.prices.index, dtype=float)
        self.trade_costs_history = pd.Series(0.0, index=self.prices.index, dtype=float)
        self.returns_history = pd.Series(0.0, index=self.prices.index, dtype=float)
        self.metrics = {}
        logger.debug("Backtester state has been reset.")

    def _default_market_impact(self, trade_value: float, avg_daily_value: float) -> float:
        """
        A simple square-root market impact model.

        Args:
            trade_value: The notional value of the trade.
            avg_daily_value: The average daily notional value traded for the asset.

        Returns:
            The total cost of market impact for the trade.
        """
        if avg_daily_value <= 0:
            return 0.0
        # Impact is proportional to the square root of the trade's participation rate
        participation_rate = trade_value / avg_daily_value
        impact_percentage = 0.1 * np.sqrt(participation_rate)  # Example: 10 bps for 1% of ADV
        return trade_value * impact_percentage

    def run(self,
            strategy: Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], pd.Series],
            rebalance_freq: str = 'ME',
            lookback_periods: int = 252) -> pd.DataFrame:
        """
        Run a portfolio backtest with regular rebalancing.

        Args:
            strategy: A function that returns target weights.
                      Signature: strategy(hist_prices, hist_returns, current_date) -> pd.Series
            rebalance_freq: Pandas frequency string for rebalancing (e.g., 'ME', 'QE', 'W').
            lookback_periods: Number of periods to use for historical calculations.

        Returns:
            A DataFrame with the portfolio's performance history.
        """
        with self.monitor.timer("full_backtest_run"):
            self._reset_state()

            # Determine rebalance dates
            rebalance_dates = self.prices.resample(rebalance_freq).last().index
            rebalance_dates = rebalance_dates[rebalance_dates.isin(self.prices.index)]

            last_weights = pd.Series(0.0, index=self.prices.columns)
            portfolio_value = self.initial_capital

            for date in tqdm(self.prices.index, desc="Running Backtest"):
                # Update portfolio value based on last period's returns
                if date > self.prices.index[0]:
                    prev_date = self.prices.index[self.prices.index.get_loc(date) - 1]
                    daily_return = (self.returns.loc[date] * last_weights).sum()
                    portfolio_value *= (1 + daily_return)
                    self.returns_history.loc[date] = float(daily_return)

                # Rebalance if it's a rebalance date and we have enough data
                if date in rebalance_dates and self.prices.index.get_loc(date) >= lookback_periods:
                    with self.monitor.timer("rebalancing_step"):
                        hist_slice = slice(max(0, self.prices.index.get_loc(date) - lookback_periods),
                                           self.prices.index.get_loc(date) + 1)
                        hist_prices = self.prices.iloc[hist_slice]
                        hist_returns = self.returns.iloc[hist_slice]

                        # Get target weights from the strategy
                        target_weights = strategy(hist_prices, hist_returns, date)
                        target_weights = target_weights.reindex(self.prices.columns).fillna(0.0).infer_objects(copy=False)
                        target_weights /= target_weights.sum() # Normalize to 1

                        # Calculate and deduct transaction costs
                        turnover = (target_weights - last_weights).abs().sum() / 2
                        # A simple cost model for demonstration
                        # A more realistic model is in `calculate_transaction_costs` but requires volume data.
                        costs = portfolio_value * turnover * self.commission_rate
                        portfolio_value -= costs

                        self.turnover_history.loc[date] = float(turnover)
                        self.trade_costs_history.loc[date] = float(costs)

                        last_weights = target_weights

                self.portfolio_value_history.loc[date] = portfolio_value
                self.weight_history.loc[date] = last_weights.astype(float)

        self.metrics = self._calculate_metrics()
        results = self._compile_results()
        logger.info(f"Backtest complete. Final portfolio value: {results['portfolio_value'].iloc[-1]:,.2f}")
        return results

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculates summary performance metrics for the backtest."""
        returns = self.returns_history.dropna()
        if returns.empty:
            return {metric: 0.0 for metric in ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']}

        total_return = (self.portfolio_value_history.iloc[-1] / self.initial_capital) - 1
        n_years = (returns.index[-1] - returns.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'average_turnover': self.turnover_history[self.turnover_history > 0].mean(),
            'total_costs_pct': self.trade_costs_history.sum() / self.initial_capital
        }

    def _compile_results(self) -> pd.DataFrame:
        """Compiles the final results DataFrame."""
        return pd.DataFrame({
            'portfolio_value': self.portfolio_value_history,
            'returns': self.returns_history,
            'turnover': self.turnover_history,
            'costs': self.trade_costs_history
        }).dropna(subset=['portfolio_value'])


class StrategyBacktester:
    """
    Backtester for comparing different bet sizing strategies.
    """
    def __init__(self, initial_equity: float = 10000.0,
                transaction_cost: float = 0.0,
                slippage: float = 0.0):
        self.initial_equity = initial_equity
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.equity_curve = None
        self.trades = None

    def backtest(self, returns: pd.Series, positions: pd.Series) -> Dict[str, Any]:
        equity = pd.Series(0.0, index=returns.index)
        equity.iloc[0] = self.initial_equity
        
        trades = []
        current_pos = 0
        
        for i in range(1, len(returns)):
            delta = positions.iloc[i] - current_pos
            cost = abs(delta) * (self.transaction_cost + self.slippage) * equity.iloc[i-1]
            
            equity.iloc[i] = equity.iloc[i-1] * (1 + returns.iloc[i] * current_pos) - cost
            
            if delta != 0:
                trades.append({'date': returns.index[i], 'position': positions.iloc[i], 'delta': delta, 'cost': cost})
            
            current_pos = positions.iloc[i]

        self.equity_curve = equity
        self.trades = pd.DataFrame(trades)
        
        from ..validation.statistics import PerformanceStatistics
        stats = PerformanceStatistics()
        
        return {
            'equity_curve': equity,
            'trades': self.trades,
            'sharpe_ratio': stats.sharpe_ratio_from_positions(returns.values, positions.values),
            'max_drawdown': stats.drawdown_from_positions(returns.values, positions.values)[0],
            'profit_factor': stats.profit_factor(returns.values, positions.values)
        } 