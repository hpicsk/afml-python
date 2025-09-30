import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
import warnings

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticData:
    """
    A comprehensive utility for generating synthetic financial data for testing.
    (Combines chap12.SyntheticDataGenerator and chap14.PortfolioDataUtils)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.seed = seed
        if seed:
            np.random.seed(seed)

    def generate_prices(self,
                        n_assets: int = 20,
                        n_days: int = 1000,
                        start_date: str = '2020-01-01',
                        mean_return: float = 0.0001,
                        volatility: float = 0.01,
                        correlation: float = 0.2) -> pd.DataFrame:
        """
        Generates synthetic price data using a geometric Brownian motion model.

        Args:
            n_assets: Number of assets to generate.
            n_days: Number of trading days.
            start_date: The start date for the price series.
            mean_return: The mean daily return.
            volatility: The daily volatility.
            correlation: The correlation between asset returns.

        Returns:
            A DataFrame of synthetic prices.
        """
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        
        corr_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        cov_matrix = np.outer(np.full(n_assets, volatility), np.full(n_assets, volatility)) * corr_matrix
        
        returns = np.random.multivariate_normal(
            mean=np.full(n_assets, mean_return),
            cov=cov_matrix,
            size=n_days
        )
        
        prices = 100 * np.cumprod(1 + returns, axis=0)
        prices_df = pd.DataFrame(prices, index=dates, columns=[f'Asset_{i+1}' for i in range(n_assets)])
        logger.info(f"Generated synthetic prices for {n_assets} assets over {n_days} days.")
        return prices_df

    def generate_factor_returns(self,
                                n_factors: int = 3,
                                n_days: int = 1000,
                                start_date: str = '2020-01-01',
                                factor_volatility: float = 0.005,
                                factor_correlation: float = 0.0) -> pd.DataFrame:
        """
        Generates synthetic factor returns.

        Args:
            n_factors: Number of factors to generate.
            n_days: Number of trading days.
            start_date: The start date for the series.
            factor_volatility: The daily volatility of the factors.
            factor_correlation: The correlation between factors.

        Returns:
            A DataFrame of synthetic factor returns.
        """
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        
        factor_means = np.random.normal(0.0001, 0.0002, n_factors) # Small positive drift
        
        corr_matrix = np.full((n_factors, n_factors), factor_correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        cov_matrix = np.outer(np.full(n_factors, factor_volatility), np.full(n_factors, factor_volatility)) * corr_matrix
        
        returns = np.random.multivariate_normal(mean=factor_means, cov=cov_matrix, size=n_days)
        factor_df = pd.DataFrame(returns, index=dates, columns=[f'Factor_{i+1}' for i in range(n_factors)])
        logger.info(f"Generated synthetic returns for {n_factors} factors.")
        return factor_df

    def generate_features_from_prices(self, prices: pd.DataFrame, n_features: int = 10) -> pd.DataFrame:
        """
        Generates technical analysis style features from price data.

        Args:
            prices: A DataFrame of asset prices.
            n_features: The number of features to generate.

        Returns:
            A DataFrame of synthetic features.
        """
        returns = prices.pct_change()
        features = pd.DataFrame(index=prices.index)
        
        for i in range(n_features):
            feature_type = np.random.choice(['momentum', 'volatility', 'noise'])
            asset = np.random.choice(prices.columns)
            
            if feature_type == 'momentum':
                window = np.random.randint(5, 60)
                features[f'mom_{i+1}'] = returns[asset].rolling(window=window).mean()
            elif feature_type == 'volatility':
                window = np.random.randint(10, 100)
                features[f'vol_{i+1}'] = returns[asset].rolling(window=window).std()
            else: # noise
                features[f'noise_{i+1}'] = np.random.randn(len(prices))
        
        logger.info(f"Generated {n_features} features from prices.")
        return features.dropna()


class BacktestSimulator:
    """
    Simulate backtests to demonstrate overfitting and multiple testing issues.
    """
    
    @staticmethod
    def _generate_random_returns(n_periods: int, 
                                mean: float = 0.0, 
                                std: float = 0.01, 
                                skew: float = 0.0, 
                                random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random returns.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        if skew == 0:
            returns = np.random.normal(mean, std, n_periods)
        else:
            try:
                from scipy.stats import skewnorm
                a = skew
                scale = std
                loc = mean - a * scale * np.sqrt(2/np.pi) / np.sqrt(1 + a**2)
                returns = skewnorm.rvs(a=a, loc=loc, scale=scale, size=n_periods)
            except ImportError:
                warnings.warn("scipy.stats skewnorm not available, using normal distribution instead")
                returns = np.random.normal(mean, std, n_periods)
        
        return returns
    
    @staticmethod
    def generate_strategy_returns(n_strategies: int, 
                               n_periods: int, 
                               mean: float = 0.0, 
                               std: float = 0.01,
                               random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate returns for multiple random strategies.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        all_returns = np.zeros((n_strategies, n_periods))
        for i in range(n_strategies):
            seed = random_state + i if random_state is not None else None
            all_returns[i] = BacktestSimulator._generate_random_returns(n_periods, mean, std, 0, seed)
        
        return all_returns
    
    @staticmethod
    def generate_correlated_strategy_returns(n_strategies: int,
                                          n_periods: int,
                                          n_factors: int = 3,
                                          mean: float = 0.0,
                                          std: float = 0.01,
                                          random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate returns for multiple correlated strategies using a factor model.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        factor_returns = np.zeros((n_factors, n_periods))
        for i in range(n_factors):
            factor_returns[i] = BacktestSimulator._generate_random_returns(
                n_periods, mean, std, 0, random_state + i if random_state is not None else None
            )
        
        factor_loadings = np.random.normal(0, 1, (n_strategies, n_factors))
        idiosyncratic_returns = np.random.normal(0, std/2, (n_strategies, n_periods))
        
        strategy_returns = np.zeros((n_strategies, n_periods))
        for i in range(n_strategies):
            strategy_returns[i] = np.sum(factor_loadings[i].reshape(-1, 1) * factor_returns, axis=0) + idiosyncratic_returns[i]
            current_mean = np.mean(strategy_returns[i])
            current_std = np.std(strategy_returns[i])
            strategy_returns[i] = (strategy_returns[i] - current_mean) / current_std * std + mean
        
        return strategy_returns
    
    @staticmethod
    def demonstrate_backtest_overfitting(n_strategies: int = 100,
                                       n_periods: int = 504,
                                       is_fraction: float = 0.5,
                                       random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Demonstrate backtest overfitting by comparing in-sample and out-of-sample performance.
        """
        from ..validation.statistics import PerformanceStatistics
        from ..utils.visualization import VisualizationTools

        all_returns = BacktestSimulator.generate_strategy_returns(
            n_strategies, n_periods, mean=0.0, std=0.01, random_state=random_state
        )
        
        split_idx = int(n_periods * is_fraction)
        is_returns = all_returns[:, :split_idx]
        oos_returns = all_returns[:, split_idx:]
        
        is_sharpes = np.array([PerformanceStatistics.sharpe_ratio(returns) for returns in is_returns])
        oos_sharpes = np.array([PerformanceStatistics.sharpe_ratio(returns) for returns in oos_returns])
        
        pbo = PerformanceStatistics.probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
        
        best_is_idx = np.argmax(is_sharpes)
        best_is_sharpe = is_sharpes[best_is_idx]
        
        dsr = PerformanceStatistics.deflated_sharpe_ratio(best_is_sharpe, n_strategies, n_obs=split_idx)
        
        pbo_fig = VisualizationTools.plot_pbo_analysis(is_sharpes, oos_sharpes)
        degradation_fig = VisualizationTools.plot_performance_degradation(is_sharpes, oos_sharpes)
        
        results = {
            'is_sharpes': is_sharpes,
            'oos_sharpes': oos_sharpes,
            'pbo': pbo,
            'dsr': dsr,
            'pbo_fig': pbo_fig,
            'degradation_fig': degradation_fig
        }
        
        return results 