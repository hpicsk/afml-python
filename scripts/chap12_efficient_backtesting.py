import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Import from the refactored library
from afml.data.simulation import SyntheticData
from afml.core.streaming import StreamingBacktester, KalmanFilterBacktester
from afml.validation.efficiency import EfficiencyAnalyzer
from afml.utils.visualization import VisualizationTools
from afml.core.backtester import VectorizedBacktester
# Assuming a strategies module exists as in the other examples
from afml.portfolio import strategies

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_features(prices: pd.DataFrame, n_features: int = 10, n_informative: int = 3, feature_noise: float = 0.5) -> pd.DataFrame:
    """
    Generate features from price data. A simplified version for the example.
    """
    returns = prices.pct_change().dropna()
    ma_5 = prices.rolling(5).mean().pct_change().dropna()
    ma_20 = prices.rolling(20).mean().pct_change().dropna()
    vol_20 = returns.rolling(20).std().dropna()
    
    common_idx = returns.index.intersection(ma_5.index).intersection(ma_20.index).intersection(vol_20.index)
    features = pd.DataFrame(index=common_idx)
    
    for i in range(n_informative):
        asset_idx = i % len(returns.columns)
        asset_col = returns.columns[asset_idx]
        features[f'informative_feature_{i+1}'] = (ma_5[asset_col] - ma_20[asset_col]) + np.random.normal(0, feature_noise * ma_5[asset_col].std(), len(ma_5))
    
    for i in range(n_informative, n_features):
        features[f'noise_feature_{i+1}'] = np.random.normal(0, 1, len(common_idx))
        
    return features


def main():
    """
    An example script demonstrating the use of the efficient backtesting tools from the refactored library.
    """
    logger.info("--- Efficient Backtesting Demonstration ---")

    # 1. Data Generation
    logger.info("\n[1] Generating synthetic data...")
    data_generator = SyntheticData(seed=42)
    prices = data_generator.generate_prices(n_assets=10, n_days=500)
    features = generate_features(prices)
    returns = prices.pct_change().dropna()
    
    # Align data
    final_idx = returns.index.intersection(features.index)
    X = features.loc[final_idx]
    y = returns.loc[final_idx].mean(axis=1) # Example target: average daily return

    # 2. Efficiency Analysis
    logger.info("\n[2] Analyzing approximation error...")
    efficiency_analyzer = EfficiencyAnalyzer()
    error_results = efficiency_analyzer.analyze_approximation_error(
        model_factory=lambda: Ridge(alpha=1.0),
        X=X, y=y,
        subsample_ratios=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_repetitions=3,
        cv_splits=3
    )
    print("Approximation Analysis Results:")
    print(error_results['results_df'])

    # 3. Vectorized Backtest
    logger.info("\n[3] Running a vectorized portfolio backtest...")
    # This uses a simple momentum strategy for demonstration
    vectorized_bt = VectorizedBacktester(prices=prices)
    results_df = vectorized_bt.run(strategy=strategies.momentum_strategy, rebalance_freq='ME')
    
    print("\nVectorized Backtest Performance Metrics:")
    for metric, value in vectorized_bt.metrics.items():
        print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")

    # 4. Kalman Filter Backtest (on a single feature for simplicity)
    logger.info("\n[4] Running a Kalman Filter backtest...")
    kf_backtester = KalmanFilterBacktester()
    kf_X = pd.DataFrame(X.iloc[:, 0]) # Use one feature
    kf_results = kf_backtester.run(kf_X, y)
    print(f"Kalman Filter RMSE: {kf_results['rmse']:.4f}")

    # 5. Streaming Backtest (conceptual example)
    logger.info("\n[5] Running a streaming backtest...")
    # We simulate a data loader from our dataframe for demonstration
    data_with_target = pd.concat([X, y.rename('target')], axis=1)
    def simulated_data_loader(start, end):
        return data_with_target.iloc[start:end]

    streaming_bt = StreamingBacktester(
        data_loader=simulated_data_loader,
        total_rows=len(data_with_target),
        feature_cols=list(X.columns),
        target_col='target',
        chunk_size=100
    )
    stream_results = streaming_bt.run(
        model_factory=lambda: Ridge(),
        initial_train_size=150,
        retrain_freq=50
    )
    print(f"Streaming Backtest Final Score (RMSE): {stream_results['rmse']:.4f}")
    
    # 6. Visualization
    logger.info("\n[6] Generating visualization...")
    VisualizationTools.plot_portfolio_performance(results_df)
    plt.show()


if __name__ == "__main__":
    main()