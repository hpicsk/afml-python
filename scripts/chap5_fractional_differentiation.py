import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from afml.features.fractional_differentiation import (
    FractionalDifferentiation,
    StationarityTests
)

def example_with_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    prices = pd.Series(100 + np.random.randn(1000).cumsum(), 
                       index=pd.date_range(start='2020-01-01', periods=1000, freq='D'))
    
    # 1. Fractional Differentiation
    ffd = FractionalDifferentiation.frac_diff_ffd(prices, d=0.5)
    
    # 2. Stationarity Tests
    optimal_d, results_df = StationarityTests.find_optimal_d(prices)
    StationarityTests.plot_optimization_results(results_df, optimal_d)
    
    print(f"Optimal d: {optimal_d}")
    
    return ffd, optimal_d, results_df

if __name__ == "__main__":
    example_with_synthetic_data()