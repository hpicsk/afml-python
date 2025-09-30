import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from afml.validation.structural_breaks import StructuralBreaks
from afml.utils.visualization import VisualizationTools

def example_structural_breaks():
    """
    Example demonstrating the use of structural break detection methods.
    """
    print("--- Structural Break Detection Demonstration ---")

    # Generate synthetic data with a structural break
    np.random.seed(42)
    returns1 = np.random.normal(0.001, 0.01, 250)
    returns2 = np.random.normal(0.003, 0.02, 250) # Break in mean and vol
    returns = pd.Series(np.concatenate([returns1, returns2]), 
                        index=pd.date_range(start='2020-01-01', periods=500))

    # 1. CUSUM Test for Mean
    print("\n1. Running CUSUM test for mean shift...")
    cusum_mean_stats, thresholds_mean = StructuralBreaks.get_cusum_stat(returns, threshold=5)
    
    # Find break points
    break_points_mean = cusum_mean_stats.index[
        (cusum_mean_stats['S+'] > thresholds_mean) | (cusum_mean_stats['S-'] < -thresholds_mean)
    ]
    if not break_points_mean.empty:
        print(f"  - Potential break in mean detected at: {break_points_mean[0]}")
    else:
        print("  - No break in mean detected.")

    # 2. CUSUM Test for Volatility
    print("\n2. Running CUSUM test for volatility shift...")
    cusum_vol_stats, thresholds_vol = StructuralBreaks.get_cusum_vol_stat(returns, threshold=10)
    
    # Find break points
    break_points_vol = cusum_vol_stats.index[
        (cusum_vol_stats['S+'] > thresholds_vol) | (cusum_vol_stats['S-'] < -thresholds_vol)
    ]
    if not break_points_vol.empty:
        print(f"  - Potential break in volatility detected at: {break_points_vol[0]}")
    else:
        print("  - No break in volatility detected.")

    # 3. Visualization
    print("\n3. Visualizing CUSUM results...")
    fig1 = VisualizationTools.plot_cusum(cusum_mean_stats, thresholds_mean, title='CUSUM Test for Mean Shift')
    fig2 = VisualizationTools.plot_cusum(cusum_vol_stats, thresholds_vol, title='CUSUM Test for Volatility Shift')
    
    plt.show()

if __name__ == "__main__":
    example_structural_breaks()
