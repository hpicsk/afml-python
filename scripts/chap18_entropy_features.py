import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from afml.features.entropy import EntropyFeatures

def example_entropy_features():
    """
    Example demonstrating the use of entropy-based features.
    """
    print("--- Entropy-Based Features Demonstration ---")

    # Generate synthetic data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500),
                        index=pd.date_range(start='2020-01-01', periods=500))

    # 1. Plug-in Entropy
    print("\n1. Calculating Plug-in Entropy...")
    # Discretize the returns into 10 bins
    discretized_returns = EntropyFeatures.discretize_series(returns, n_bins=10)
    entropy_full = EntropyFeatures.plug_in_entropy(discretized_returns)
    print(f"  - Plug-in entropy of the full series: {entropy_full:.4f} bits")

    # 2. Lempel-Ziv Complexity
    print("\n2. Calculating Lempel-Ziv Complexity...")
    # Convert returns to a binary sequence (1 if above mean, 0 if below)
    binary_sequence = "".join((returns > returns.mean()).astype(int).astype(str))
    lz_complexity_full = EntropyFeatures.lempel_ziv_complexity(binary_sequence)
    print(f"  - Normalized LZ complexity of the full series: {lz_complexity_full:.4f}")

    # 3. Rolling Entropy Features
    print("\n3. Calculating rolling entropy features...")
    window = 50
    rolling_plug_in = EntropyFeatures.rolling_entropy(returns, window=window, method='plug_in')
    rolling_lz = EntropyFeatures.rolling_entropy(returns, window=window, method='lz')

    # 4. Visualization
    print("\n4. Visualizing results...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot original series
    axes[0].plot(returns.index, returns, label='Returns', color='black', alpha=0.7)
    axes[0].set_title('Original Returns Series')
    axes[0].set_ylabel('Returns')
    axes[0].grid(True, alpha=0.3)

    # Plot rolling plug-in entropy
    axes[1].plot(rolling_plug_in.index, rolling_plug_in, label=f'Rolling Plug-in Entropy (w={window})', color='blue')
    axes[1].set_title('Rolling Plug-in Entropy')
    axes[1].set_ylabel('Entropy (bits)')
    axes[1].grid(True, alpha=0.3)

    # Plot rolling LZ complexity
    axes[2].plot(rolling_lz.index, rolling_lz, label=f'Rolling LZ Complexity (w={window})', color='red')
    axes[2].set_title('Rolling Lempel-Ziv Complexity')
    axes[2].set_ylabel('Normalized Complexity')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example_entropy_features()
