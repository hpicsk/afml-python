import numpy as np
import pandas as pd
from collections import Counter

class EntropyFeatures:
    """
    Methods for computing entropy-based features, as described in Chapter 18
    of "Advances in Financial Machine Learning".
    """

    @staticmethod
    def discretize_series(series: pd.Series, n_bins: int) -> pd.Series:
        """
        Discretizes a continuous series into a series of integer labels.

        Args:
            series: The continuous time series (e.g., returns).
            n_bins: The number of bins to discretize into.

        Returns:
            A series of integer labels representing the bins.
        """
        return pd.cut(series, bins=n_bins, labels=False, include_lowest=True)

    @staticmethod
    def plug_in_entropy(series: pd.Series) -> float:
        """
        Computes the plug-in (or Maximum Likelihood) estimator of Shannon entropy.
        This works on a series of discrete observations (e.g., binned returns).

        Args:
            series: A time series of discrete values.

        Returns:
            The estimated entropy of the series.
        """
        counts = Counter(series)
        n_samples = len(series)
        
        if n_samples == 0:
            return 0.0
            
        probs = np.array([count / n_samples for count in counts.values()])
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy

    @staticmethod
    def lempel_ziv_complexity(binary_sequence: str) -> float:
        """
        Computes the Lempel-Ziv (LZ) complexity for a binary string.
        The complexity is the number of unique patterns encountered.
        The result is normalized by the length of the sequence.

        Args:
            binary_sequence: A string of '0's and '1's.

        Returns:
            The normalized LZ complexity.
        """
        patterns = set()
        i = 0
        n = len(binary_sequence)
        if n == 0:
            return 0.0
            
        while i < n:
            j = i
            while j < n:
                pattern = binary_sequence[i:j+1]
                if pattern not in patterns:
                    patterns.add(pattern)
                    i = j + 1
                    break
                j += 1
            if j == n: # Reached the end without finding a new pattern
                i = j

        return len(patterns) / n

    @staticmethod
    def rolling_entropy(series: pd.Series, window: int, method: str = 'plug_in', n_bins: int = 10) -> pd.Series:
        """
        Computes entropy over a rolling window.

        Args:
            series: The continuous time series (e.g., returns).
            window: The size of the rolling window.
            method: The entropy calculation method ('plug_in' or 'lz').
            n_bins: The number of bins for discretization (for plug_in method).

        Returns:
            A series of rolling entropy values.
        """
        if method == 'plug_in':
            discretized = EntropyFeatures.discretize_series(series, n_bins)
            return discretized.rolling(window=window).apply(EntropyFeatures.plug_in_entropy, raw=False)
        elif method == 'lz':
            # Convert to binary sequence (e.g., sign of returns)
            binary_series = (series > series.mean()).astype(int).astype(str)
            rolling_lz = pd.Series(index=series.index, dtype=float)
            for i in range(window - 1, len(series)):
                window_str = "".join(binary_series.iloc[i-window+1:i+1])
                rolling_lz.iloc[i] = EntropyFeatures.lempel_ziv_complexity(window_str)
            return rolling_lz
        else:
            raise ValueError("Method must be 'plug_in' or 'lz'")
