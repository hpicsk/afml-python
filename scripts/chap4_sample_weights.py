import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Dict, Optional
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
from sklearn.cluster import KMeans

from afml.data.weights import SampleWeights, UniquenessSampling
from afml.data.sampling import SequentialBootstrap


# Example usage
def example_with_synthetic_data():
    """
    Example using synthetic data to demonstrate sample weighting techniques.
    """
    np.random.seed(42)
    
    # Generate synthetic times for triple-barrier events
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=500, freq='4h')
    
    # Create event dataframe
    events = pd.DataFrame(index=dates[:250])  # Use first 250 timestamps as start times
    
    # Assign random end times (between 1 and 10 days later)
    random_days = np.random.randint(1, 11, size=len(events))
    events['label_end_time'] = events.index + pd.to_timedelta(random_days, unit='D')
    
    # Create feature matrix (random for this example)
    features = pd.DataFrame(
        np.random.normal(0, 1, size=(len(events), 5)),
        index=events.index,
        columns=[f'feature_{i}' for i in range(5)]
    )
    
    # 1. Time decay weights
    time_decay_weights = SampleWeights.get_time_decay(events.index, decay_factor=0.5)
    
    # 2. Concurrency weights
    concurrency_weights = SampleWeights.get_concurrency_weights(
        events['label_end_time'],
        events.index # label_starttime_index
    )
    
    # 3. Overlap matrix and information-driven weights
    label_endtimes = events['label_end_time']
    overlap_matrix = SampleWeights.compute_overlap_matrix(label_endtimes)
    info_weights = SampleWeights.compute_information_driven_weights(overlap_matrix)
    
    # 4. Uniqueness weights
    uniqueness_weights = UniquenessSampling.get_distance_based_uniqueness(features)
    
    # 5. Sequential bootstrap
    embargo_time = pd.Timedelta(days=5)
    ind_matrix = SequentialBootstrap.get_ind_matrix(events.index, embargo_time)
    bootstrap_indices = SequentialBootstrap.seq_bootstrap(ind_matrix, sample_length=100)
    
    # Plot the different weighting schemes
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(events.index, time_decay_weights)
    plt.title('Time Decay Weights')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 2, 2)
    plt.plot(concurrency_weights.index, concurrency_weights.values)
    plt.title('Concurrency Weights')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 2, 3)
    plt.plot(info_weights.index, info_weights.values)
    plt.title('Information-Driven Weights')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 2, 4)
    plt.plot(uniqueness_weights.index, uniqueness_weights.values)
    plt.title('Uniqueness Weights')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 2, 5)
    plt.hist(bootstrap_indices, bins=20)
    plt.title('Sequential Bootstrap Sample Distribution')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('sample_weights.png')
    
    # Visualize overlap matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(overlap_matrix.iloc[:50, :50], cmap='viridis')
    plt.colorbar()
    plt.title('Label Overlap Matrix (First 50 Events)')
    plt.show()
    # plt.savefig('overlap_matrix.png')
    
    # Return results for further inspection
    results = {
        'time_decay_weights': pd.Series(time_decay_weights, index=events.index),
        'concurrency_weights': concurrency_weights,
        'info_weights': info_weights,
        'uniqueness_weights': uniqueness_weights,
        'bootstrap_indices': bootstrap_indices,
        'overlap_matrix': overlap_matrix
    }
    
    # Print summary statistics
    print("Summary Statistics for Different Weighting Schemes:")
    for name, weights in results.items():
        if name != 'overlap_matrix' and name != 'bootstrap_indices':
            print(f"\n{name}:")
            print(f"  Mean: {weights.mean():.6f}")
            print(f"  Std: {weights.std():.6f}")
            print(f"  Min: {weights.min():.6f}")
            print(f"  Max: {weights.max():.6f}")
    
    return results


if __name__ == "__main__":
    results = example_with_synthetic_data()