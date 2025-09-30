import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from afml.data.labeling import DailyVolatility, TripleBarrierLabeling, MetaLabeling
from afml.data.sampling import FeatureSampling


def example_with_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    prices = pd.Series(100 + np.random.randn(1000).cumsum(), 
                       index=pd.date_range(start='2020-01-01', periods=1000, freq='D'))
    
    # 1. Calculate daily volatility
    vol = DailyVolatility.get_daily_vol(prices, span=30)
    
    # 2. Define events
    events = pd.DataFrame({
        'trgt': vol,
        'tl': prices.index.intersection(vol.index) + pd.Timedelta(days=7)
    }, index=vol.index)
    
    # 3. Apply Triple Barrier Labeling
    labeling = TripleBarrierLabeling(prices, events, pt_sl=(1.5, 1.5), min_ret=0.005, num_threads=1)
    labeled_events = labeling.get_events()
    
    # 4. Get bins
    bins = TripleBarrierLabeling.get_bins(labeled_events, prices)
    
    # 5. Meta-Labeling example
    # Assume we have a primary model's probabilities
    primary_model_probs = pd.Series(np.random.rand(len(bins)), index=bins.index)
    
    # Sizing the bet
    bet_sizes = primary_model_probs.apply(MetaLabeling.bet_size)
    
    # Plotting precision vs accuracy
    MetaLabeling.plot_precision_vs_accuracy(primary_model_probs.values, bins['bin'].values)
    
    # 6. Feature Sampling
    features = pd.DataFrame(np.random.rand(len(prices), 3), index=prices.index, columns=['f1', 'f2', 'f3'])
    event_features = FeatureSampling.get_features_at_events(features, labeled_events)
    
    print("Labeled Events Head:")
    print(labeled_events.head())
    print("\nBins Head:")
    print(bins.head())
    print("\nEvent Features Head:")
    print(event_features.head())
    
    return labeled_events, bins, event_features

if __name__ == "__main__":
    example_with_synthetic_data()