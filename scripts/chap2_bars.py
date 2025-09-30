import pandas as pd
import numpy as np

# Import from the refactored library
from afml.data.bars import BarSampler
from afml.data.processing import etf_trick, futures_roll, pca_hedge_weights


# Example usage:
def generate_sample_data(n_samples=10000):
    """Generate sample tick data for demonstration purposes."""
    np.random.seed(42)
    
    # Create timestamps (1-second intervals for simplicity)
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1s')
    
    # Generate price with random walk
    price_changes = np.random.normal(0, 0.01, n_samples)
    price = 100 + np.cumsum(price_changes)
    
    # Generate random volumes
    volume = np.random.exponential(scale=100, size=n_samples).astype(int) + 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': price,
        'volume': volume
    }, index=timestamps)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    tick_data = generate_sample_data()
    print("Original Tick Data:")
    print(tick_data.head())
    
    # Create bar sampler
    sampler = BarSampler(tick_data)
    
    # Sample different bar types
    print("\nSampling time bars...")
    time_bars = sampler.get_time_bars('5min')
    
    print("Sampling tick bars...")
    tick_bars = sampler.get_tick_bars(tick_threshold=100)
    
    print("Sampling volume bars...")
    volume_bars = sampler.get_volume_bars(volume_threshold=5000)
    
    print("Sampling dollar bars...")
    dollar_bars = sampler.get_dollar_bars(dollar_threshold=500000)
    
    # --- Information-Driven Bars ---
    print("\nSampling tick imbalance bars...")
    tib = sampler.get_tick_imbalance_bars(expected_ticks_per_bar=100)
    
    print("Sampling volume imbalance bars...")
    vib = sampler.get_volume_imbalance_bars(expected_volume_per_bar=50000)
    
    print("Sampling dollar imbalance bars...")
    dib = sampler.get_dollar_imbalance_bars(expected_dollar_per_bar=1000000)
    
    print("Sampling tick runs bars...")
    trb = sampler.get_tick_runs_bars()

    print("Sampling volume runs bars...")
    vrb = sampler.get_volume_runs_bars()

    print("Sampling dollar runs bars...")
    drb = sampler.get_dollar_runs_bars()

    # Print sample of each
    print("\n--- Sampled Bars ---")
    print("Time Bars (5min):")
    print(time_bars.head())
    
    print("\nTick Bars (100 ticks):")
    print(tick_bars.head())
    
    print("\nVolume Bars (5000 volume):")
    print(volume_bars.head())
    
    print("\nDollar Bars ($500,000):")
    print(dollar_bars.head())

    print("\nTick Imbalance Bars (E[T]=100):")
    print(tib.head())

    print("\nVolume Imbalance Bars (E[V]=50k):")
    print(vib.head())

    print("\nDollar Imbalance Bars (E[D]=1M):")
    print(dib.head())
    
    print("\nTick Runs Bars:")
    print(trb.head())
    
    print("\nVolume Runs Bars:")
    print(vrb.head())

    print("\nDollar Runs Bars:")
    print(drb.head())

    # --- Multi-product series examples ---
    print("\n--- Multi-Product Series Examples ---")

    # 1. ETF Trick Example
    print("\n1. ETF Trick")
    etf1_data = generate_sample_data(2000)
    etf2_data = generate_sample_data(3000)
    etf2_data.index = etf1_data.index[-1] + pd.to_timedelta(np.arange(3000) + 1, 's')
    
    etf_data_dict = {'etf1': etf1_data[['price']].rename(columns={'price': 'close'}),
                     'etf2': etf2_data[['price']].rename(columns={'price': 'close'})}
    etf_volume_dict = {'etf1': etf1_data['volume'], 'etf2': etf2_data['volume']}
    
    continuous_etf = etf_trick(etf_data_dict, etf_volume_dict)
    print("Continuous ETF series (from two ETFs):")
    print(continuous_etf.head())
    print(continuous_etf.tail())

    # 2. Futures Roll Example
    print("\n2. Futures Roll")
    fut_data = pd.DataFrame({
        'F1': np.arange(100, 120, 0.5),
        'F2': np.arange(105, 125, 0.5) + 2,
    }, index=pd.date_range(start='2023-01-01', periods=40, freq='D'))
    fut_data['active_contract'] = 'F1'
    fut_data.loc['2023-01-21':, 'active_contract'] = 'F2'
    
    rolled_series = futures_roll(fut_data, 'active_contract')
    print("Continuous futures series:")
    print(rolled_series.head())
    print(rolled_series.tail())
    
    # 3. PCA Hedge Weights Example
    print("\n3. PCA Hedge Weights")
    asset1 = pd.Series(100 + np.random.randn(100).cumsum(), name='AssetA')
    asset2 = pd.Series(100 + asset1 * 0.5 + np.random.randn(100).cumsum(), name='AssetB')
    prices_df = pd.concat([asset1, asset2], axis=1).dropna()
    prices_df.index = pd.date_range(start='2023-01-01', periods=len(prices_df))
    
    hedge_w = pca_hedge_weights(prices_df)
    print("PCA Hedge Weights:")
    print(hedge_w)

    hedged_portfolio = prices_df.dot(hedge_w)
    print("\nHedged (stationary) portfolio value:")
    print(hedged_portfolio.head())