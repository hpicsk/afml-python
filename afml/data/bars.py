import pandas as pd
import numpy as np
from typing import List, Union, Tuple


class BarSampler:
    """
    Class for implementing various bar sampling techniques from Chapter 2 of
    "Advances in Financial Machine Learning" by Marcos López de Prado.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with market data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing market data with at least:
            - timestamp (index or column)
            - price (column)
            - volume (column)
            - optionally: dollar value of trades
        """
        self.data = data.copy()
        if 'timestamp' in self.data.columns and not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.set_index('timestamp', inplace=True)
        
        # Ensure required columns exist
        required_cols = ['price', 'volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain a '{col}' column")
        
        # Calculate dollar value if not provided
        if 'dollar_value' not in self.data.columns:
            self.data['dollar_value'] = self.data['price'] * self.data['volume']
    
    def get_time_bars(self, frequency: str) -> pd.DataFrame:
        """
        Sample data into time bars of specified frequency.
        
        Parameters:
        -----------
        frequency : str
            Time frequency (e.g., '1min', '1h', '1d')
            
        Returns:
        --------
        pandas.DataFrame with OHLCV data
        """
        ohlc = self.data['price'].resample(frequency).ohlc()
        volume = self.data['volume'].resample(frequency).sum()
        dollar_value = self.data['dollar_value'].resample(frequency).sum()
        
        bars = pd.concat([ohlc, volume, dollar_value], axis=1)
        bars.columns = ['open', 'high', 'low', 'close', 'volume', 'dollar_value']
        return bars.dropna()
    
    def get_tick_bars(self, tick_threshold: int) -> pd.DataFrame:
        """
        Sample data into bars containing a specific number of ticks (trades).
        
        Parameters:
        -----------
        tick_threshold : int
            Number of ticks per bar
            
        Returns:
        --------
        pandas.DataFrame with OHLCV data
        """
        bars = []
        tick_count = 0
        open_price = high_price = low_price = close_price = None
        cum_volume = 0
        cum_dollar = 0
        start_time = None
        
        for idx, row in self.data.iterrows():
            if tick_count == 0:
                open_price = row['price']
                high_price = row['price']
                low_price = row['price']
                start_time = idx
            
            tick_count += 1
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += row['volume']
            cum_dollar += row['dollar_value']
            
            # If we've reached the threshold, create a bar and reset
            if tick_count >= tick_threshold:
                bars.append({
                    'timestamp': idx,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': cum_volume,
                    'dollar_value': cum_dollar,
                    'start_time': start_time
                })
                
                # Reset for next bar
                tick_count = 0
                cum_volume = 0
                cum_dollar = 0
        
        # Add any remaining ticks as the final bar if they exist
        if tick_count > 0:
            bars.append({
                'timestamp': self.data.index[-1],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': cum_volume,
                'dollar_value': cum_dollar,
                'start_time': start_time
            })
        
        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df
    
    def get_volume_bars(self, volume_threshold: float) -> pd.DataFrame:
        """
        Sample data into bars containing a specific amount of volume.
        
        Parameters:
        -----------
        volume_threshold : float
            Volume threshold for each bar
            
        Returns:
        --------
        pandas.DataFrame with OHLCV data
        """
        bars = []
        cum_volume = 0
        open_price = high_price = low_price = close_price = None
        cum_dollar = 0
        start_time = None
        
        for idx, row in self.data.iterrows():
            if cum_volume == 0:  # This is the first tick of the bar
                open_price = row['price']
                high_price = row['price']
                low_price = row['price']
                start_time = idx
            
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += row['volume']
            cum_dollar += row['dollar_value']
            
            # If we've reached the threshold, create a bar and reset
            if cum_volume >= volume_threshold:
                bars.append({
                    'timestamp': idx,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': cum_volume,
                    'dollar_value': cum_dollar,
                    'start_time': start_time
                })
                
                # Reset for next bar
                cum_volume = 0
                cum_dollar = 0
        
        # Add any remaining volume as the final bar if it exists
        if cum_volume > 0:
            bars.append({
                'timestamp': self.data.index[-1],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': cum_volume,
                'dollar_value': cum_dollar,
                'start_time': start_time
            })
        
        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df
    
    def get_dollar_bars(self, dollar_threshold: float) -> pd.DataFrame:
        """
        Sample data into bars containing a specific dollar value.
        
        Parameters:
        -----------
        dollar_threshold : float
            Dollar value threshold for each bar
            
        Returns:
        --------
        pandas.DataFrame with OHLCV data
        """
        bars = []
        cum_dollar = 0
        open_price = high_price = low_price = close_price = None
        cum_volume = 0
        start_time = None
        
        for idx, row in self.data.iterrows():
            if cum_dollar == 0:  # This is the first tick of the bar
                open_price = row['price']
                high_price = row['price']
                low_price = row['price']
                start_time = idx
            
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += row['volume']
            cum_dollar += row['dollar_value']
            
            # If we've reached the threshold, create a bar and reset
            if cum_dollar >= dollar_threshold:
                bars.append({
                    'timestamp': idx,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': cum_volume,
                    'dollar_value': cum_dollar,
                    'start_time': start_time
                })
                
                # Reset for next bar
                cum_dollar = 0
                cum_volume = 0
        
        # Add any remaining dollar value as the final bar if it exists
        if cum_dollar > 0:
            bars.append({
                'timestamp': self.data.index[-1],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': cum_volume,
                'dollar_value': cum_dollar,
                'start_time': start_time
            })
        
        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df
    
    def _calculate_imbalance(self):
        """
        Calculate tick imbalance sequence b_t as per the book.
        b_t = sign(delta p_t), with b_t = b_{t-1} if delta p_t = 0.
        """
        price_diff = self.data['price'].diff()
        self.data['b'] = np.sign(price_diff).fillna(0)
        
        # Propagate last sign on zero price change
        self.data['b'] = self.data['b'].replace(0, np.nan).ffill()

    def get_tick_imbalance_bars(self, expected_ticks_per_bar: int, initial_prob_buy: float = 0.5):
        """
        Sample data into Tick Imbalance Bars (TIB).
        Bars are formed when the accumulated tick imbalance exceeds a dynamic threshold.
        
        Parameters:
        - expected_ticks_per_bar (int): Expected number of ticks in a bar, used to set the threshold.
        - initial_prob_buy (float): Initial estimate for P[b_t = 1].
        
        Returns:
        - pd.DataFrame: OHLCV dataframe of TIBs.
        """
        self._calculate_imbalance()
        
        bars = []
        prob_buy = initial_prob_buy
        expected_threshold = expected_ticks_per_bar * abs(2 * prob_buy - 1)
        
        cum_imbalance = 0
        cum_ticks = 0
        open_price = high_price = low_price = close_price = None
        cum_volume = 0
        cum_dollar = 0
        start_time = None
        
        for idx, row in self.data.iterrows():
            if cum_ticks == 0:
                open_price = high_price = low_price = row['price']
                start_time = idx
            
            b_t = row['b']
            cum_imbalance += b_t
            cum_ticks += 1
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += row['volume']
            cum_dollar += row['dollar_value']
            
            if abs(cum_imbalance) >= expected_threshold:
                bars.append({
                    'timestamp': idx, 'open': open_price, 'high': high_price,
                    'low': low_price, 'close': close_price, 'volume': cum_volume,
                    'dollar_value': cum_dollar, 'start_time': start_time
                })
                
                # Update probability and threshold for next bar
                prob_buy = cum_ticks / (cum_ticks + (1 if b_t > 0 else 0)) # simplified update
                expected_threshold = expected_ticks_per_bar * abs(2 * prob_buy - 1)
                
                # Reset for next bar
                cum_imbalance = 0
                cum_ticks = 0
                cum_volume = 0
                cum_dollar = 0
        
        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df

    def get_volume_imbalance_bars(self, expected_volume_per_bar: float, initial_prob_buy: float = 0.5):
        """
        Sample data into Volume Imbalance Bars (VIB).
        Bars are formed when the accumulated volume imbalance exceeds a dynamic threshold.
        """
        self._calculate_imbalance()
        
        bars = []
        prob_buy = initial_prob_buy
        expected_threshold = expected_volume_per_bar * abs(2 * prob_buy - 1)
        
        cum_imbalance = 0
        cum_ticks = 0
        open_price = high_price = low_price = close_price = None
        cum_volume = 0
        cum_dollar = 0
        start_time = None
        
        for idx, row in self.data.iterrows():
            if cum_ticks == 0:
                open_price = high_price = low_price = row['price']
                start_time = idx
            
            b_t = row['b']
            v_t = row['volume']
            cum_imbalance += b_t * v_t
            cum_ticks += 1
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += v_t
            cum_dollar += row['dollar_value']
            
            if abs(cum_imbalance) >= expected_threshold:
                bars.append({
                    'timestamp': idx, 'open': open_price, 'high': high_price,
                    'low': low_price, 'close': close_price, 'volume': cum_volume,
                    'dollar_value': cum_dollar, 'start_time': start_time
                })
                prob_buy = np.sum(self.data.loc[start_time:idx]['b'] > 0) / cum_ticks
                expected_threshold = expected_volume_per_bar * abs(2 * prob_buy - 1)
                cum_imbalance = 0
                cum_ticks = 0
                cum_volume = 0
                cum_dollar = 0

        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df

    def get_dollar_imbalance_bars(self, expected_dollar_per_bar: float, initial_prob_buy: float = 0.5):
        """
        Sample data into Dollar Imbalance Bars (DIB).
        Bars are formed when the accumulated dollar imbalance exceeds a dynamic threshold.
        """
        self._calculate_imbalance()
        
        bars = []
        prob_buy = initial_prob_buy
        expected_threshold = expected_dollar_per_bar * abs(2 * prob_buy - 1)
        
        cum_imbalance = 0
        cum_ticks = 0
        open_price = high_price = low_price = close_price = None
        cum_volume = 0
        cum_dollar = 0
        start_time = None
        
        for idx, row in self.data.iterrows():
            if cum_ticks == 0:
                open_price = high_price = low_price = row['price']
                start_time = idx
            
            b_t = row['b']
            d_t = row['dollar_value']
            cum_imbalance += b_t * d_t
            cum_ticks += 1
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += row['volume']
            cum_dollar += d_t
            
            if abs(cum_imbalance) >= expected_threshold:
                bars.append({
                    'timestamp': idx, 'open': open_price, 'high': high_price,
                    'low': low_price, 'close': close_price, 'volume': cum_volume,
                    'dollar_value': cum_dollar, 'start_time': start_time
                })
                prob_buy = np.sum(self.data.loc[start_time:idx]['b'] > 0) / cum_ticks
                expected_threshold = expected_dollar_per_bar * abs(2 * prob_buy - 1)
                cum_imbalance = 0
                cum_ticks = 0
                cum_volume = 0
                cum_dollar = 0

        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df

    def _get_runs_bars(self, bar_type: str, threshold_multiplier: float):
        """
        Generic function to compute runs bars (Tick, Volume, or Dollar).
        
        - bar_type: 'tick', 'volume', or 'dollar'
        - threshold_multiplier: Factor to multiply the expected run length by to get the threshold.
        """
        self._calculate_imbalance()
        
        prob_buy = (self.data['b'] > 0).mean()
        prob_sell = 1 - prob_buy
        
        expected_buy_run = 1 / prob_sell if prob_sell > 0 else np.inf
        expected_sell_run = 1 / prob_buy if prob_buy > 0 else np.inf

        bars = []
        last_b = 0
        run_value = 0
        threshold = np.inf # Initialize with a high value
        
        cum_ticks = 0
        open_price = high_price = low_price = close_price = None
        cum_volume = 0
        cum_dollar = 0
        start_time = None

        for idx, row in self.data.iterrows():
            if cum_ticks == 0:
                open_price = high_price = low_price = row['price']
                start_time = idx

            b_t = row['b']
            
            if b_t != last_b: # New run
                last_b = b_t
                threshold = (expected_buy_run if b_t > 0 else expected_sell_run) * threshold_multiplier
                run_value = 0

            # Accumulate based on bar type
            if bar_type == 'tick':
                run_value += 1
            elif bar_type == 'volume':
                run_value += row['volume']
            elif bar_type == 'dollar':
                run_value += row['dollar_value']

            cum_ticks += 1
            high_price = max(high_price, row['price'])
            low_price = min(low_price, row['price'])
            close_price = row['price']
            cum_volume += row['volume']
            cum_dollar += row['dollar_value']

            if run_value >= threshold:
                bars.append({
                    'timestamp': idx, 'open': open_price, 'high': high_price,
                    'low': low_price, 'close': close_price, 'volume': cum_volume,
                    'dollar_value': cum_dollar, 'start_time': start_time
                })
                # Reset for next bar
                cum_ticks = 0
                cum_volume = 0
                cum_dollar = 0
                run_value = 0
        
        bars_df = pd.DataFrame(bars)
        if not bars_df.empty:
            bars_df.set_index('timestamp', inplace=True)
        return bars_df

    def get_tick_runs_bars(self, tick_run_threshold_multiplier: float = 10.0):
        return self._get_runs_bars('tick', tick_run_threshold_multiplier)

    def get_volume_runs_bars(self, volume_run_threshold_multiplier: float = 10.0):
        return self._get_runs_bars('volume', volume_run_threshold_multiplier)

    def get_dollar_runs_bars(self, dollar_run_threshold_multiplier: float = 10.0):
        return self._get_runs_bars('dollar', dollar_run_threshold_multiplier) 