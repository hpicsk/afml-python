import numpy as np
import pandas as pd
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TickDataProcessor:
    """
    Handles the cleaning, processing, and aggregation of high-frequency tick data.
    (Derived from chap13.TickDataProcessor)
    """

    def clean_trades(self, trades: pd.DataFrame, price_col: str = 'price',
                     volume_col: str = 'volume') -> pd.DataFrame:
        """
        Performs basic cleaning of raw trade data.

        Args:
            trades: DataFrame of trade data.
            price_col: The name of the price column.
            volume_col: The name of the volume column.

        Returns:
            A cleaned DataFrame.
        """
        cleaned = trades.dropna(subset=[price_col, volume_col])
        cleaned = cleaned[cleaned[volume_col] > 0]
        cleaned = cleaned[cleaned[price_col] > 1e-6] # Remove zero-price trades
        logger.info(f"Cleaned trades: {len(trades)} -> {len(cleaned)} records.")
        return cleaned

    def _create_bars(self, trades: pd.DataFrame, price_col: str, volume_col: str,
                     grouper: pd.Grouper) -> pd.DataFrame:
        """Helper function to create bars based on a grouper object."""
        ohlc = trades[price_col].resample(grouper).ohlc()
        volume = trades[volume_col].resample(grouper).sum()
        
        bars = ohlc.join(volume)
        bars = bars.rename(columns={volume_col: 'volume'})
        return bars.dropna()

    def create_time_bars(self, trades: pd.DataFrame, freq: str = '1Min',
                         price_col: str = 'price', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Aggregates tick data into time-based bars (e.g., 1-minute, 5-minute).

        Args:
            trades: DataFrame of trade data.
            freq: The time frequency for the bars (e.g., '1T', '5T', '1H').
            price_col: The name of the price column.
            volume_col: The name of the volume column.

        Returns:
            A DataFrame of time-based OHLCV bars.
        """
        logger.info(f"Creating time bars with frequency: {freq}...")
        return self._create_bars(trades, price_col, volume_col, pd.Grouper(freq=freq))

    def create_tick_bars(self, trades: pd.DataFrame, ticks_per_bar: int = 1000,
                         price_col: str = 'price', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Aggregates trades into bars containing a fixed number of ticks (trades).

        Args:
            trades: DataFrame of trade data.
            ticks_per_bar: The number of trades to include in each bar.
            price_col: The name of the price column.
            volume_col: The name of the volume column.

        Returns:
            A DataFrame of tick-based OHLCV bars.
        """
        logger.info(f"Creating tick bars with {ticks_per_bar} ticks per bar...")
        bar_indices = range(0, len(trades), ticks_per_bar)
        bars_list = []
        for i in tqdm(bar_indices, desc="Creating Tick Bars"):
            chunk = trades.iloc[i : i + ticks_per_bar]
            if chunk.empty: continue

            bars_list.append({
                'timestamp': chunk.index[-1],
                'open': chunk[price_col].iloc[0],
                'high': chunk[price_col].max(),
                'low': chunk[price_col].min(),
                'close': chunk[price_col].iloc[-1],
                'volume': chunk[volume_col].sum(),
            })
            
        return pd.DataFrame(bars_list).set_index('timestamp')

    def create_volume_bars(self, trades: pd.DataFrame, volume_per_bar: float,
                           price_col: str = 'price', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Aggregates trades into bars representing a fixed amount of traded volume.

        Args:
            trades: DataFrame of trade data.
            volume_per_bar: The amount of volume to include in each bar.
            price_col: The name of the price column.
            volume_col: The name of the volume column.

        Returns:
            A DataFrame of volume-based OHLCV bars.
        """
        logger.info(f"Creating volume bars with {volume_per_bar} volume per bar...")
        cumulative_volume = trades[volume_col].cumsum()
        bar_end_indices = cumulative_volume.searchsorted(np.arange(volume_per_bar, cumulative_volume.iloc[-1], volume_per_bar))
        
        bars_list = []
        last_idx = 0
        for end_idx in tqdm(bar_end_indices, desc="Creating Volume Bars"):
            chunk = trades.iloc[last_idx:end_idx]
            if chunk.empty: continue
            
            bars_list.append({
                'timestamp': chunk.index[-1],
                'open': chunk[price_col].iloc[0],
                'high': chunk[price_col].max(),
                'low': chunk[price_col].min(),
                'close': chunk[price_col].iloc[-1],
                'volume': chunk[volume_col].sum(),
            })
            last_idx = end_idx
        
        return pd.DataFrame(bars_list).set_index('timestamp') 