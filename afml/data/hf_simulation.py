import numpy as np
import pandas as pd
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HighFrequencyDataSimulator:
    """
    Generates synthetic high-frequency (tick-by-tick) trade data for testing
    market microstructure models.
    (Derived from chap13.HighFrequencyDataSimulator)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the simulator.

        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed:
            np.random.seed(seed)

    def generate_trades(self,
                        n_trades: int = 20000,
                        initial_price: float = 100.0,
                        tick_size: float = 0.01,
                        volatility: float = 0.0001,
                        mean_trade_size: int = 100) -> pd.DataFrame:
        """
        Generates a DataFrame of synthetic trades.

        Args:
            n_trades: The number of trades to generate.
            initial_price: The starting price of the asset.
            tick_size: The minimum price movement.
            volatility: Volatility of the price process per tick.
            mean_trade_size: The mean size of a single trade (in shares).

        Returns:
            A DataFrame containing synthetic trade data with columns 'price' and 'volume'.
        """
        timestamps = pd.to_datetime(pd.date_range(start='2023-01-01 09:30:00', periods=n_trades, freq='50ms'))
        
        # Price path as a random walk
        price_moves = np.random.normal(0, volatility, n_trades)
        prices = initial_price + np.cumsum(price_moves)
        prices = np.round(prices / tick_size) * tick_size # Adhere to tick size

        # Trade sizes using a Poisson distribution
        volumes = np.random.poisson(mean_trade_size, n_trades) + 1 # Ensure volume is at least 1

        trades = pd.DataFrame({'price': prices, 'volume': volumes}, index=timestamps)
        logger.info(f"Generated {n_trades} synthetic high-frequency trades.")
        return trades 