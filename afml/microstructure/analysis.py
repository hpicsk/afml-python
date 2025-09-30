import numpy as np
import pandas as pd
import statsmodels.api as sm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketMicrostructureAnalyzer:
    """
    Calculates key market microstructure features from high-frequency data.
    These features can provide insight into liquidity, price impact, and order flow.
    (Derived from chap13.MarketMicrostructureAnalyzer)
    """

    def classify_trades_tick_rule(self, trades: pd.DataFrame, price_col: str = 'price') -> pd.Series:
        """
        Classifies trades as buyer or seller initiated using the simple tick rule.
        - A trade at a higher price than the previous is a 'buy' (1).
        - A trade at a lower price is a 'sell' (-1).
        - A trade at the same price is classified the same as the previous trade.

        Args:
            trades: DataFrame of trade data, must be sorted by time.
            price_col: The name of the price column.

        Returns:
            A Series indicating the direction of each trade (1 for buy, -1 for sell).
        """
        price_diff = trades[price_col].diff()
        direction = pd.Series(np.nan, index=trades.index, name='trade_direction')
        
        direction[price_diff > 0] = 1
        direction[price_diff < 0] = -1
        
        # Forward-fill to handle zero-tick trades
        direction = direction.ffill().fillna(1) # Start with a buy
        logger.info("Classified trades using the tick rule.")
        return direction

    def calculate_kyle_lambda(self, trades: pd.DataFrame, window: str = '5T',
                              price_col: str = 'price', volume_col: str = 'volume') -> tuple:
        """
        Calculates Kyle's Lambda, a measure of price impact or illiquidity.
        It is estimated by regressing price changes on net order flow (signed volume).
        Lambda = |Change in Price| / Order Flow

        Args:
            trades: DataFrame of trade data.
            window: The time frequency to use for aggregation (e.g., '1T', '5T').
            price_col: The name of the price column.
            volume_col: The name of the volume column.

        Returns:
            A tuple containing the estimated lambda value and the regression summary.
        """
        logger.info(f"Calculating Kyle's Lambda with a '{window}' window...")
        data = trades[[price_col, volume_col]].copy()
        data['direction'] = self.classify_trades_tick_rule(data, price_col)
        data['signed_volume'] = data[volume_col] * data['direction']
        
        # Resample to the desired window (replace 'T' with 'min' if needed)
        window_fixed = window.replace('T', 'min') if 'T' in window else window
        price_change = data[price_col].resample(window_fixed).last().diff()
        order_flow = data['signed_volume'].resample(window_fixed).sum()
        
        # Align series and prepare for regression
        model_data = pd.DataFrame({'price_change': price_change, 'order_flow': order_flow}).dropna()
        
        X = sm.add_constant(model_data['order_flow'])
        y = model_data['price_change']
        
        if len(X) < 2:
            logger.warning("Not enough data points to calculate Kyle's Lambda.")
            return np.nan, None

        model = sm.OLS(y, X).fit()
        kyle_lambda_value = model.params.get('order_flow', np.nan)
        
        logger.info(f"Estimated Kyle's Lambda: {kyle_lambda_value:.6f}")
        return kyle_lambda_value, model.summary()

    def calculate_roll_measure(self, trades: pd.DataFrame, price_col: str = 'price') -> float:
        """
        Calculates Roll's spread estimator, an early measure of the effective bid-ask spread
        derived from the serial covariance of price changes.

        Spread = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))

        Args:
            trades: DataFrame of trade data.
            price_col: The name of the price column.

        Returns:
            The estimated effective spread.
        """
        logger.info("Calculating Roll's spread measure...")
        price_changes = trades[price_col].diff().dropna()
        
        # The measure is only valid if the autocovariance is negative
        autocov = price_changes.autocorr(lag=1) * price_changes.var()
        if autocov >= 0:
            logger.warning("Positive autocovariance in price changes; Roll's measure is not applicable.")
            return np.nan
            
        spread = 2 * np.sqrt(-autocov)
        logger.info(f"Estimated Roll Spread: {spread:.4f}")
        return spread 