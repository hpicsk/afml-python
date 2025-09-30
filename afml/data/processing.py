import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def etf_trick(dataframes: dict, volumes: dict) -> pd.DataFrame:
    """
    Construct a continuous price series from multiple ETFs using the ETF trick.

    This method combines multiple time series into a single one by weighting
    each series by its relative liquidity (dollar volume) at each point in time.

    Parameters:
    -----------
    dataframes : dict
        A dictionary where keys are asset names and values are pandas DataFrames
        containing at least a 'close' price column. The index must be a
        DatetimeIndex.
        
    volumes : dict
        A dictionary where keys are asset names and values are pandas Series
        of dollar volumes, with a DatetimeIndex.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with a single 'close' column representing the continuous
        price series, and a 'volume' column with the aggregated volume.
    """
    
    # Align all dataframes and volumes to a common index
    all_dates = pd.Index([])
    for df in dataframes.values():
        all_dates = all_dates.union(df.index)
    for vol in volumes.values():
        all_dates = all_dates.union(vol.index)
        
    # Combine all close prices and volumes into single dataframes
    close_prices = pd.DataFrame(index=all_dates)
    dollar_volumes = pd.DataFrame(index=all_dates)
    
    for name, df in dataframes.items():
        close_prices[name] = df['close']
        
    for name, vol in volumes.items():
        dollar_volumes[name] = vol
        
    # Fill missing values
    close_prices = close_prices.ffill()
    dollar_volumes = dollar_volumes.fillna(0)
    
    # Calculate daily returns
    returns = close_prices.pct_change()
    
    # Calculate weights based on relative dollar volume
    total_volume = dollar_volumes.sum(axis=1)
    weights = dollar_volumes.div(total_volume, axis=0).fillna(0)
    
    # Calculate weighted average of returns
    weighted_returns = (returns * weights).sum(axis=1)
    
    # Reconstruct the price series from returns
    # Start with a base price of 100
    continuous_price = 100 * (1 + weighted_returns).cumprod()
    
    # Create the final dataframe
    result_df = pd.DataFrame({
        'close': continuous_price,
        'volume': total_volume
    })
    
    return result_df.dropna()

def futures_roll(futures_data: pd.DataFrame, active_contract_col: str):
    """
    Create a continuous futures price series by rolling contracts.

    This function uses the 'panama canal' method, where historical prices are
    adjusted to match the price of the new contract at the roll date.

    Parameters:
    -----------
    futures_data : pd.DataFrame
        A DataFrame containing price columns for multiple futures contracts.
        The index must be a DatetimeIndex.
        
    active_contract_col : str
        The name of the column that specifies which contract is the active one
        for each timestamp. This column should contain the column name of the
        active contract's price.

    Returns:
    --------
    pd.Series
        A pandas Series representing the continuous rolled price series.
    """
    
    rolled_prices = pd.Series(index=futures_data.index, dtype=float)
    
    # Get the price of the active contract for each day
    active_contract_series = futures_data[active_contract_col]
    
    # Using stack and merge to avoid deprecated lookup
    price_stacked = futures_data.drop(columns=[active_contract_col]).stack().reset_index()
    price_stacked.columns = ['date', 'contract', 'price']
    
    active_df = active_contract_series.reset_index()
    active_df.columns = ['date', 'contract']
    
    merged = pd.merge(active_df, price_stacked, on=['date', 'contract'])
    rolled_prices = merged.set_index('date')['price']

    # Calculate adjustments
    roll_dates = futures_data[active_contract_col].drop_duplicates(keep='first').index[1:]
    
    for roll_date in roll_dates:
        prev_day = futures_data.index[futures_data.index.get_loc(roll_date) - 1]
        
        old_contract_col = futures_data.loc[prev_day, active_contract_col]
        new_contract_col = futures_data.loc[roll_date, active_contract_col]
        
        price_old = futures_data.loc[roll_date, old_contract_col]
        price_new = futures_data.loc[roll_date, new_contract_col]
        
        adjustment = price_new - price_old
        
        # Adjust all prices before the roll date
        rolled_prices.loc[:prev_day] += adjustment
        
    return rolled_prices

def pca_hedge_weights(prices_df: pd.DataFrame):
    """
    Calculate hedge weights for a portfolio of assets using PCA.

    This method finds the weights of the linear combination of assets that
    results in the most stationary (minimum variance) portfolio.

    Parameters:
    -----------
    prices_df : pd.DataFrame
        A DataFrame of asset prices. Index is DatetimeIndex, columns are
        asset names.

    Returns:
    --------
    pd.Series
        A pandas Series containing the PCA-derived hedge weights for each asset.
    """
    from sklearn.decomposition import PCA
    
    # Normalize prices to start at 1 to avoid scaling issues
    normalized_prices = prices_df / prices_df.iloc[0]
    
    pca = PCA(n_components=normalized_prices.shape[1])
    pca.fit(normalized_prices)
    
    # The eigenvector with the smallest eigenvalue corresponds to the
    # most stationary portfolio.
    most_stationary_weights = pca.components_[-1]
    
    return pd.Series(most_stationary_weights, index=prices_df.columns) 