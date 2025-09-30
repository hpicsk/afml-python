import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Dict, Optional

class SequentialBootstrap:
    """
    Implementation of Sequential Bootstrap method for time series.
    """
    
    @staticmethod
    def get_ind_matrix(timestamps: pd.DatetimeIndex, 
                      embargo_time: pd.Timedelta) -> pd.DataFrame:
        """
        Compute indicator matrix for sequential bootstrapping.
        
        Parameters:
        -----------
        timestamps : pd.DatetimeIndex
            DatetimeIndex of observations
        embargo_time : pd.Timedelta
            Time to embargo after each observation
            
        Returns:
        --------
        pd.DataFrame
            Indicator matrix (1 where observations can't be drawn together)
        """
        # Initialize indicator matrix
        ind_matrix = pd.DataFrame(0, index=timestamps, columns=timestamps)
        
        # Fill matrix
        for i, t_i in enumerate(timestamps):
            for j, t_j in enumerate(timestamps):
                # Can't draw together if too close in time
                if abs(t_j - t_i) <= embargo_time:
                    ind_matrix.loc[t_i, t_j] = 1
                    
        return ind_matrix
    
    @staticmethod
    def seq_bootstrap(ind_matrix: pd.DataFrame, 
                     sample_length: Optional[int] = None) -> list:
        """
        Generate indices for sequential bootstrap.
        
        Parameters:
        -----------
        ind_matrix : pd.DataFrame
            Indicator matrix from get_ind_matrix
        sample_length : int, optional
            Length of the bootstrapped sample
            
        Returns:
        --------
        list
            List of indices for bootstrapped sample
        """
        if sample_length is None:
            sample_length = ind_matrix.shape[0]
            
        # Initialize
        draw_idx = []
        available_idx = ind_matrix.index.tolist()
        
        while len(draw_idx) < sample_length and len(available_idx) > 0:
            # Draw random index from available indices
            i = np.random.choice(len(available_idx))
            draw_idx.append(available_idx[i])
            
            # Find indices that can't be drawn with current selection
            excluded = ind_matrix.loc[available_idx[i]] == 1
            available_idx = [idx for j, idx in enumerate(available_idx) 
                            if j != i and not ind_matrix.loc[idx, available_idx[i]]]
            
        return draw_idx


class FeatureSampling:
    """
    Sample features for model training based on triple-barrier events.
    """
    
    @staticmethod
    def get_features_at_events(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        Get features at event timestamps.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame containing features
        events : pd.DataFrame
            DataFrame with event information
            
        Returns:
        --------
        pd.DataFrame with features at event timestamps
        """
        if not isinstance(events.index, pd.DatetimeIndex):
            raise ValueError("Events DataFrame must be indexed by timestamps")
            
        # Get timestamps from events index
        timestamps = events.index
        
        # Get features at those timestamps that exist in the features DataFrame
        valid_timestamps = timestamps[timestamps.isin(features.index)]
        features_at_events = features.loc[valid_timestamps]
        
        return features_at_events 