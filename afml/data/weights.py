import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Dict, Optional
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
from sklearn.cluster import KMeans


class SampleWeights:
    """
    Implements various sample weighting schemes from Chapter 4 of 
    "Advances in Financial Machine Learning" by Marcos López de Prado.
    """
    
    @staticmethod
    def get_time_decay(timestamps: Union[List, pd.Series, pd.DatetimeIndex], 
                       decay_factor: float = 0.5, 
                       normalize: bool = True) -> np.ndarray:
        """
        Apply time decay to weights based on recency.
        
        Parameters:
        -----------
        timestamps : Union[List, pd.Series, pd.DatetimeIndex]
            Timestamps for which to compute weights
        decay_factor : float, optional
            Factor determining the rate of time decay (default=0.5)
        normalize : bool, optional
            Whether to normalize the weights to sum to 1 (default=True)
            
        Returns:
        --------
        pd.Series
            Series of time decay weights
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            if isinstance(timestamps, pd.Series):
                timestamps = pd.DatetimeIndex(timestamps)
            else:
                timestamps = pd.DatetimeIndex(pd.Series(timestamps))
        
        # Convert to numeric (days since earliest timestamp)
        delta_time = (timestamps - timestamps.min()).total_seconds() / (24 * 60 * 60)
        
        # Newest observation gets weight=1, oldest gets weight=decay_factor
        weights = decay_factor ** (delta_time.max() - delta_time)
        
        if normalize:
            weights = weights / np.sum(weights)
            
        return pd.Series(weights, index=timestamps)

    @staticmethod
    def get_concurrency_weights(label_endtimes: pd.Series,
                            label_starttime_index: pd.DatetimeIndex,
                            molecule: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """
        Compute sample weights based on the concurrency of labels.
        
        Parameters:
        -----------
        label_endtimes : pd.Series
            Series of label end times indexed by label start times
        label_starttime_index : pd.DatetimeIndex
            Index of label start times (not used if molecule is provided)
        molecule : pd.DatetimeIndex, optional
            Subset of label indices to process
            
        Returns:
        --------
        pd.Series
            Sample weights indexed by label start times
        """
        # Default to all start times if molecule not provided
        if molecule is None:
            molecule = label_endtimes.index  # Use the index from label_endtimes
            
        # Filter molecule to only include timestamps that exist in label_endtimes
        valid_molecule = [t for t in molecule if t in label_endtimes.index]
        
        if not valid_molecule:
            # Return uniform weights if no valid molecules
            return pd.Series(1.0, index=molecule)
        
        # Get all available timestamps from label_endtimes for overlap calculation
        all_timestamps = label_endtimes.index
        
        # Convert to numpy arrays for efficient computation
        # Ensure we're working with datetime64 objects
        valid_molecule_arr = pd.to_datetime(valid_molecule).to_numpy()
        all_timestamps_arr = pd.to_datetime(all_timestamps).to_numpy()
        
        # Get end times as numpy array
        valid_end_times = []
        all_end_times = []
        
        for t in valid_molecule:
            end_t = label_endtimes.loc[t]
            # Handle both datetime and Series cases
            if isinstance(end_t, pd.Series):
                end_t = end_t.iloc[0] if len(end_t) > 0 else pd.NaT
            valid_end_times.append(pd.to_datetime(end_t))
        
        for t in all_timestamps:
            end_t = label_endtimes.loc[t]
            # Handle both datetime and Series cases
            if isinstance(end_t, pd.Series):
                end_t = end_t.iloc[0] if len(end_t) > 0 else pd.NaT
            all_end_times.append(pd.to_datetime(end_t))
        
        valid_end_times_arr = pd.to_datetime(valid_end_times).to_numpy()
        all_end_times_arr = pd.to_datetime(all_end_times).to_numpy()
        
        # Initialize weights
        weights = np.zeros(len(valid_molecule))
        
        # Vectorized overlap calculation
        for i in range(len(valid_molecule_arr)):
            start_time = valid_molecule_arr[i]
            end_time = valid_end_times_arr[i]
            
            # Skip if invalid timestamps
            if pd.isna(start_time) or pd.isna(end_time):
                weights[i] = 1.0
                continue
            
            # Count overlaps: label i overlaps with label j if start_i <= end_j AND start_j <= end_i
            overlaps = np.sum(
                (start_time <= all_end_times_arr) & 
                (all_timestamps_arr <= end_time)
            )
            
            # Set weight as inverse of concurrency
            if overlaps > 0:
                weights[i] = 1.0 / overlaps
            else:
                weights[i] = 1.0
        
        # Create result Series with proper index alignment
        result = pd.Series(1.0, index=molecule)
        for i, timestamp in enumerate(valid_molecule):
            result.loc[timestamp] = weights[i]
                
        return result
    
    @staticmethod
    def compute_overlap_matrix(label_endtimes: pd.Series, 
                              molecule: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Compute the overlapping matrix for labels.
        
        Parameters:
        -----------
        label_endtimes : pd.Series
            Series of label end times indexed by label start times
        molecule : pd.DatetimeIndex, optional
            Subset of label indices to process
            
        Returns:
        --------
        pd.DataFrame
            Overlapping matrix (square matrix with 1s where labels overlap)
        """
        if molecule is None:
            molecule = label_endtimes.index # label_starttime_index
            
        overlap_matrix = pd.DataFrame(0, index=molecule, columns=molecule)
        
        for i, start_time_1 in enumerate(molecule):
            end_time_1 = label_endtimes.loc[start_time_1]
            
            for j, start_time_2 in enumerate(molecule):
                end_time_2 = label_endtimes.loc[start_time_2]
                
                # Check for overlap
                if (start_time_1 <= end_time_2) and (start_time_2 <= end_time_1):
                    overlap_matrix.loc[start_time_1, start_time_2] = 1
                    
        return overlap_matrix
    
    @staticmethod
    def compute_information_driven_weights(overlap_matrix: pd.DataFrame) -> pd.Series:
        """
        Compute sample weights based on information overlap.
        
        Parameters:
        -----------
        overlap_matrix : pd.DataFrame
            Matrix of label overlaps (1 where labels overlap)
            
        Returns:
        --------
        pd.Series
            Sample weights that minimize information overlap
        """
        # Method 1: Simple inverse of overlap count
        row_sums = overlap_matrix.sum(axis=1)
        simple_weights = 1.0 / row_sums
        
        # Method 2: Analytical solution (more precise)
        try:
            analytical_weights = np.linalg.inv(overlap_matrix.values).sum(axis=1)
            analytical_weights = pd.Series(analytical_weights, index=overlap_matrix.index)
            analytical_weights = analytical_weights / analytical_weights.sum()
        except np.linalg.LinAlgError:
            # If matrix is singular, fall back to simple method
            analytical_weights = simple_weights / simple_weights.sum()
            
        return analytical_weights
    
    @staticmethod
    def weighted_bootstrap(sample_weights: pd.Series, 
                           size: Optional[int] = None,
                           replace: bool = True) -> np.ndarray:
        """
        Implements weighted bootstrap sampling.
        
        Parameters:
        -----------
        sample_weights : pd.Series
            Series of sample weights
        size : int, optional
            Number of samples to draw
        replace : bool, optional
            Whether to sample with replacement
            
        Returns:
        --------
        np.ndarray
            Array of indices selected by bootstrap
        """
        if size is None:
            size = len(sample_weights)
            
        # Convert weights to probabilities
        prob = sample_weights / sample_weights.sum()
        
        # Draw indices with given probabilities
        # Convert index to range if it's not integer-based
        if isinstance(sample_weights.index, pd.DatetimeIndex) or not np.issubdtype(sample_weights.index.dtype, np.integer):
            position_indices = np.arange(len(sample_weights))
            selected_positions = np.random.choice(position_indices, 
                                                size=size, 
                                                replace=replace, 
                                                p=prob)
            indices = sample_weights.index[selected_positions]
        else:
            indices = np.random.choice(sample_weights.index, 
                                      size=size, 
                                      replace=replace, 
                                      p=prob)
        
        return indices


class UniquenessSampling:
    """
    Class implementing uniqueness-based sampling methods.
    """
    
    @staticmethod
    def get_average_uniqueness(feature_matrix: pd.DataFrame, 
                              clusters: Optional[np.ndarray] = None, 
                              num_clusters: int = 10) -> pd.Series:
        """
        Compute the average uniqueness of each observation.
        
        Parameters:
        -----------
        feature_matrix : pd.DataFrame
            DataFrame of features
        clusters : np.ndarray, optional
            Array of cluster labels (if None, will generate clusters)
        num_clusters : int, optional
            Number of clusters to use if clusters not provided
            
        Returns:
        --------
        pd.Series
            Series of uniqueness scores
        """
        if clusters is None:
            # Generate clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(feature_matrix)
            
        # Count observations per cluster
        cluster_counts = pd.Series(clusters).value_counts()
        
        # Map counts back to observations
        uniqueness = 1.0 / cluster_counts.loc[clusters].values
        
        # Return as Series
        return pd.Series(uniqueness, index=feature_matrix.index)
    
    @staticmethod
    def get_distance_based_uniqueness(feature_matrix: pd.DataFrame, 
                                    metric: str = 'euclidean') -> pd.Series:
        """
        Compute uniqueness based on average distance to other samples.
        
        Parameters:
        -----------
        feature_matrix : pd.DataFrame
            DataFrame of features
        metric : str, optional
            Distance metric to use
            
        Returns:
        --------
        pd.Series
            Series of uniqueness scores
        """
        # Compute distance matrix
        dist_matrix = pairwise_distances(feature_matrix.values, metric=metric)
        
        # Average distance to all other points
        avg_distances = np.mean(dist_matrix, axis=1)
        
        # Normalize to [0, 1]
        if avg_distances.max() > avg_distances.min():
            normalized = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min())
        else:
            normalized = np.ones_like(avg_distances)
        
        # Return as Series
        return pd.Series(normalized, index=feature_matrix.index) 