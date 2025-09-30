import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Callable
from sklearn.base import BaseEstimator
from scipy.stats import norm
from scipy.optimize import minimize

class MetaLabeler:
    """
    Meta-labeling approach for bet sizing.
    """
    def __init__(self, primary_model: BaseEstimator, 
                secondary_model: Union[BaseEstimator, None] = None,
                continuous_secondary: bool = False,
                prob_threshold: float = 0.5,
                scale_positions: bool = True):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.continuous_secondary = continuous_secondary
        self.prob_threshold = prob_threshold
        self.scale_positions = scale_positions
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
           sample_weight: Optional[np.ndarray] = None) -> 'MetaLabeler':
        if hasattr(self.primary_model, 'fit'):
            if sample_weight is not None:
                self.primary_model.fit(X, np.sign(y), sample_weight=sample_weight)
            else:
                self.primary_model.fit(X, np.sign(y))
        
        primary_preds = self.primary_model.predict(X)
        meta_labels = (primary_preds * np.sign(y) > 0).astype(int)
        
        if self.secondary_model is not None and hasattr(self.secondary_model, 'fit'):
            valid_indices = primary_preds != 0
            if np.sum(valid_indices) > 0:
                if sample_weight is not None:
                    self.secondary_model.fit(X[valid_indices], meta_labels[valid_indices], 
                                           sample_weight=sample_weight[valid_indices])
                else:
                    self.secondary_model.fit(X[valid_indices], meta_labels[valid_indices])
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        primary_preds = self.primary_model.predict(X)
        positions = primary_preds.copy()
        
        if self.secondary_model is not None:
            if self.continuous_secondary:
                secondary_preds = self.secondary_model.predict(X)
                if self.scale_positions:
                    min_pred, max_pred = np.min(secondary_preds), np.max(secondary_preds)
                    scaled_preds = (secondary_preds - min_pred) / (max_pred - min_pred) if max_pred > min_pred else np.ones_like(secondary_preds) * 0.5
                    filter_mask = scaled_preds >= self.prob_threshold
                    positions = positions * filter_mask * scaled_preds
                else:
                    filter_mask = secondary_preds >= self.prob_threshold
                    positions = positions * filter_mask
            else:
                if hasattr(self.secondary_model, 'predict_proba'):
                    try:
                        class_1_idx = list(self.secondary_model.classes_).index(1)
                        secondary_probs = self.secondary_model.predict_proba(X)[:, class_1_idx]
                    except ValueError:
                        secondary_probs = np.zeros(X.shape[0])
                    
                    if self.scale_positions:
                        filter_mask = secondary_probs >= self.prob_threshold
                        positions = positions * filter_mask * secondary_probs
                    else:
                        filter_mask = secondary_probs >= self.prob_threshold
                        positions = positions * filter_mask
                else:
                    secondary_preds = self.secondary_model.predict(X)
                    positions = positions * secondary_preds
        return positions


class KellyBetSizing:
    """
    Kelly Criterion for optimal bet sizing.
    """
    
    @staticmethod
    def optimal_kelly(p: float, win_loss_ratio: float) -> float:
        return p - (1 - p) / win_loss_ratio if win_loss_ratio > 0 else 0

    @staticmethod
    def restricted_kelly(p: float, win_loss_ratio: float, max_bet: float = 0.05) -> float:
        return min(max_bet, KellyBetSizing.optimal_kelly(p, win_loss_ratio))

    @staticmethod
    def kelly_with_estimate_uncertainty(p: float, win_loss_ratio: float, 
                                      p_std: float, wl_std: float,
                                      confidence: float = 0.95) -> float:
        p_adj = p - norm.ppf(1 - (1 - confidence) / 2) * p_std
        wl_adj = win_loss_ratio - norm.ppf(1 - (1 - confidence) / 2) * wl_std
        return KellyBetSizing.optimal_kelly(p_adj, wl_adj)


class BetSizingStrategies:
    """
    Various bet sizing strategies.
    """
    
    @staticmethod
    def size_by_probability(probabilities: np.ndarray, 
                          max_pos_size: float = 1.0,
                          min_prob_threshold: float = 0.5,
                          scaling: str = 'linear') -> np.ndarray:
        if scaling == 'linear':
            sizes = (probabilities - min_prob_threshold) / (1 - min_prob_threshold)
        elif scaling == 'normal':
            sizes = norm.ppf(probabilities)
        else:
            raise ValueError("Unsupported scaling method")
        return np.clip(sizes, 0, max_pos_size)
    
    @staticmethod
    def size_by_kelly(probabilities: np.ndarray, 
                    win_loss_ratio: Union[float, np.ndarray],
                    fraction: float = 0.5,
                    max_pos_size: float = 1.0) -> np.ndarray:
        kelly_sizes = np.array([KellyBetSizing.optimal_kelly(p, win_loss_ratio) for p in probabilities])
        return np.clip(kelly_sizes * fraction, 0, max_pos_size) 