import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
import warnings
import itertools
from joblib import Parallel, delayed
import seaborn as sns
from tqdm import tqdm

class PurgedKFold:
    """
    Purged k-fold cross-validation.
    This method is suitable for financial data where observations may overlap.
    Purging removes training samples whose labels overlap with the test set.
    Embargoing removes training samples that immediately follow the test set.
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
             groups: Optional[pd.Series] = None, t1: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must be a pandas DataFrame with a DatetimeIndex.")
        if t1 is None:
            raise ValueError("The 't1' series (event end times) is required for purging.")
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate embargo delta based on total time span
        total_duration = X.index[-1] - X.index[0]
        embargo_delta = total_duration * self.embargo_pct

        kf = KFold(n_splits=self.n_splits)
        for train_iloc, test_iloc in kf.split(indices):
            test_times = X.index[test_iloc]
            test_start, test_end = test_times.min(), test_times.max()

            # 1. Purge from training set
            train_iloc_purged = []
            for i in train_iloc:
                train_time = X.index[i]
                train_end_time = t1.get(train_time)
                if pd.isna(train_end_time):
                    # If no end time, assume it doesn't overlap forward
                    if train_time < test_start:
                        train_iloc_purged.append(i)
                    continue
                
                # Overlap condition: (start1 <= end2) and (start2 <= end1)
                # Here: (train_time <= test_end) and (test_start <= train_end_time)
                if not ((train_time <= test_end) and (test_start <= train_end_time)):
                    train_iloc_purged.append(i)
            
            # 2. Embargo from the purged training set
            train_iloc_final = []
            embargo_end_time = test_end + embargo_delta
            for i in train_iloc_purged:
                train_time = X.index[i]
                if train_time < test_start or train_time > embargo_end_time:
                    train_iloc_final.append(i)

            yield np.array(train_iloc_final), test_iloc


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold cross-validation.
    This method generates all combinations of test sets from the K folds.
    """
    
    def __init__(self, n_splits: int = 5, 
                n_test_splits: int = 2,
                embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        import itertools
        return len(list(itertools.combinations(range(self.n_splits), self.n_test_splits)))
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
             groups: Optional[pd.Series] = None, t1: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must be a pandas DataFrame with a DatetimeIndex.")
        if t1 is None:
            raise ValueError("The 't1' series (event end times) is required for purging.")

        n_samples = len(X)
        indices = np.arange(n_samples)
        total_duration = X.index[-1] - X.index[0]
        embargo_delta = total_duration * self.embargo_pct

        # Create base folds
        kf = KFold(n_splits=self.n_splits)
        base_folds = [test_idx for _, test_idx in kf.split(indices)]

        # Generate combinations of test folds
        for test_fold_indices in itertools.combinations(range(self.n_splits), self.n_test_splits):
            test_iloc = np.concatenate([base_folds[i] for i in test_fold_indices])
            train_iloc = np.concatenate([base_folds[i] for i in range(self.n_splits) if i not in test_fold_indices])

            test_times = X.index[test_iloc]
            test_start, test_end = test_times.min(), test_times.max()

            # Purge and embargo logic is the same as in PurgedKFold
            train_iloc_purged = []
            for i in train_iloc:
                train_time = X.index[i]
                train_end_time = t1.get(train_time)
                if pd.isna(train_end_time):
                    if train_time < test_start:
                        train_iloc_purged.append(i)
                    continue
                
                if not ((train_time <= test_end) and (test_start <= train_end_time)):
                    train_iloc_purged.append(i)

            train_iloc_final = []
            embargo_end_time = test_end + embargo_delta
            for i in train_iloc_purged:
                train_time = X.index[i]
                if train_time < test_start or train_time > embargo_end_time:
                    train_iloc_final.append(i)
            
            yield np.array(train_iloc_final), test_iloc


class WalkForwardAnalysis:
    """
    Walk-forward analysis for time-series models.
    """

    def __init__(self, model_factory: Callable[..., BaseEstimator],
                 param_grid: Dict[str, List[Any]],
                 n_splits: int = 5,
                 embargo_pct: float = 0.01,
                 scoring: Union[str, Callable] = 'sharpe_ratio',
                 n_jobs: int = -1):
        self.model_factory = model_factory
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.results_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series, 
            t1: pd.Series, returns: Optional[pd.Series] = None) -> 'WalkForwardAnalysis':
        param_combinations = list(itertools.product(*self.param_grid.values()))
        
        cv = PurgedKFold(self.n_splits, self.embargo_pct)

        def evaluate_params(params_idx):
            params = dict(zip(self.param_grid.keys(), param_combinations[params_idx]))
            model = self.model_factory(**params)
            
            scores = []
            for train_idx, test_idx in cv.split(X, t1=t1):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                
                score = self.score_model(model, X.iloc[test_idx], y.iloc[test_idx], 
                                         returns.iloc[test_idx] if returns is not None else None)
                scores.append(score)
            
            return np.mean(scores)

        self.results_ = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_params)(i) for i in tqdm(range(len(param_combinations)))
        )
        return self
        
    def get_best_params(self) -> List[Dict[str, Any]]:
        best_idx = np.argmax(self.results_)
        param_combinations = list(itertools.product(*self.param_grid.values()))
        return dict(zip(self.param_grid.keys(), param_combinations[best_idx]))

    def score_model(self, model, X, y, returns=None):
        if self.scoring == 'sharpe_ratio':
            if returns is None:
                raise ValueError("Returns must be provided for sharpe_ratio scoring")
            positions = model.predict(X)
            from ..validation.statistics import PerformanceStatistics
            return PerformanceStatistics.sharpe_ratio_from_positions(returns.values, positions)
        elif callable(self.scoring):
            return self.scoring(y, model.predict(X))
        else:
            raise ValueError("Scoring method not supported") 