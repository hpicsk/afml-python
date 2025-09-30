import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import itertools
import seaborn as sns


class DisjointFeatureEnsemble(BaseEstimator):
    """
    Ensemble using disjoint feature subsets.
    """
    
    def __init__(self, base_estimator: BaseEstimator, 
                 n_estimators: int = 10,
                 feature_groups: Optional[List[List[str]]] = None,
                 random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.feature_groups = feature_groups
        self.random_state = random_state
        self.estimators_ = []
        self.feature_indices_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DisjointFeatureEnsemble':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.feature_groups is None:
            features = list(X.columns)
            np.random.shuffle(features)
            self.feature_groups = np.array_split(features, self.n_estimators)
        
        for features in self.feature_groups:
            estimator = self._clone_estimator()
            estimator.fit(X[features], y)
            self.estimators_.append(estimator)
            self.feature_indices_.append(features)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([est.predict(X[feats]) for est, feats in zip(self.estimators_, self.feature_indices_)])
        return np.mean(predictions, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = np.array([est.predict_proba(X[feats]) for est, feats in zip(self.estimators_, self.feature_indices_)])
        return np.mean(probas, axis=0)

    def _clone_estimator(self) -> BaseEstimator:
        return clone(self.base_estimator)


class DiversityEnsemble(BaseEstimator):
    """
    Ensemble that promotes diversity by penalizing correlated features.
    """

    def __init__(self, base_estimator: BaseEstimator,
                 n_estimators: int = 10,
                 n_features_per_estimator: Optional[int] = None,
                 random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_features_per_estimator = n_features_per_estimator
        self.random_state = random_state
        self.estimators_ = []
        self.feature_indices_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DiversityEnsemble':
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        if self.n_features_per_estimator is None:
            self.n_features_per_estimator = int(np.sqrt(X.shape[1]))
            
        features = list(X.columns)
        
        for _ in range(self.n_estimators):
            selected_features = np.random.choice(features, self.n_features_per_estimator, replace=False)
            self.feature_indices_.append(selected_features)
            
            estimator = self._clone_estimator()
            estimator.fit(X[selected_features], y)
            self.estimators_.append(estimator)
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([est.predict(X[feats]) for est, feats in zip(self.estimators_, self.feature_indices_)])
        return np.mean(predictions, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = np.array([est.predict_proba(X[feats]) for est, feats in zip(self.estimators_, self.feature_indices_)])
        return np.mean(probas, axis=0)
        
    def _clone_estimator(self) -> BaseEstimator:
        return clone(self.base_estimator)


class StackedGeneralizationEnsemble(BaseEstimator):
    """
    Stacked generalization ensemble (stacking).
    """

    def __init__(self, base_estimators: List[BaseEstimator], 
                 meta_estimator: Optional[BaseEstimator] = None,
                 cv: int = 5,
                 use_probabilities: bool = True,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.use_probabilities = use_probabilities
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackedGeneralizationEnsemble':
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        for i, estimator in enumerate(self.base_estimators):
            for train_idx, test_idx in kf.split(X, y):
                est = self._clone_estimator(estimator)
                est.fit(X.iloc[train_idx], y.iloc[train_idx])
                
                if self.use_probabilities and hasattr(est, 'predict_proba'):
                    meta_features[test_idx, i] = est.predict_proba(X.iloc[test_idx])[:, 1]
                else:
                    meta_features[test_idx, i] = est.predict(X.iloc[test_idx])
        
        self.meta_estimator_ = self._clone_estimator(self.meta_estimator or RandomForestClassifier())
        self.meta_estimator_.fit(meta_features, y)
        
        for i, estimator in enumerate(self.base_estimators):
            self.base_estimators[i] = self._clone_estimator(estimator).fit(X, y)
            
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        return self.meta_estimator_.predict(meta_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        return self.meta_estimator_.predict_proba(meta_features)

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))
        for i, estimator in enumerate(self.base_estimators):
            if self.use_probabilities and hasattr(estimator, 'predict_proba'):
                meta_features[:, i] = estimator.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = estimator.predict(X)
        return meta_features

    def _clone_estimator(self, estimator: BaseEstimator) -> BaseEstimator:
        return clone(estimator)


class BetSizingEnsemble(BaseEstimator):
    """
    Ensemble for bet sizing based on multiple models' predictions.
    """

    def __init__(self, base_estimators: List[BaseEstimator],
                 weights: Optional[List[float]] = None,
                 bet_sizing_func: Optional[Callable] = None,
                 threshold: float = 0.5):
        self.base_estimators = base_estimators
        self.weights = weights or [1.0/len(base_estimators)] * len(base_estimators)
        self.bet_sizing_func = bet_sizing_func or self._default_bet_sizing
        self.threshold = threshold
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] > self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = np.array([est.predict_proba(X) for est in self.base_estimators])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def predict_bet_size(self, X: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.array([self.bet_sizing_func(p) for p in probas[:, 1]])

    @staticmethod
    def _default_bet_sizing(proba: float) -> float:
        if proba > 0.5:
            return (proba - 0.5) / (proba * (1 - proba))
        else:
            return -(0.5 - proba) / (proba * (1 - proba)) 