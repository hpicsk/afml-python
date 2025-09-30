import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from afml.validation.cross_validation import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    WalkForwardAnalysis
)
from afml.validation.statistics import PerformanceStatistics

def model_factory(**kwargs):
    return RandomForestClassifier(**kwargs)

def example_with_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 5), columns=[f'f{i}' for i in range(5)],
                       index=pd.date_range(start='2020-01-01', periods=200, freq='B'))
    y = pd.Series(np.random.randint(0, 2, 200), index=X.index)
    returns = pd.Series(np.random.randn(200) * 0.01, index=X.index)
    
    # Generate event end times (t1) - needed for purging
    t1 = pd.Series(X.index + pd.Timedelta(days=5), index=X.index)

    # 1. Purged K-Fold
    print("--- PurgedKFold Example ---")
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    for train_idx, test_idx in pkf.split(X, t1=t1):
        print("PurgedKFold train/test split sizes:", len(train_idx), len(test_idx))

    # 2. Combinatorial Purged K-Fold
    print("\n--- CombinatorialPurgedKFold Example ---")
    cpkf = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2, embargo_pct=0.01)
    for train_idx, test_idx in cpkf.split(X, t1=t1):
        print("CombinatorialPurgedKFold train/test split sizes:", len(train_idx), len(test_idx))

    # 3. Walk-Forward Analysis
    print("\n--- WalkForwardAnalysis Example ---")
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    wfa = WalkForwardAnalysis(model_factory, param_grid, n_splits=4, scoring=accuracy_score)
    wfa.fit(X, y, t1=t1, returns=returns)
    
    best_params = wfa.get_best_params()
    print("\nBest parameters from Walk-Forward Analysis:", best_params)
    
    return wfa

if __name__ == "__main__":
    example_with_synthetic_data()
