import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, t, skew, kurtosis, probplot
from scipy.optimize import minimize, minimize_scalar
import warnings
from tqdm import tqdm
import datetime as dt
import random
from joblib import Parallel, delayed

from afml.portfolio.bet_sizing import (
    MetaLabeler,
    KellyBetSizing,
    BetSizingStrategies
)
from afml.core.backtester import StrategyBacktester


# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set1')


def generate_synthetic_data(n_samples=1000, n_features=5):
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    return X, y

def example_with_synthetic_data():
    X, y = generate_synthetic_data()
    
    # 1. MetaLabeler
    primary_model = RandomForestClassifier(n_estimators=10, random_state=42)
    secondary_model = LogisticRegression()
    meta_labeler = MetaLabeler(primary_model, secondary_model)
    meta_labeler.fit(X, y)
    positions = meta_labeler.predict(X)
    
    print("MetaLabeler positions sample:", positions[:10])
    
    # 2. Bet Sizing Strategies
    probs = np.random.rand(len(X))
    prob_sizes = BetSizingStrategies.size_by_probability(probs)
    kelly_sizes = BetSizingStrategies.size_by_kelly(probs, win_loss_ratio=1.5)
    
    print("\nProbability-based sizes sample:", prob_sizes[:10])
    print("Kelly-based sizes sample:", kelly_sizes[:10])
    
    # 3. Backtester
    returns = pd.Series(np.random.randn(len(X)) * 0.01, index=X.index)
    backtester = StrategyBacktester()
    results = backtester.backtest(returns, pd.Series(positions, index=X.index))
    
    print("\nBacktest Sharpe Ratio:", results['sharpe_ratio'])
    
    return results

if __name__ == "__main__":
    example_with_synthetic_data()
