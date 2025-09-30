import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustnessChecker:
    """
    A collection of static methods to check the robustness of backtesting results.
    (Derived from chap11.BacktestRobustnessChecker)
    """

    @staticmethod
    def walk_forward_validation(model_class: type, params: Dict,
                                X: pd.DataFrame, y: pd.Series,
                                initial_train_size: int, step_size: int,
                                metric: Callable = roc_auc_score) -> Dict[str, Any]:
        """
        Performs walk-forward validation to test model stability over time.

        Args:
            model_class: The model class to instantiate (e.g., RandomForestClassifier).
            params: Dictionary of parameters for the model.
            X: Full feature DataFrame.
            y: Full target Series.
            initial_train_size: The size of the initial training window.
            step_size: The number of samples to step forward for each validation fold.
            metric: A scoring function from sklearn.metrics.

        Returns:
            A dictionary containing the test scores for each fold and summary statistics.
        """
        logger.info("Performing walk-forward validation...")
        tscv = TimeSeriesSplit(n_splits=(len(X) - initial_train_size) // step_size)
        scores = []

        for train_idx, test_idx in tscv.split(X):
            # To implement a true walk-forward, we need to manually adjust the splits
            # For simplicity, this example uses TimeSeriesSplit as an approximation.
            # A more accurate implementation would expand the training window.
            model = model_class(**params)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test) if not hasattr(model, "predict_proba") else model.predict_proba(X_test)[:, 1]
            scores.append(metric(y_test, pred))

        mean_score = np.mean(scores)
        score_std = np.std(scores)
        consistency = 'High' if score_std / abs(mean_score) < 0.2 else 'Medium' if score_std / abs(mean_score) < 0.5 else 'Low'

        logger.info(f"Walk-forward scores: {[f'{s:.3f}' for s in scores]}")
        logger.info(f"Mean Score: {mean_score:.4f}, Std Dev: {score_std:.4f}, Consistency: {consistency}")
        
        return {'test_scores': scores, 'mean_score': mean_score, 'std_score': score_std, 'consistency': consistency}

    @staticmethod
    def subsample_robustness_test(model_class: type, params: Dict,
                                  X: pd.DataFrame, y: pd.Series,
                                  n_iterations: int = 20, sample_fraction: float = 0.8,
                                  metric: Callable = roc_auc_score) -> Dict[str, Any]:
        """
        Tests model stability by training on random subsamples of the data.

        Args:
            model_class: The model class to instantiate.
            params: Dictionary of parameters for the model.
            X, y: The training data.
            n_iterations: Number of subsamples to test.
            sample_fraction: The fraction of data to use in each subsample.
            metric: A scoring function.

        Returns:
            A dictionary containing the scores and stability assessment.
        """
        logger.info("Performing subsample robustness test...")
        scores = []
        for _ in range(n_iterations):
            sample_indices = np.random.choice(X.index, size=int(len(X) * sample_fraction), replace=False)
            X_sample, y_sample = X.loc[sample_indices], y.loc[sample_indices]
            
            # Simple train-test split for this subsample
            train_size = int(len(X_sample) * 0.7)
            X_train, X_test = X_sample.iloc[:train_size], X_sample.iloc[train_size:]
            y_train, y_test = y_sample.iloc[:train_size], y_sample.iloc[train_size:]

            model = model_class(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test) if not hasattr(model, "predict_proba") else model.predict_proba(X_test)[:, 1]
            scores.append(metric(y_test, pred))

        mean_score = np.mean(scores)
        score_std = np.std(scores)
        stability = 'High' if score_std / abs(mean_score) < 0.1 else 'Medium' if score_std / abs(mean_score) < 0.3 else 'Low'
        
        logger.info(f"Subsample mean score: {mean_score:.4f}, Std Dev: {score_std:.4f}, Stability: {stability}")

        return {'test_scores': scores, 'mean_score': mean_score, 'std_score': score_std, 'stability': stability}

    @staticmethod
    def parameter_stability_test(model_class: type, param_grid: Dict[str, list],
                                 X: pd.DataFrame, y: pd.Series, n_periods: int = 4) -> Dict[str, Any]:
        """
        Tests if the optimal model parameters are stable across different time periods.

        Args:
            model_class: The model class to instantiate.
            param_grid: A dictionary of parameters to test (like GridSearchCV).
            X, y: The full dataset.
            n_periods: Number of time periods to split the data into.

        Returns:
            A dictionary with the best parameters for each period and a stability assessment.
        """
        logger.info("Performing parameter stability test...")
        period_len = len(X) // n_periods
        best_params_per_period = []

        from sklearn.model_selection import GridSearchCV
        
        for i in range(n_periods):
            period_slice = slice(i * period_len, (i + 1) * period_len)
            X_period, y_period = X.iloc[period_slice], y.iloc[period_slice]

            # Use a simple time series split within the period for validation
            tscv = TimeSeriesSplit(n_splits=2)
            grid_search = GridSearchCV(estimator=model_class(), param_grid=param_grid, cv=tscv)
            grid_search.fit(X_period, y_period)
            best_params_per_period.append(grid_search.best_params_)
        
        # Assess stability by counting how many times each parameter setting was optimal
        param_counts = pd.Series([str(p) for p in best_params_per_period]).value_counts()
        most_common_freq = param_counts.iloc[0] / n_periods if not param_counts.empty else 0
        stability = 'High' if most_common_freq >= 0.75 else 'Medium' if most_common_freq >= 0.5 else 'Low'

        logger.info(f"Best parameters per period: {best_params_per_period}")
        logger.info(f"Most common parameter set occurred {most_common_freq:.0%} of the time. Stability: {stability}")
        
        return {'best_params_per_period': best_params_per_period, 'stability': stability} 