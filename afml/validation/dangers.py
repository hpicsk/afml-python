import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DangerDetector:
    """
    A collection of static methods to detect common backtesting pitfalls and biases.
    (Derived from chap11.BacktestDangerDetector)
    """

    @staticmethod
    def detect_look_ahead_bias(features: pd.DataFrame, target: pd.Series, max_forward_lag: int = 5) -> List[str]:
        """
        Detects potential look-ahead bias by checking if features are more correlated
        with future returns than with past or contemporaneous returns.

        Args:
            features: DataFrame of features.
            target: Series of target values (e.g., returns).
            max_forward_lag: The maximum number of steps to look into the future.

        Returns:
            A list of feature names that are suspiciously correlated with future target values.
        """
        suspicious_features = []
        logger.info("Detecting look-ahead bias...")
        for col in features.columns:
            corrs = {lag: features[col].corr(target.shift(-lag)) for lag in range(max_forward_lag + 1)}
            
            # Suspicious if correlation peaks in the future (lag > 0)
            if len(corrs) > 1:
                peak_lag = max(corrs, key=corrs.get)
                if peak_lag > 0 and corrs[peak_lag] > corrs[0] * 1.5 and corrs[peak_lag] > 0.1:
                    suspicious_features.append(col)
                    logger.warning(f"Feature '{col}' has peak correlation of {corrs[peak_lag]:.3f} at future lag {peak_lag}.")
        return suspicious_features

    @staticmethod
    def detect_data_leakage(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Tests for data leakage by comparing in-sample vs. out-of-sample performance.
        A very large drop in performance can indicate overfitting or data leakage.

        Args:
            model: The scikit-learn compatible model to test.
            X_train, y_train: Training data.
            X_test, y_test: Testing data.

        Returns:
            A dictionary with the train score, test score, and a leakage risk assessment.
        """
        logger.info("Detecting potential data leakage...")
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        # Determine metric based on model type
        if hasattr(model_clone, "predict_proba"):
            train_pred = model_clone.predict_proba(X_train)[:, 1]
            test_pred = model_clone.predict_proba(X_test)[:, 1]
            train_score = roc_auc_score(y_train, train_pred)
            test_score = roc_auc_score(y_test, test_pred)
            metric_name = 'AUC'
        else: # Regression or non-probabilistic classification
            train_pred = model_clone.predict(X_train)
            test_pred = model_clone.predict(X_test)
            train_score = mean_squared_error(y_train, train_pred, squared=False)
            test_score = mean_squared_error(y_test, test_pred, squared=False)
            metric_name = 'RMSE'

        # Calculate performance drop
        if metric_name == 'RMSE': # Lower is better
            perf_drop = (test_score - train_score) / train_score if train_score > 0 else np.inf
        else: # Higher is better
            perf_drop = (train_score - test_score) / train_score if train_score > 0 else np.inf

        leakage_risk = 'High' if perf_drop > 0.5 else 'Medium' if perf_drop > 0.2 else 'Low'
        
        logger.info(f"Train {metric_name}: {train_score:.4f}, Test {metric_name}: {test_score:.4f}")
        logger.info(f"Performance drop: {perf_drop:.2%}. Leakage risk assessed as: {leakage_risk}")

        return {'train_score': train_score, 'test_score': test_score, 'leakage_risk': leakage_risk}

    @staticmethod
    def detect_survivorship_bias(current_symbols: List[str], historical_universe_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects potential survivorship bias by comparing the returns of currently
        active symbols vs. delisted symbols.

        Args:
            current_symbols: A list of symbols that are currently active or in the final universe.
            historical_universe_returns: A DataFrame of returns for ALL symbols that ever existed
                                         in the universe during the backtest period.

        Returns:
            A dictionary analyzing the impact of survivorship bias.
        """
        logger.info("Detecting survivorship bias...")
        all_symbols = historical_universe_returns.columns.tolist()
        delisted_symbols = [s for s in all_symbols if s not in current_symbols]

        if not delisted_symbols:
            logger.info("No delisted symbols found. No survivorship bias detected.")
            return {'has_bias': False, 'bias_impact': 0.0}

        survivor_returns = historical_universe_returns[current_symbols].mean(axis=1)
        delisted_returns = historical_universe_returns[delisted_symbols].mean(axis=1)

        avg_survivor_return = survivor_returns.mean() * 252
        avg_delisted_return = delisted_returns.mean() * 252

        bias_impact = avg_survivor_return - avg_delisted_return
        has_bias = bias_impact > 0.01  # 1% difference in annualized returns is a flag

        logger.info(f"Survivor avg annual return: {avg_survivor_return:.2%}")
        logger.info(f"Delisted avg annual return: {avg_delisted_return:.2%}")
        logger.warning(f"Survivorship bias impact (annualized): {bias_impact:.2%}")

        return {
            'has_bias': has_bias,
            'bias_impact': bias_impact,
            'survivor_return': avg_survivor_return,
            'delisted_return': avg_delisted_return
        }

    @staticmethod
    def check_train_test_overlap(train_set: pd.DataFrame, test_set: pd.DataFrame) -> Dict[str, Any]:
        """
        Checks for direct index overlap between training and testing sets.

        Args:
            train_set: The training DataFrame.
            test_set: The testing DataFrame.

        Returns:
            A dictionary detailing the overlap.
        """
        logger.info("Checking for train/test data overlap...")
        train_indices = train_set.index
        test_indices = test_set.index

        overlap_indices = train_indices.intersection(test_indices)
        has_overlap = not overlap_indices.empty
        is_temporally_sound = train_indices.max() < test_indices.min() if has_overlap else True
        
        if has_overlap:
             logger.warning(f"Overlap detected! {len(overlap_indices)} samples are in both train and test sets.")
        else:
            logger.info("No direct overlap found.")

        return {
            'has_overlap': has_overlap,
            'overlap_count': len(overlap_indices),
            'is_temporally_sound': is_temporally_sound
        } 