import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from ..utils.performance import PerformanceMonitor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EfficiencyAnalyzer:
    """
    Tools to analyze the trade-offs between backtest speed and accuracy.
    Helps in determining optimal subsample sizes for faster preliminary testing.
    (Derived from chap12.EfficientBacktester)
    """

    def __init__(self):
        self.monitor = PerformanceMonitor()

    def analyze_approximation_error(self,
                                    model_factory: Callable,
                                    X: pd.DataFrame, y: pd.Series,
                                    subsample_ratios: List[float] = None,
                                    n_repetitions: int = 5,
                                    cv_splits: int = 3,
                                    scoring_func: Callable = None) -> Dict[str, Any]:
        """
        Estimates the error introduced by backtesting on smaller subsamples of data.

        This helps answer: "How much performance do I lose if I only use 20% of my
        data for initial tests?"

        Args:
            model_factory: A function that returns a new, unfitted model instance.
            X, y: The full dataset.
            subsample_ratios: A list of fractions of the data to test (e.g., [0.1, 0.25, 0.5, 1.0]).
            n_repetitions: Number of times to repeat the test for each ratio to get stable estimates.
            cv_splits: Number of cross-validation folds to use.
            scoring_func: A metric function from sklearn.metrics.

        Returns:
            A dictionary containing the error analysis results.
        """
        with self.monitor.timer("approximation_error_analysis"):
            if subsample_ratios is None:
                subsample_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]

            logger.info("Running baseline CV on full dataset...")
            full_score = self._run_cv_on_sample(model_factory, X, y, cv_splits, scoring_func)
            logger.info(f"Full dataset score: {full_score:.4f}")

            ratio_results = []
            for ratio in tqdm(subsample_ratios, desc="Analyzing Subsample Ratios"):
                if ratio == 1.0:
                    ratio_scores = [full_score] * n_repetitions
                else:
                    ratio_scores = []
                    for _ in range(n_repetitions):
                        sample_indices = np.random.choice(X.index, size=int(len(X) * ratio), replace=False)
                        X_sub, y_sub = X.loc[sample_indices], y.loc[sample_indices]
                        score = self._run_cv_on_sample(model_factory, X_sub, y_sub, cv_splits, scoring_func)
                        ratio_scores.append(score)

                mean_score = np.mean(ratio_scores)
                error = abs(mean_score - full_score)
                rel_error = error / abs(full_score) if full_score != 0 else np.inf

                ratio_results.append({
                    'ratio': ratio,
                    'mean_score': mean_score,
                    'std_score': np.std(ratio_scores),
                    'absolute_error': error,
                    'relative_error': rel_error,
                })

            results_df = pd.DataFrame(ratio_results)
            logger.info("Approximation Error Analysis Results:\n" + str(results_df))
            
            return {
                'full_dataset_score': full_score,
                'results_df': results_df,
                'performance': self.monitor.report_performance()
            }

    def _run_cv_on_sample(self, model_factory, X_sample, y_sample, n_splits, scoring_func):
        """Helper to run cross-validation on a given data sample."""
        from sklearn.metrics import mean_squared_error
        
        if scoring_func is None:
            scoring_func = mean_squared_error

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(X_sample):
            model = model_factory()
            X_train, X_test = X_sample.iloc[train_idx], X_sample.iloc[test_idx]
            y_train, y_test = y_sample.iloc[train_idx], y_sample.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(scoring_func(y_test, pred))
        
        return np.mean(scores) if scores else np.nan 