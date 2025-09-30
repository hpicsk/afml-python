import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Dict, Any

from ..utils.performance import PerformanceMonitor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingBacktester:
    """
    Performs a backtest on data that is too large to fit into memory.
    It processes the data in chunks, updating the model sequentially.
    (Derived from chap12.StreamingBacktester)
    """

    def __init__(self,
                 data_loader: Callable[[int, int], pd.DataFrame],
                 total_rows: int,
                 feature_cols: List[str],
                 target_col: str,
                 chunk_size: int = 10000):
        """
        Initialize the StreamingBacktester.

        Args:
            data_loader: A function that loads a slice of data, e.g., `lambda start, end: df.iloc[start:end]`.
            total_rows: The total number of rows in the dataset.
            feature_cols: A list of column names to be used as features.
            target_col: The name of the target column.
            chunk_size: The number of rows to process in each chunk.
        """
        self.data_loader = data_loader
        self.total_rows = total_rows
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.chunk_size = chunk_size
        self.monitor = PerformanceMonitor()
        logger.info(f"Initialized StreamingBacktester with chunk size {chunk_size}.")

    def run(self,
            model_factory: Callable,
            initial_train_size: int = 1000,
            retrain_freq: int = 500) -> Dict[str, Any]:
        """
        Run the streaming backtest.

        Args:
            model_factory: A function that returns a new, unfitted model instance.
            initial_train_size: The number of initial samples to use for the first model training.
            retrain_freq: How often to retrain the model (in number of rows).

        Returns:
            A dictionary containing the results DataFrame and performance metrics.
        """
        with self.monitor.timer("streaming_backtest"):
            # Initial training
            logger.info(f"Loading initial training data (rows 0-{initial_train_size})...")
            initial_data = self.data_loader(0, initial_train_size)
            X_train, y_train = initial_data[self.feature_cols], initial_data[self.target_col]

            logger.info("Training initial model...")
            model = model_factory()
            model.fit(X_train, y_train)

            # Streaming prediction and retraining
            predictions, actuals, timestamps = [], [], []
            current_pos = initial_train_size
            next_retrain_pos = current_pos + retrain_freq

            pbar = tqdm(total=self.total_rows, initial=initial_train_size, desc="Streaming Backtest")
            while current_pos < self.total_rows:
                chunk_end = min(current_pos + self.chunk_size, self.total_rows)
                chunk = self.data_loader(current_pos, chunk_end)

                for idx, row in chunk.iterrows():
                    x_live = pd.DataFrame([row[self.feature_cols].values], columns=self.feature_cols)
                    y_pred = model.predict(x_live)[0]

                    predictions.append(y_pred)
                    actuals.append(row[self.target_col])
                    timestamps.append(idx)

                    # Check for retraining
                    absolute_pos = current_pos + len(predictions) - initial_train_size
                    if absolute_pos >= next_retrain_pos:
                        logger.info(f"Retraining model at row {absolute_pos}...")
                        train_start = max(0, absolute_pos - initial_train_size)
                        retrain_data = self.data_loader(train_start, absolute_pos)
                        X_retrain = retrain_data[self.feature_cols]
                        y_retrain = retrain_data[self.target_col]

                        model = model_factory()
                        model.fit(X_retrain, y_retrain)
                        next_retrain_pos = absolute_pos + retrain_freq
                
                pbar.update(len(chunk))
                current_pos = chunk_end
            pbar.close()

            results_df = pd.DataFrame({'actual': actuals, 'predicted': predictions}, index=timestamps)
            rmse = np.sqrt(np.mean((results_df['actual'] - results_df['predicted']) ** 2))
            logger.info(f"Streaming backtest complete. Final RMSE: {rmse:.4f}")

            return {'results': results_df, 'rmse': rmse, 'performance': self.monitor.report_performance()}


class KalmanFilterBacktester:
    """
    An extremely efficient single-pass backtester using a Kalman filter for sequential
    parameter updates of a linear model. This avoids retraining from scratch.
    (Derived from chap12.KalmanFilterBacktester)
    """

    def __init__(self, forget_factor: float = 0.999, reg_factor: float = 1e-5):
        """
        Initialize the Kalman filter backtester.

        Args:
            forget_factor: Factor for how quickly to adapt to new data (0 to 1). Lower is faster.
            reg_factor: Regularization factor for numerical stability.
        """
        self.forget_factor = forget_factor
        self.reg_factor = reg_factor
        self.theta = None  # Model parameters (state)
        self.P = None      # Parameter covariance matrix (uncertainty)
        self.monitor = PerformanceMonitor()
        logger.info(f"Initialized KalmanFilterBacktester with forget_factor={forget_factor}.")

    def _initialize(self, n_features: int):
        """Initializes the filter's state and covariance."""
        self.theta = np.zeros(n_features)
        self.P = np.eye(n_features) * 1000  # Large initial uncertainty

    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Run a backtest using the Kalman filter.

        Args:
            X: DataFrame of features.
            y: Series of target values.

        Returns:
            A dictionary containing the results DataFrame and performance metrics.
        """
        with self.monitor.timer("kalman_filter_backtest"):
            self._initialize(X.shape[1])
            predictions = np.zeros(len(X))

            for i in tqdm(range(len(X)), desc="Kalman Filter Backtest"):
                x_i = X.iloc[i].values
                y_i = y.iloc[i]

                # Predict before update
                y_pred = np.dot(x_i, self.theta)
                predictions[i] = y_pred
                prediction_error = y_i - y_pred

                # Update step
                self.P /= self.forget_factor
                PxT = np.dot(self.P, x_i)
                kalman_gain = PxT / (np.dot(x_i, PxT) + self.reg_factor)
                self.theta += kalman_gain * prediction_error
                self.P -= np.outer(kalman_gain, PxT)

            results_df = pd.DataFrame({'actual': y.values, 'predicted': predictions}, index=y.index)
            rmse = np.sqrt(np.mean((results_df['actual'] - results_df['predicted']) ** 2))
            logger.info(f"Kalman filter backtest complete. Final RMSE: {rmse:.4f}")

            return {
                'results': results_df,
                'rmse': rmse,
                'final_parameters': pd.Series(self.theta, index=X.columns),
                'performance': self.monitor.report_performance()
            } 