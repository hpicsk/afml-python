import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import from the refactored library
from afml.validation.dangers import DangerDetector
from afml.validation.robustness import RobustnessChecker


def generate_synthetic_data_for_robustness(
    n_samples=1000, n_features=5, random_state=42
):
    """Generates synthetic data for demonstrating robustness checks."""
    if random_state:
        np.random.seed(random_state)
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=pd.to_datetime(pd.date_range(start='2020-01-01', periods=n_samples))
    )
    
    y = pd.Series(
        np.random.randn(n_samples),
        index=X.index
    )
    
    return X, y

def example_dangers_and_robustness():
    """
    Example demonstrating the use of DangerDetector and RobustnessChecker.
    """
    print("--- Demonstrating Backtest Danger Detection ---")
    
    # Generate synthetic data
    X, y = generate_synthetic_data_for_robustness(n_samples=200, random_state=42)
    
    # --- 1. Look-ahead Bias Detection ---
    print("\n1. Detecting Look-ahead Bias...")
    # Introduce a look-ahead bias into one feature
    X_leaky = X.copy()
    X_leaky['feature_0'] = y.shift(-1) # feature_0 at time t is y at time t+1
    
    suspicious_features = DangerDetector.detect_look_ahead_bias(X_leaky, y)
    print(f"Suspicious features (potential look-ahead): {suspicious_features}")
    
    # --- 2. Train/Test Overlap ---
    print("\n2. Detecting Train/Test Overlap...")
    X_train, X_test = X.iloc[:100], X.iloc[95:150] # Create overlapping sets
    
    overlap_results = DangerDetector.check_train_test_overlap(X_train, X_test)
    print(f"Overlap detected: {overlap_results['has_overlap']}")
    print(f"Number of overlapping samples: {overlap_results['overlap_count']}")
    print(f"Is temporally sound (train ends before test begins): {overlap_results['is_temporally_sound']}")

    print("\n\n--- Demonstrating Backtest Robustness Checks ---")
    
    # Define a simple model class and parameters for the checks
    model_class = LinearRegression
    params = {}
    
    X, y = generate_synthetic_data_for_robustness(n_samples=500, random_state=123)

    # --- 1. Walk-Forward Validation ---
    print("\n1. Performing Walk-Forward Validation...")
    wf_results = RobustnessChecker.walk_forward_validation(
        model_class, params, X, y,
        initial_train_size=250, step_size=50,
        metric=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    )
    print("Walk-forward scores:", [round(s, 4) for s in wf_results['test_scores']])
    print(f"Walk-forward performance consistency: {wf_results['consistency']}")

    # --- 2. Subsample Robustness ---
    print("\n2. Performing Subsample Robustness Test...")
    subsample_results = RobustnessChecker.subsample_robustness_test(
        model_class, params, X, y,
        n_iterations=5, sample_fraction=0.8,
        metric=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    )
    print("Subsample scores:", [round(s, 4) for s in subsample_results['test_scores']])
    print(f"Subsample performance consistency: {subsample_results['stability']}")

if __name__ == "__main__":
    example_dangers_and_robustness()