import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from afml.machine_learning.feature_importance import FeatureImportance
from afml.machine_learning.feature_importance_analysis import (
    TimeSeriesFeatureImportance,
    FeatureInteractionImportance
)

def create_rf():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

def mda_wrapper(model, X, y):
    return FeatureImportance.get_mda_feature_importances(model, X, y, cv=3)

def example_with_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 10), columns=[f'f{i}' for i in range(10)],
                       index=pd.date_range(start='2020-01-01', periods=200, freq='B'))
    y = pd.Series(np.random.randint(0, 2, 200), index=X.index)

    # 1. Feature Importance Clustering
    model = create_rf().fit(X, y)
    mdi = FeatureImportance.get_mdi_feature_importances(model, list(X.columns))
    cluster_df = FeatureImportance.feature_importance_clustering(X, mdi, n_clusters=5)
    print("Feature Importance Clustering:\n", cluster_df)
    
    # 2. Select Features from Clusters
    selected_features = FeatureImportance.select_features_from_clusters(cluster_df)
    print("\nSelected features from clusters:", selected_features)
    
    # 3. Time-Series Feature Importance
    rolling_mda = TimeSeriesFeatureImportance.rolling_mean_decrease_accuracy(
        create_rf(), X, y, window_size=50, step_size=10, test_size=10, n_jobs=-1
    )
    if rolling_mda is not None and not rolling_mda.empty:
        print("\nRolling MDA:\n", rolling_mda.head())
        TimeSeriesFeatureImportance.plot_rolling_importance(rolling_mda)
    
    # 4. Feature Interaction Importance
    interaction_df = FeatureInteractionImportance.pairwise_feature_importance(model, X, y, n_jobs=-1)
    if interaction_df is not None and not interaction_df.empty:
        print("\nPairwise Feature Interactions:\n", interaction_df.head())
        FeatureInteractionImportance.visualize_interaction_network(interaction_df)


if __name__ == "__main__":
    example_with_synthetic_data()