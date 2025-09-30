import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from afml.machine_learning.feature_importance import FeatureImportance
from afml.machine_learning.ensembling import (
    DisjointFeatureEnsemble,
    DiversityEnsemble,
    StackedGeneralizationEnsemble,
    BetSizingEnsemble
)
from afml.data.sampling import SequentialBootstrap


def example_with_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # 1. Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    mdi = FeatureImportance.get_mdi_feature_importances(rf, X.columns)
    mda = FeatureImportance.get_mda_feature_importances(rf, X, y)
    
    FeatureImportance.plot_feature_importances(mdi, "MDI Feature Importance")
    FeatureImportance.plot_feature_importances(mda, "MDA Feature Importance")

    # 2. Disjoint Feature Ensemble
    dfe = DisjointFeatureEnsemble(RandomForestClassifier(n_estimators=10), n_estimators=5, random_state=42)
    dfe.fit(X, y)
    print("Disjoint Feature Ensemble prediction:", dfe.predict(X).shape)
    
    # 3. Diversity Ensemble
    de = DiversityEnsemble(RandomForestClassifier(n_estimators=10), n_estimators=5, random_state=42)
    de.fit(X, y)
    print("Diversity Ensemble prediction:", de.predict(X).shape)

    # 4. Stacked Generalization Ensemble
    base_estimators = [
        RandomForestClassifier(n_estimators=10, random_state=42),
        LogisticRegression()
    ]
    sge = StackedGeneralizationEnsemble(base_estimators, meta_estimator=LogisticRegression(), cv=3)
    sge.fit(X, y)
    print("Stacked Ensemble prediction:", sge.predict(X).shape)
    
    # 5. Bet Sizing Ensemble
    bse = BetSizingEnsemble(base_estimators)
    print("Bet Sizing Ensemble prediction:", bse.predict(X).shape)
    print("Bet Sizing Ensemble bet sizes:", bse.predict_bet_size(X).shape)
    
    return mdi, mda

if __name__ == "__main__":
    example_with_synthetic_data()