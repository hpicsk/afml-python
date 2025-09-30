import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.stats import norm
from collections import defaultdict
from joblib import Parallel, delayed
import warnings

class FeatureImportance:
    """
    Feature importance techniques as described in Chapter 8.
    """
    
    @staticmethod
    def get_mdi_feature_importances(model: Union[RandomForestClassifier, RandomForestRegressor], 
                             feature_names: List[str],
                             normalize: bool = True) -> pd.Series:
        """
        Calculate Mean Decrease Impurity (MDI) feature importance.
        
        Parameters:
        -----------
        model : Union[RandomForestClassifier, RandomForestRegressor]
            Trained tree-based model
        feature_names : List[str]
            List of feature names
        normalize : bool, optional
            Whether to normalize importance scores
            
        Returns:
        --------
        pd.Series
            Feature importance scores
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("Model doesn't have feature_importances_ attribute.")
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Normalize if requested
        if normalize and np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        # Create Series with feature names
        importance_series = pd.Series(importances, index=feature_names)
        
        # Sort in descending order
        return importance_series.sort_values(ascending=False)
    
    @staticmethod
    def get_mda_feature_importances(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                             cv: Optional[Union[int, TimeSeriesSplit]] = None,
                             groups: Optional[pd.Series] = None,
                             scoring: Callable = accuracy_score,
                             n_jobs: int = 1,
                             random_state: Optional[int] = None) -> pd.Series:
        """
        Calculates feature importance using Mean Decrease Accuracy (MDA), a permutation-based method.
        This implementation performs MDA within a cross-validation loop.

        Parameters:
        - model: An unfitted instance of a scikit-learn compatible model.
        - X: Feature data.
        - y: Target data.
        - cv: A cross-validation splitter object (e.g., from sklearn.model_selection) or an integer for KFold.
              If None, a single split on the whole data is used (not recommended).
        - groups: Group labels for CV splitters that require it (e.g., PurgedKFold).
        - scoring: A callable scoring function (e.g., accuracy_score, f1_score).
        - n_jobs: Number of parallel jobs to run.
        - random_state: Seed for reproducibility.

        Returns:
        - A pandas Series with feature importances.
        """
        
        if groups is not None:
            warnings.warn("The 'groups' parameter is provided but not currently used by mean_decrease_accuracy.", UserWarning)

        if cv is None:
            # If no CV, use a single train-test split (not recommended for MDA)
            cv_splitter = [(np.arange(len(X)), np.arange(len(X)))]
        elif isinstance(cv, int):
            # Default to KFold if an integer is provided
            from sklearn.model_selection import KFold
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            cv_splitter = list(cv_splitter.split(X, y))
        else:
            # Use the provided CV splitter object, passing groups if it's supported
            import inspect
            sig = inspect.signature(cv.split)
            if 'groups' in sig.parameters:
                cv_splitter = list(cv.split(X, y, groups=groups))
            else:
                cv_splitter = list(cv.split(X, y))

        if random_state is not None:
            np.random.seed(random_state)

        # Helper function to compute permuted score for a single feature
        def get_permuted_score(model, X_test, y_test, feat_name, scoring):
            X_test_permuted = X_test.copy()
            col_idx = X_test.columns.get_loc(feat_name)
            
            # Permute the column
            permuted_col = X_test_permuted.iloc[:, col_idx].values.copy()
            np.random.shuffle(permuted_col)
            X_test_permuted.iloc[:, col_idx] = permuted_col
            
            # Score after permutation
            if hasattr(model, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred_permuted = model.predict_proba(X_test_permuted)
                return scoring(y_test, y_pred_permuted)
            else:
                y_pred_permuted = model.predict(X_test_permuted)
                return scoring(y_test, y_pred_permuted)
        
        # Store results
        all_importances = defaultdict(list)
        
        # Iterate over CV folds
        for train_idx, test_idx in cv_splitter:
            model_clone = clone(model)
            model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
            
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # Score on test set (baseline)
            if hasattr(model_clone, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred = model_clone.predict_proba(X_test)
                baseline_score = scoring(y_test, y_pred)
            else:
                y_pred = model_clone.predict(X_test)
                baseline_score = scoring(y_test, y_pred)
            
            if n_jobs != 1:
                # Parallel execution over features
                permuted_scores = Parallel(n_jobs=n_jobs)(
                    delayed(get_permuted_score)(model_clone, X_test, y_test, feat, scoring) 
                    for feat in X.columns
                )
                fold_importances = baseline_score - np.array(permuted_scores)
            else:
                # Sequential execution
                fold_importances = []
                for feat_name in X.columns:
                    permuted_score = get_permuted_score(model_clone, X_test, y_test, feat_name, scoring)
                    fold_importances.append(baseline_score - permuted_score)

            for i, feat_name in enumerate(X.columns):
                all_importances[feat_name].append(fold_importances[i])
        
        # Average over folds
        mean_importances = {feat: np.mean(imp) for feat, imp in all_importances.items()}
        importance_series = pd.Series(mean_importances)
        
        # Sort in descending order
        return importance_series.sort_values(ascending=False)
    
    @staticmethod
    def single_feature_importance(X: pd.DataFrame, y: pd.Series,
                                model_factory: Callable[[], BaseEstimator],
                                cv: Optional[Union[int, TimeSeriesSplit]] = None,
                                scoring: Callable = accuracy_score,
                                n_jobs: int = 1,
                                random_state: Optional[int] = None) -> pd.Series:
        """
        Calculate Single Feature Importance (SFI).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model_factory : Callable
            Function to create a new model instance
        cv : Union[int, TimeSeriesSplit], optional
            Cross-validation strategy
        scoring : Callable, optional
            Scoring function
        n_jobs : int, optional
            Number of jobs for parallel processing
        random_state : int, optional
            Random state for reproducibility
            
        Returns:
        --------
        pd.Series
            Feature importance scores
        """
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Create cross-validation strategy if not provided
        if cv is None:
            cv = TimeSeriesSplit(n_splits=5)
        elif isinstance(cv, int):
            cv = TimeSeriesSplit(n_splits=cv)
        
        # Function to compute feature importance
        def compute_importance(idx_train, idx_test, feat_name):
            # Create and train model on single feature
            X_single = X[[feat_name]]
            model = model_factory()
            model.fit(X_single.iloc[idx_train], y.iloc[idx_train])
            
            # Score on test set
            if hasattr(model, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred = model.predict_proba(X_single.iloc[idx_test])
                score = scoring(y.iloc[idx_test], y_pred)
            else:
                y_pred = model.predict(X_single.iloc[idx_test])
                score = scoring(y.iloc[idx_test], y_pred)
            
            return feat_name, score
        
        # Store results
        all_importances = defaultdict(list)
        
        # Iterate over CV folds
        for train_idx, test_idx in cv.split(X):
            # Calculate feature importances for this fold
            if n_jobs != 1:
                # Parallel execution
                results = Parallel(n_jobs=n_jobs)(
                    delayed(compute_importance)(train_idx, test_idx, feat) 
                    for feat in X.columns
                )
                
                # Process results
                for feat_name, score in results:
                    all_importances[feat_name].append(score)
            else:
                # Sequential execution
                for feat_name in X.columns:
                    _, score = compute_importance(train_idx, test_idx, feat_name)
                    all_importances[feat_name].append(score)
        
        # Average over folds
        mean_importances = {feat: np.mean(imp) for feat, imp in all_importances.items()}
        importance_series = pd.Series(mean_importances)
        
        # Sort in descending order
        return importance_series.sort_values(ascending=False)
    
    @staticmethod
    def orthogonal_features_importance(X: pd.DataFrame, y: pd.Series,
                                     model: BaseEstimator,
                                     scoring: Callable = accuracy_score,
                                     random_state: Optional[int] = None,
                                     test_size: float = 0.3) -> pd.Series:
        """
        Calculate feature importance by constructing orthogonal features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model : BaseEstimator
            Model to use
        scoring : Callable, optional
            Scoring function
        random_state : int, optional
            Random state for reproducibility
        test_size : float, optional
            Proportion of data to use for testing
            
        Returns:
        --------
        pd.Series
            Feature importance scores
        """
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Split data
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Calculate correlation matrix
        corr_matrix = X_train.corr()
        
        # Store results
        importances = {}
        
        # Iterate over features
        for i, feat in enumerate(X.columns):
            # Create orthogonal features
            X_ortho = X_train.copy()
            
            # Orthogonalize all other features with respect to this feature
            for other_feat in X.columns:
                if other_feat != feat:
                    # Regression of other_feat on feat
                    beta = corr_matrix.loc[feat, other_feat]
                    
                    # Orthogonalize
                    X_ortho[other_feat] = X_ortho[other_feat] - beta * X_ortho[feat]
            
            # Clone the model and fit on orthogonalized features
            model_clone = clone(model)
            model_clone.fit(X_ortho, y_train)
            
            # Apply orthogonalization to test data
            X_test_ortho = X_test.copy()
            for other_feat in X.columns:
                if other_feat != feat:
                    beta = corr_matrix.loc[feat, other_feat]
                    X_test_ortho[other_feat] = X_test_ortho[other_feat] - beta * X_test_ortho[feat]
            
            # Score on test set
            if hasattr(model_clone, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred = model_clone.predict_proba(X_test_ortho)
                score = scoring(y_test, y_pred)
            else:
                y_pred = model_clone.predict(X_test_ortho)
                score = scoring(y_test, y_pred)
            
            # Store importance
            importances[feat] = score
        
        # Convert to Series
        importance_series = pd.Series(importances)
        
        # Sort in descending order
        return importance_series.sort_values(ascending=False)
    
    @staticmethod
    def importance_vs_random_probe(X: pd.DataFrame, y: pd.Series,
                                 model: BaseEstimator,
                                 importance_func: Callable,
                                 n_probes: int = 10,
                                 random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Compare feature importance against random probes to establish significance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model : BaseEstimator
            Model to use
        importance_func : Callable
            Function to calculate feature importance
        n_probes : int, optional
            Number of random probes to generate
        random_state : int, optional
            Random state for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importances and p-values
        """
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Calculate feature importance
        feature_importance = importance_func(model, X, y)
        
        # Generate random probes
        probe_importances = []
        
        for i in range(n_probes):
            # Create dataset with a random feature
            X_probe = X.copy()
            X_probe['random_probe'] = np.random.randn(len(X))
            
            # Calculate importance with random probe
            probe_importance = importance_func(model, X_probe, y)
            probe_importances.append(probe_importance['random_probe'])
        
        # Calculate statistics
        mean_probe = np.mean(probe_importances)
        std_probe = np.std(probe_importances)
        
        # Calculate p-values
        p_values = {}
        
        for feat, imp in feature_importance.items():
            # Z-score
            z = (imp - mean_probe) / std_probe
            
            # P-value (one-sided)
            p_value = 1 - norm.cdf(z)
            p_values[feat] = p_value
        
        # Create DataFrame
        results = pd.DataFrame({
            'importance': feature_importance,
            'p_value': pd.Series(p_values)
        })
        
        # Sort by importance
        return results.sort_values('importance', ascending=False)
    
    @staticmethod
    def feature_importance_clustering(X: pd.DataFrame, 
                                    importance_scores: pd.Series,
                                    n_clusters: Optional[int] = None,
                                    linkage_method: str = 'ward',
                                    distance_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Cluster features based on correlation and importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        importance_scores : pd.Series
            Feature importance scores
        n_clusters : int, optional
            Number of clusters to form
        linkage_method : str, optional
            Linkage method for hierarchical clustering
        distance_threshold : float, optional
            Distance threshold for cluster formation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature clusters and importance scores
        """
        # Calculate correlation matrix
        corr = np.abs(X.corr())
        
        # Convert to distance matrix
        distance = 1 - corr
        
        # Perform hierarchical clustering
        Z = hierarchy.linkage(hierarchy.distance.squareform(distance), method=linkage_method)
        
        # Form clusters
        if n_clusters is not None:
            clusters = hierarchy.fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            clusters = hierarchy.fcluster(Z, t=distance_threshold, criterion='distance')
        
        # Create DataFrame with results
        cluster_df = pd.DataFrame({
            'feature': X.columns,
            'cluster': clusters,
            'importance': importance_scores.reindex(X.columns)
        })
        
        # Sort by cluster and importance
        cluster_df = cluster_df.sort_values(['cluster', 'importance'], ascending=[True, False])
        
        return cluster_df
    
    @staticmethod
    def select_features_from_clusters(cluster_df: pd.DataFrame, 
                                     n_per_cluster: int = 1) -> List[str]:
        """
        Select top features from each cluster.
        
        Parameters:
        -----------
        cluster_df : pd.DataFrame
            DataFrame from feature_importance_clustering
        n_per_cluster : int, optional
            Number of features to select from each cluster
            
        Returns:
        --------
        List[str]
            List of selected features
        """
        # Group by cluster
        grouped = cluster_df.groupby('cluster')
        
        # Select top features from each cluster
        selected_features = []
        
        for cluster, group in grouped:
            # Sort by importance
            top_features = group.sort_values('importance', ascending=False)['feature'][:n_per_cluster].tolist()
            selected_features.extend(top_features)
        
        return selected_features
    
    @staticmethod
    def feature_importance_pca(X: pd.DataFrame, 
                             importance_scores: pd.Series,
                             n_components: int = 2) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Visualize feature importance in principal component space.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        importance_scores : pd.Series
            Feature importance scores
        n_components : int, optional
            Number of principal components
            
        Returns:
        --------
        Tuple[pd.DataFrame, plt.Figure]
            (DataFrame with PCA results, Matplotlib figure)
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Get component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=X.columns
        )
        
        # Add importance scores
        loadings['importance'] = importance_scores.reindex(X.columns)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot loadings
        scatter = ax.scatter(
            loadings.iloc[:, 0],
            loadings.iloc[:, 1],
            s=loadings['importance'] * 100,  # Size based on importance
            alpha=0.7,
            c=loadings['importance'],  # Color based on importance
            cmap='viridis'
        )
        
        # Add feature names as annotations
        for i, feat in enumerate(loadings.index):
            ax.annotate(
                feat,
                (loadings.iloc[i, 0], loadings.iloc[i, 1]),
                fontsize=10,
                alpha=0.75
            )
        
        # Add colorbar
        plt.colorbar(scatter, label='Feature Importance')
        
        # Plot grid lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set title and labels
        ax.set_title('Feature Importance in Principal Component Space', fontsize=15)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=12)
        
        return loadings, fig
    
    @staticmethod
    def plot_feature_importances(importances: pd.Series, title: str, 
                                 ax: Optional[plt.Axes] = None) -> None:
        """
        Plot feature importances as a bar chart.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.show() 