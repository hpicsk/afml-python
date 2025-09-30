import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from itertools import combinations
from .feature_importance import FeatureImportance

class TimeSeriesFeatureImportance:
    """
    Feature importance techniques specifically for time series data.
    """
    
    @staticmethod
    def rolling_mean_decrease_accuracy(model: BaseEstimator, 
                                     X: pd.DataFrame, 
                                     y: pd.Series,
                                     window_size: int = 63,  # ~3 months of trading days
                                     step_size: int = 21,    # ~1 month of trading days
                                     test_size: int = 21,
                                     scoring: Callable = accuracy_score,
                                     n_jobs: int = 1,
                                     random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Mean Decrease Accuracy (MDA) in a rolling window fashion.
        
        Parameters:
        -----------
        model : BaseEstimator
            Model to use
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        window_size : int, optional
            Size of rolling window
        step_size : int, optional
            Step size for rolling window
        test_size : int, optional
            Size of test set within each window
        scoring : Callable, optional
            Scoring function
        n_jobs : int, optional
            Number of jobs for parallel processing
        random_state : int, optional
            Random state for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importance over time
        """
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Prepare results storage
        feature_names = X.columns
        dates = []
        importances = []
        
        # Function to compute feature importance for a window
        def compute_window_importance(start_idx):
            # Extract window
            end_idx = start_idx + window_size
            if end_idx > len(X):
                return None
                
            # Split into train and test
            X_window = X.iloc[start_idx:end_idx]
            y_window = y.iloc[start_idx:end_idx]
            
            train_idx = np.arange(len(X_window) - test_size)
            test_idx = np.arange(len(X_window) - test_size, len(X_window))
            
            # Train model
            model_clone = clone(model)
            model_clone.fit(X_window.iloc[train_idx], y_window.iloc[train_idx])
            
            # Calculate baseline score
            if hasattr(model_clone, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred = model_clone.predict_proba(X_window.iloc[test_idx])
                baseline_score = scoring(y_window.iloc[test_idx], y_pred)
            else:
                y_pred = model_clone.predict(X_window.iloc[test_idx])
                baseline_score = scoring(y_window.iloc[test_idx], y_pred)
            
            # --- Start of optimization: Permute all features at once
            X_test_window = X_window.iloc[test_idx]
            
            # Create a dictionary to hold permuted scores
            permuted_scores = {}

            # Permute each column and get score
            for feat_name in feature_names:
                X_test_permuted = X_test_window.copy()
                X_test_permuted[feat_name] = np.random.permutation(X_test_permuted[feat_name].values)
                
                if hasattr(model_clone, 'predict_proba') and scoring.__name__ == 'log_loss':
                    y_pred_permuted = model_clone.predict_proba(X_test_permuted)
                    permuted_scores[feat_name] = scoring(y_window.iloc[test_idx], y_pred_permuted)
                else:
                    y_pred_permuted = model_clone.predict(X_test_permuted)
                    permuted_scores[feat_name] = scoring(y_window.iloc[test_idx], y_pred_permuted)
            
            window_importances = [baseline_score - permuted_scores[feat] for feat in feature_names]
            # --- End of optimization

            # Get date for this window (last date in train set)
            window_date = X.index[start_idx + len(train_idx) - 1]
            
            return window_date, window_importances
        
        # Calculate importances for each window
        window_starts = range(0, len(X) - window_size + 1, step_size)
        
        if n_jobs != 1:
            # Parallel execution
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_window_importance)(start_idx) 
                for start_idx in window_starts
            )
            
            # Process results
            for result in results:
                if result is not None:
                    window_date, window_importances = result
                    dates.append(window_date)
                    importances.append(window_importances)
        else:
            # Sequential execution
            for start_idx in window_starts:
                result = compute_window_importance(start_idx)
                if result is not None:
                    window_date, window_importances = result
                    dates.append(window_date)
                    importances.append(window_importances)
        
        # Create DataFrame with results
        importances_df = pd.DataFrame(importances, index=dates, columns=feature_names)
        
        return importances_df
    
    @staticmethod
    def plot_rolling_importance(importances_df: pd.DataFrame, 
                              top_n: int = 5,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot rolling feature importance.
        
        Parameters:
        -----------
        importances_df : pd.DataFrame
            DataFrame from rolling_mean_decrease_accuracy
        top_n : int, optional
            Number of top features to plot
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Calculate average importance
        avg_importance = importances_df.mean()
        
        # Get top features
        top_features = avg_importance.nlargest(top_n).index.tolist()
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot rolling importance for top features
        for feature in top_features:
            axes[0].plot(importances_df.index, importances_df[feature], 
                       label=feature, linewidth=2, alpha=0.8)
        
        # Add legend, grid, and labels
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Rolling Feature Importance (Top Features)', fontsize=15)
        axes[0].set_ylabel('Importance', fontsize=12)
        
        # Plot heatmap of all features
        sorted_features = avg_importance.sort_values(ascending=False).index
        im = axes[1].imshow(
            importances_df[sorted_features].T, 
            aspect='auto', 
            cmap='viridis'
        )
        
        # Set ticks for heatmap
        axes[1].set_yticks(np.arange(len(sorted_features)))
        axes[1].set_yticklabels(sorted_features)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], label='Importance')
        
        # Set title and labels for heatmap
        axes[1].set_title('Feature Importance Heatmap (All Features)', fontsize=12)
        axes[1].set_xlabel('Time', fontsize=10)
        
        # Set x-axis to show dates
        n_dates = len(importances_df.index)
        if n_dates <= 10:
            step = 1
        else:
            step = n_dates // 10
            
        date_indices = np.arange(0, n_dates, step)
        axes[1].set_xticks(date_indices)
        axes[1].set_xticklabels([importances_df.index[i].strftime('%Y-%m-%d') for i in date_indices], 
                              rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def temporal_stability_score(importances_df: pd.DataFrame) -> pd.Series:
        """
        Calculate temporal stability score for each feature.
        
        Parameters:
        -----------
        importances_df : pd.DataFrame
            DataFrame from rolling_mean_decrease_accuracy
            
        Returns:
        --------
        pd.Series
            Stability scores (higher is more stable)
        """
        # Calculate statistics
        mean_importance = importances_df.mean()
        std_importance = importances_df.std()
        
        # Calculate coefficient of variation (lower means more stable)
        cv = std_importance / mean_importance
        
        # Convert to stability score (higher is more stable)
        stability = 1 / (1 + cv)
        
        # Sort in descending order
        return stability.sort_values(ascending=False)
    
    @staticmethod
    def plot_feature_stability(importances_df: pd.DataFrame, 
                             top_n: int = 10) -> plt.Figure:
        """
        Plot feature importance stability.
        
        Parameters:
        -----------
        importances_df : pd.DataFrame
            DataFrame from rolling_mean_decrease_accuracy
        top_n : int, optional
            Number of top features to plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Calculate stability scores
        stability = TimeSeriesFeatureImportance.temporal_stability_score(importances_df)
        
        # Get top stable features
        top_stable = stability.nlargest(top_n)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot stability scores
        top_stable.plot(kind='bar', ax=axes[0], color='teal', alpha=0.7)
        axes[0].set_title('Top Stable Features', fontsize=15)
        axes[0].set_ylabel('Stability Score', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot importance distribution with box plots
        importances_df[top_stable.index].boxplot(ax=axes[1], vert=False, grid=False)
        axes[1].set_title('Feature Importance Distribution', fontsize=15)
        axes[1].set_xlabel('Importance', fontsize=12)
        
        # Add mean markers
        means = importances_df[top_stable.index].mean()
        axes[1].scatter(means, np.arange(1, len(means) + 1), 
                      color='red', marker='o', s=80, alpha=0.7, label='Mean')
        axes[1].legend()
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def feature_correlation_analysis(importances_df: pd.DataFrame, 
                                   lag: int = 1) -> pd.DataFrame:
        """
        Analyze correlation between feature importances over time.
        
        Parameters:
        -----------
        importances_df : pd.DataFrame
            DataFrame from rolling_mean_decrease_accuracy
        lag : int, optional
            Lag for autocorrelation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with correlation metrics
        """
        # Calculate autocorrelation
        autocorr = {}
        for feature in importances_df.columns:
            series = importances_df[feature].dropna()
            if len(series) > lag:
                autocorr[feature] = series.autocorr(lag=lag)
            else:
                autocorr[feature] = np.nan
        
        # Calculate correlation with other features
        cross_corr = {}
        for feature in importances_df.columns:
            # Calculate mean correlation with all other features
            corrs = []
            for other_feature in importances_df.columns:
                if feature != other_feature:
                    corr = importances_df[feature].corr(importances_df[other_feature])
                    corrs.append(corr)
            
            cross_corr[feature] = np.mean(corrs)
        
        # Calculate trend
        trend = {}
        for feature in importances_df.columns:
            # Use simple linear regression to detect trend
            x = np.arange(len(importances_df[feature]))
            y = importances_df[feature].values
            
            # Handle NaNs
            mask = ~np.isnan(y)
            if np.sum(mask) > 2:  # Need at least 3 points for regression
                x_clean = x[mask]
                y_clean = y[mask]
                
                # Calculate slope
                slope, _, _, _, _ = np.polyfit(x_clean, y_clean, 1, full=True)
                trend[feature] = slope[0]
            else:
                trend[feature] = np.nan
        
        # Combine results
        results = pd.DataFrame({
            'autocorrelation': pd.Series(autocorr),
            'cross_correlation': pd.Series(cross_corr),
            'trend': pd.Series(trend),
            'mean_importance': importances_df.mean(),
            'std_importance': importances_df.std()
        })
        
        # Sort by mean importance
        return results.sort_values('mean_importance', ascending=False)


class FeatureInteractionImportance:
    """
    Methods to analyze feature interactions and their importance.
    """
    
    @staticmethod
    def pairwise_feature_importance(model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                   scoring: Callable = accuracy_score,
                                   n_jobs: int = 1,
                                   random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate importance of pairwise feature interactions.
        
        Parameters:
        -----------
        model : BaseEstimator
            Trained model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        scoring : Callable, optional
            Scoring function
        n_jobs : int, optional
            Number of jobs for parallel processing
        random_state : int, optional
            Random state for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with pairwise feature importance
        """
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get individual feature importances first
        mda_individual = FeatureImportance.get_mda_feature_importances(model, X, y, scoring=scoring, n_jobs=n_jobs)

        # Get feature pairs
        feature_pairs = list(combinations(X.columns, 2))
        
        # Baseline score
        if hasattr(model, 'predict_proba') and scoring.__name__ == 'log_loss':
            y_pred = model.predict_proba(X)
            baseline_score = scoring(y, y_pred)
        else:
            y_pred = model.predict(X)
            baseline_score = scoring(y, y_pred)
        
        # Function to compute interaction importance
        def compute_interaction(pair):
            feat1, feat2 = pair
            
            # Permute both features
            X_permuted = X.copy()
            X_permuted[feat1] = np.random.permutation(X_permuted[feat1].values)
            X_permuted[feat2] = np.random.permutation(X_permuted[feat2].values)
            
            # Score after permutation
            if hasattr(model, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred_permuted = model.predict_proba(X_permuted)
                permuted_score = scoring(y, y_pred_permuted)
            else:
                y_pred_permuted = model.predict(X_permuted)
                permuted_score = scoring(y, y_pred_permuted)
            
            # Calculate importance (decrease in score)
            joint_importance = baseline_score - permuted_score
            
            # Interaction effect is the additional importance from the pair
            interaction_importance = joint_importance - mda_individual[feat1] - mda_individual[feat2]
            
            return pair, interaction_importance
        
        # Calculate interaction importance
        if n_jobs != 1:
            # Parallel execution
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_interaction)(pair) 
                for pair in feature_pairs
            )
        else:
            # Sequential execution
            results = [compute_interaction(pair) for pair in feature_pairs]
        
        # Process results
        interaction_importance = {}
        for pair, importance in results:
            feat1, feat2 = pair
            interaction_importance[(feat1, feat2)] = importance
        
        # Convert to DataFrame
        pairs = list(interaction_importance.keys())
        values = list(interaction_importance.values())
        
        interaction_df = pd.DataFrame({
            'feature1': [p[0] for p in pairs],
            'feature2': [p[1] for p in pairs],
            'importance': values
        })
        
        # Sort by importance
        return interaction_df.sort_values('importance', ascending=False)
    
    @staticmethod
    def visualize_interaction_network(interaction_df: pd.DataFrame, 
                                    top_n: int = 20,
                                    figsize: Tuple[int, int] = (14, 12)) -> None:
        """
        Visualize feature interactions as a network.
        
        Parameters:
        -----------
        interaction_df : pd.DataFrame
            DataFrame from pairwise_feature_importance
        top_n : int, optional
            Number of top interactions to visualize
        figsize : Tuple[int, int], optional
            Figure size
        """
        try:
            import networkx as nx
        except ImportError:
            print("Please install networkx: pip install networkx")
            return
        
        # Get top interactions
        top_interactions = interaction_df.nlargest(top_n, 'importance')
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        all_features = set(top_interactions['feature1']).union(set(top_interactions['feature2']))
        for feature in all_features:
            G.add_node(feature)
        
        # Add edges with weights
        for _, row in top_interactions.iterrows():
            G.add_edge(
                row['feature1'], 
                row['feature2'], 
                weight=row['importance']
            )
        
        # Calculate node size based on importance
        node_size = {}
        for feature in all_features:
            # Sum of importance of all interactions involving this feature
            size = top_interactions[
                (top_interactions['feature1'] == feature) | 
                (top_interactions['feature2'] == feature)
            ]['importance'].sum()
            node_size[feature] = size
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Set positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]  # Scale for visualization
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='lightblue')
        
        # Draw nodes
        node_sizes = [node_size[node] * 1000 for node in G.nodes()]  # Scale for visualization
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title('Feature Interaction Network', fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_individual_vs_interaction(X: pd.DataFrame, y: pd.Series,
                                        model_factory: Callable[[], BaseEstimator],
                                        cv: int = 5,
                                        scoring: Callable = accuracy_score,
                                        n_jobs: int = 1,
                                        random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Compare individual feature performance vs. feature pairs.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model_factory : Callable
            Function to create a new model instance
        cv : int, optional
            Number of cross-validation folds
        scoring : Callable, optional
            Scoring function
        n_jobs : int, optional
            Number of jobs for parallel processing
        random_state : int, optional
            Random state for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with comparison results
        """
        from sklearn.model_selection import TimeSeriesSplit
        from tqdm import tqdm

        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Create cross-validation strategy
        cv_splitter = TimeSeriesSplit(n_splits=cv)
        
        # Get feature pairs
        features = list(X.columns)
        feature_pairs = list(combinations(features, 2))
        
        # Store results
        results = []
        
        # Function to evaluate model on a subset of features
        def evaluate_feature_set(feature_list, train_idx, test_idx):
            # Create and train model
            model = model_factory()
            model.fit(X.iloc[train_idx][feature_list], y.iloc[train_idx])
            
            # Score on test set
            if hasattr(model, 'predict_proba') and scoring.__name__ == 'log_loss':
                y_pred = model.predict_proba(X.iloc[test_idx][feature_list])
                score = scoring(y.iloc[test_idx], y_pred)
            else:
                y_pred = model.predict(X.iloc[test_idx][feature_list])
                score = scoring(y.iloc[test_idx], y_pred)
                
            return score
        
        # Evaluate individual features
        individual_scores = {}
        
        for feature in tqdm(features, desc="Evaluating individual features"):
            # Cross-validation
            cv_scores = []
            
            for train_idx, test_idx in cv_splitter.split(X):
                score = evaluate_feature_set([feature], train_idx, test_idx)
                cv_scores.append(score)
            
            # Average score
            individual_scores[feature] = np.mean(cv_scores)
        
        # Evaluate feature pairs
        for pair in tqdm(feature_pairs, desc="Evaluating feature pairs"):
            feat1, feat2 = pair
            
            # Cross-validation
            cv_scores = []
            
            for train_idx, test_idx in cv_splitter.split(X):
                score = evaluate_feature_set(list(pair), train_idx, test_idx)
                cv_scores.append(score)
            
            # Average score
            pair_score = np.mean(cv_scores)
            
            # Calculate synergy (improvement over individual features)
            individual_max = max(individual_scores[feat1], individual_scores[feat2])
            synergy = pair_score - individual_max
            
            # Store results
            results.append({
                'feature1': feat1,
                'feature2': feat2,
                'individual1_score': individual_scores[feat1],
                'individual2_score': individual_scores[feat2],
                'pair_score': pair_score,
                'synergy': synergy
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by synergy
        return results_df.sort_values('synergy', ascending=False) 