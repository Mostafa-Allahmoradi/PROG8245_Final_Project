"""
Visualization Utilities for Text Classification Project
========================================================
This module provides visualization utilities for:
- Confusion matrices
- PCA/SVD variance plots
- Model comparison charts
- Side-by-side confusion matrix comparisons (Step 8)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualizer:
    """
    A class providing visualization utilities for the text classification project.
    """
    
    def __init__(self, figsize=(8, 6), style='whitegrid'):
        """
        Initialize the Visualizer.
        
        Args:
            figsize (tuple): Default figure size for plots.
            style (str): Seaborn style to use.
        """
        self.figsize = figsize
        sns.set_style(style)
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title='Confusion Matrix',
                               cmap='Blues', normalize=False, ax=None, annot_fontsize=14):
        """
        Create and visualize a confusion matrix heatmap.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: Class labels (e.g., ['ham', 'spam']).
            title: Plot title.
            cmap: Color map for the heatmap.
            normalize: If True, normalize the confusion matrix.
            ax: Matplotlib axis to plot on. If None, creates new figure.
            annot_fontsize: Font size for annotations.
            
        Returns:
            matplotlib.axes.Axes: The plot axis.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    xticklabels=labels if labels else 'auto',
                    yticklabels=labels if labels else 'auto',
                    annot_kws={'size': annot_fontsize},
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return ax
    
    def plot_three_confusion_matrices(self, results_dict, labels=None, 
                                       figsize=(16, 5), save_path=None):
        """
        Plot all three confusion matrices side-by-side (Step 8: Visual Comparison).
        
        Args:
            results_dict: Dictionary with model names as keys and (y_true, y_pred) tuples as values.
                         Expected format: {'Model Name': (y_true, y_pred), ...}
            labels: Class labels (e.g., ['ham', 'spam']).
            figsize: Figure size for the combined plot.
            save_path: If provided, saves the figure to this path.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        colors = ['Blues', 'Greens', 'Oranges']
        
        for idx, (model_name, (y_true, y_pred)) in enumerate(results_dict.items()):
            self.plot_confusion_matrix(
                y_true, y_pred, 
                labels=labels,
                title=f'{model_name}\nConfusion Matrix',
                cmap=colors[idx],
                ax=axes[idx]
            )
        
        plt.suptitle('Comparison of All Three Models - Confusion Matrices', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_variance_explained(self, explained_variance_ratio, title='Explained Variance by Components',
                                 n_components_to_show=None, ax=None):
        """
        Plot explained variance ratio for PCA/SVD components.
        
        Args:
            explained_variance_ratio: Array of variance ratios from PCA/SVD.
            title: Plot title.
            n_components_to_show: Number of components to display. If None, shows all.
            ax: Matplotlib axis to plot on.
            
        Returns:
            matplotlib.axes.Axes: The plot axis.
        """
        if n_components_to_show:
            variance = explained_variance_ratio[:n_components_to_show]
        else:
            variance = explained_variance_ratio
            
        cumulative = np.cumsum(variance)
        components = range(1, len(variance) + 1)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Individual variance
        ax.bar(components, variance, alpha=0.6, label='Individual Variance', color='steelblue')
        
        # Cumulative variance
        ax.plot(components, cumulative, 'r-o', label='Cumulative Variance', markersize=4)
        
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for total variance
        ax.annotate(f'Total: {cumulative[-1]:.2%}', 
                    xy=(len(variance), cumulative[-1]),
                    xytext=(len(variance)*0.8, cumulative[-1]*0.9),
                    fontsize=10, fontweight='bold')
        
        return ax
    
    def plot_pca_vs_svd_variance(self, pca_variance, svd_variance, 
                                   n_components_to_show=50, save_path=None):
        """
        Compare PCA and SVD variance curves side by side.
        
        Args:
            pca_variance: Explained variance ratio from PCA.
            svd_variance: Explained variance ratio from SVD.
            n_components_to_show: Number of components to display.
            save_path: If provided, saves the figure to this path.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # PCA variance
        self.plot_variance_explained(
            pca_variance, 
            title='PCA - Explained Variance',
            n_components_to_show=n_components_to_show,
            ax=axes[0]
        )
        
        # SVD variance
        self.plot_variance_explained(
            svd_variance,
            title='SVD - Explained Variance', 
            n_components_to_show=n_components_to_show,
            ax=axes[1]
        )
        
        plt.suptitle('PCA vs SVD Variance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Create a bar chart comparing metrics across all models.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metric dictionaries as values.
                         Expected format: {'Model Name': {'accuracy': 0.95, 'precision': 0.90, ...}, ...}
            save_path: If provided, saves the figure to this path.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['steelblue', 'forestgreen', 'darkorange']
        
        for idx, model in enumerate(models):
            values = [metrics_dict[model].get(m, 0) for m in metrics]
            bars = ax.bar(x + idx * width, values, width, label=model, color=colors[idx])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def create_metrics_table(self, metrics_dict):
        """
        Create a formatted metrics comparison table.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metric dictionaries as values.
            
        Returns:
            pandas.DataFrame: Formatted comparison table.
        """
        import pandas as pd
        
        metrics_to_include = ['accuracy', 'precision', 'recall', 'f1_score']
        
        data = {}
        for model_name, metrics in metrics_dict.items():
            data[model_name] = {m: metrics.get(m, 'N/A') for m in metrics_to_include}
        
        df = pd.DataFrame(data).T
        df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        df.index.name = 'Model'
        
        # Format as percentages
        for col in df.columns:
            df[col] = df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
        
        return df
    
    def plot_dimensionality_reduction_impact(self, original_dims, reduced_dims_dict, 
                                              accuracy_dict, save_path=None):
        """
        Visualize the impact of dimensionality reduction on feature count and accuracy.
        
        Args:
            original_dims: Original number of TF-IDF features.
            reduced_dims_dict: Dictionary of {method: reduced_dimensions}.
            accuracy_dict: Dictionary of {method: accuracy}.
            save_path: If provided, saves the figure to this path.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Dimensionality comparison
        methods = ['TF-IDF\n(Original)'] + [f'{m}\n(Reduced)' for m in reduced_dims_dict.keys()]
        dimensions = [original_dims] + list(reduced_dims_dict.values())
        colors = ['steelblue'] + ['forestgreen', 'darkorange'][:len(reduced_dims_dict)]
        
        bars = axes[0].bar(methods, dimensions, color=colors)
        axes[0].set_ylabel('Number of Features', fontsize=12)
        axes[0].set_title('Feature Dimensionality Comparison', fontsize=14, fontweight='bold')
        
        for bar, dim in zip(bars, dimensions):
            axes[0].annotate(f'{dim:,}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Accuracy comparison
        methods_acc = list(accuracy_dict.keys())
        accuracies = list(accuracy_dict.values())
        
        bars = axes[1].bar(methods_acc, accuracies, color=['steelblue', 'forestgreen', 'darkorange'])
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        
        for bar, acc in zip(bars, accuracies):
            axes[1].annotate(f'{acc:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Impact of Dimensionality Reduction', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_distribution_comparison(self, data, feature, label_col='label', 
                                  labels=['ham', 'spam'], bins=30, save_path=None):
        """
        Plot distribution comparison of a feature across different classes.
        
        Args:
            data: DataFrame containing the data.
            feature: Feature column name to plot.
            label_col: Column name for class labels.
            labels: List of class labels to compare.
            bins: Number of bins for the histogram.
            save_path: If provided, saves the figure to this path.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for label in labels:
            subset = data[data[label_col] == label]
            sns.histplot(subset[feature], bins=bins, kde=True, label=label, ax=ax, stat='density', alpha=0.6)
        
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()} by Class', fontsize=14, fontweight='bold')
        ax.legend(title='Class')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_histogram(self, data, feature, bins=30, title='Histogram', xlabel=None, ylabel='Frequency', 
                   color='steelblue', save_path=None):
        """
        Plot a histogram for a given feature.
        """
        # Use subplots to create fig and ax explicitly
        fig, ax = plt.subplots(figsize=(6, 4))
        
        sns.histplot(x=feature, data=data, bins=bins, color=color, ax=ax)
        
        # Set labels on the axes object
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_heatmap(self, pivot_table, x_label, y_label, title='Heatmap', cmap='YlGnBu', save_path=None):
        """
        Plot a heatmap for given x and y axes with values.
        """
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', cbar=True, fmt='.2f', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_idf_distribution(self, idf_values, title='Distribution of IDF Weights', 
                              xlabel='IDF Weight', ylabel='Frequency', bins=30, save_path=None):
        """
        Plot the distribution of IDF weights.
        
        Args:
            idf_values: Array-like of IDF weights.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            bins: Number of bins for the histogram.
            save_path: If provided, saves the figure to this path.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """

        fig, ax = plt.subplots(figsize=(11, 8))
        
        sns.histplot(idf_values, bins=bins, kde=True, color='steelblue', ax=ax)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig