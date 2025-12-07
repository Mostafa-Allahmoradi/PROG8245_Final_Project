"""
PCA-Reduced Logistic Regression Model
======================================
Author: Oluwafemi Lawal (Olawal7308@conestogac.on.ca)
Student ID: 8967308

This module implements Logistic Regression with PCA-reduced features
for text classification as part of the PROG8245 Final Project.

Step 6 & 7: Dimensionality Reduction with PCA and Model Training
"""

# Data handling
import numpy as np
import pandas as pd

# Preprocessing and Feature Extraction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Model
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


class PCAReducedModel:
    """
    A class to perform PCA dimensionality reduction and train a Logistic Regression classifier.
    
    This model standardizes TF-IDF features (required for PCA), reduces dimensionality using
    Principal Component Analysis, and trains a Logistic Regression classifier on the reduced features.
    
    Attributes:
        n_components (int): Number of PCA components to retain.
        random_state (int): Random seed for reproducibility.
        scaler (StandardScaler): Scaler for standardizing features.
        pca (PCA): PCA transformer.
        model (LogisticRegression): Logistic Regression classifier.
    """
    
    def __init__(self, n_components=100, random_state=42):
        """
        Initialize the PCA-Reduced Logistic Regression model.
        
        Args:
            n_components (int): Number of principal components to retain. Default is 100.
            random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
        self.pca = None
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        
        # Store results
        self.train_features_pca = None
        self.test_features_pca = None
        self.predictions = None
        self.metrics = {}
        
    def fit_transform_pca(self, X_train, X_test):
        """
        Standardize and apply PCA transformation to the features.
        
        PCA requires standardized data (zero mean, unit variance) for optimal performance.
        Since TF-IDF matrices are sparse, we use StandardScaler with with_mean=False.
        
        Args:
            X_train: Training TF-IDF features (sparse or dense matrix).
            X_test: Test TF-IDF features (sparse or dense matrix).
            
        Returns:
            tuple: (X_train_pca, X_test_pca) - PCA-transformed features.
        """
        # Convert sparse matrices to dense (required for PCA)
        if hasattr(X_train, 'toarray'):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = np.array(X_train)
            
        if hasattr(X_test, 'toarray'):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = np.array(X_test)
        
        # Standardize the data (required for PCA)
        print("Standardizing TF-IDF features...")
        X_train_scaled = self.scaler.fit_transform(X_train_dense)
        X_test_scaled = self.scaler.transform(X_test_dense)
        
        # Determine number of components (cannot exceed min of samples and features)
        max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
        actual_components = min(self.n_components, max_components)
        
        if actual_components < self.n_components:
            print(f"Note: Reducing n_components from {self.n_components} to {actual_components} "
                  f"(limited by data dimensions)")
        
        # Apply PCA
        print(f"Applying PCA with {actual_components} components...")
        self.pca = PCA(n_components=actual_components, random_state=self.random_state)
        
        self.train_features_pca = self.pca.fit_transform(X_train_scaled)
        self.test_features_pca = self.pca.transform(X_test_scaled)
        
        print(f"Original shape: {X_train_dense.shape}")
        print(f"PCA-reduced shape: {self.train_features_pca.shape}")
        print(f"Total explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return self.train_features_pca, self.test_features_pca
    
    def get_explained_variance(self):
        """
        Get the explained variance ratio for each component.
        
        Returns:
            tuple: (individual variance ratios, cumulative variance ratios)
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Call fit_transform_pca first.")
            
        individual = self.pca.explained_variance_ratio_
        cumulative = np.cumsum(individual)
        
        return individual, cumulative
    
    def train(self, X_train_pca=None, y_train=None):
        """
        Train the Logistic Regression model on PCA-reduced features.
        
        Args:
            X_train_pca: PCA-transformed training features. If None, uses stored features.
            y_train: Training labels.
        """
        if X_train_pca is None:
            X_train_pca = self.train_features_pca
            
        if X_train_pca is None:
            raise ValueError("No training features available. Call fit_transform_pca first.")
            
        print("Training Logistic Regression on PCA-reduced features...")
        self.model.fit(X_train_pca, y_train)
        print("Training complete!")
        
    def predict(self, X_test_pca=None):
        """
        Make predictions on test data.
        
        Args:
            X_test_pca: PCA-transformed test features. If None, uses stored features.
            
        Returns:
            np.array: Predicted labels.
        """
        if X_test_pca is None:
            X_test_pca = self.test_features_pca
            
        if X_test_pca is None:
            raise ValueError("No test features available. Call fit_transform_pca first.")
            
        self.predictions = self.model.predict(X_test_pca)
        return self.predictions
    
    def evaluate(self, y_true, y_pred=None):
        """
        Evaluate model performance and compute all metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels. If None, uses stored predictions.
            
        Returns:
            dict: Dictionary containing all evaluation metrics.
        """
        if y_pred is None:
            y_pred = self.predictions
            
        if y_pred is None:
            raise ValueError("No predictions available. Call predict first.")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute metrics
        self.metrics = {
            'confusion_matrix': cm,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label='spam', average='binary'),
            'recall': recall_score(y_true, y_pred, pos_label='spam', average='binary'),
            'f1_score': f1_score(y_true, y_pred, pos_label='spam', average='binary'),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        return self.metrics
    
    def get_confusion_matrix_details(self, y_true, y_pred=None, labels=None):
        """
        Get detailed confusion matrix breakdown with interpretations.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: Class labels (e.g., ['ham', 'spam']).
            
        Returns:
            dict: Detailed breakdown of confusion matrix components.
        """
        if y_pred is None:
            y_pred = self.predictions
            
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # For binary classification
        tn, fp, fn, tp = cm.ravel()
        
        details = {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'total_correct': tn + tp,
            'total_incorrect': fp + fn,
            'total_samples': cm.sum()
        }
        
        # Add interpretations for spam detection
        details['interpretation'] = {
            'correctly_identified_ham': tn,
            'correctly_identified_spam': tp,
            'ham_misclassified_as_spam': fp,
            'spam_misclassified_as_ham': fn
        }
        
        return details
    
    def print_evaluation_summary(self, y_true):
        """
        Print a comprehensive evaluation summary.
        
        Args:
            y_true: True labels.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Call predict first.")
            
        metrics = self.evaluate(y_true)
        
        print("\n" + "="*60)
        print("PCA-REDUCED LOGISTIC REGRESSION MODEL EVALUATION")
        print("="*60)
        print(f"\nModel Configuration:")
        print(f"  - PCA Components: {self.pca.n_components_ if self.pca else 'N/A'}")
        print(f"  - Explained Variance: {self.pca.explained_variance_ratio_.sum():.4f}" if self.pca else "")
        print(f"\nPerformance Metrics:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        print("="*60)
