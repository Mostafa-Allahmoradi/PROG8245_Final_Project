import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
import sys

# --- PATH SETUP ---
curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class SpamClassifier:
    def __init__(self, X, y, X_train=None, y_train=None, X_test=None, y_test=None):
        # min_df=5: Ignore terms that appear in less than 5 documents
        # max_features=1000: Limit vocabulary to top 1000 terms
        self.vectorizer = TfidfVectorizer(min_df=5, max_features=1000)
        self.nb_classifier = MultinomialNB()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y

        self.X_train_tfidf = None
        self.X_test_tfidf = None

        self.y_pred = None

        self.cm = None

    def train(self):
        """
        Vectorizes training data and trains the Naive Bayes classifier.
        """

        # Fit the vectorizer on the training data and transform it
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        # Only transform the test data (do not re-fit)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)

        self.nb_classifier.fit(self.X_train_tfidf, self.y_train)
        
        print("Training complete.")

    def get_features(self):
        """
        Returns the feature names from the TF-IDF vectorizer.
        """
        return self.vectorizer.get_feature_names_out()


    def evaluate(self):
        """
        Evaluates the model on test data and returns metrics.
        """
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        y_pred = self.nb_classifier.predict(X_test_tfidf)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, pos_label='spam'),
            'recall': recall_score(self.y_test, y_pred, pos_label='spam'),
            'f1': f1_score(self.y_test, y_pred, pos_label='spam'),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred, labels=['ham', 'spam'])
        }
        return metrics
    
    def predict(self):
        """
        Predicts labels for the given data.
        """
        self.y_pred = self.nb_classifier.predict(self.X_test_tfidf)
        return self.y_pred
    
    def create_confusion_matrix(self):
        """
        Creates and returns the confusion matrix for the test data.
        """
        self.cm = confusion_matrix(self.y_test, self.y_pred, labels=['ham', 'spam'])
        return self.cm