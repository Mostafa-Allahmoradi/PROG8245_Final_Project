

#train logistic Regression + SVD for Classiifcation for spam
#imports
import sys
import  matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score

#Local imports
from  Data_Processing.EDA import DataProcessor
from  Models.TF_IDF_Model import  SpamClassifier


curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class SVDSpamClassifier:
    def __init__(self, x, y, x_train=None, y_train=None, x_test=None, y_test=None):
        self.count_vect = CountVectorizer(stop_words='english', min_df=5)
        self.tfidf_transformer = TfidfTransformer()

        #SVD dimension reduction
        self.svd = TruncatedSVD(n_components=100, random_state=42)

        #Classifier
        self.nb_classifier = MultinomialNB()

        #Data
        self.x = x
        self.y = y
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


        #Transformed matrix holders
        self.x_train_svd = None
        self.x_test_svd = None

        #Predictions and confusion matrix
        self.y_pred = None
        self.cm = None


    def train(self):
        #Applies vectorization
        x_train_counts = self.count_vect.fit_transform(self.x_train)

        # TF-IDF transofrm

        x_train_tfidf = self.tfidf_transformer.fit_transform(x_train_counts)

        #Dimension Reduction
        self.x_train_svd = self.svd.fit_transform(x_train_tfidf)

        #Train classifier
        self.nb_classifier.fit(self.x_train_svd, self.y_train)

        print("SVD Model Training complete")


    def evaluate(self):

            #Evaluate on test data and return metrics
            x_test_counts = self.count_vect.transform(self.x_test)
            x_test_tfidf = self.tfidf_transformer.transform(x_test_counts)
            self.x_test_svd = self.svd.transform(x_test_tfidf)
            self.y_pred = self.nb_classifier.predict(self.x_test_svd)

            metrics = {
                "accuracy": accuracy_score(self.y_test, self.y_pred),
                "precision": precision_score(self.y_test, self.y_pred, pos_label="spam"),
                "recall": recall_score(self.y_test, self.y_pred, pos_label="spam"),
                "f1": f1_score(self.y_test, self.y_pred, pos_label="spam"),
                "confusion_matrix": confusion_matrix(self.y_test, self.y_pred, labels=["ham", "spam"])
            }

            return  metrics

    def predict(self):
            #Predict using trained SVD model

            if self.x_test_svd is None:
                raise ValueError("You have to call evaluate() before predict()")

            self.y_pred = self.nb_classifier.predict(self.x_test_svd)
            return self.y_pred
    def create_matrix(self):

        #returns confusion matrix

        if self.y_pred is None:
            raise ValueError("You have to call predict() before creating confusion matrix")

        self.cm = confusion_matrix(self.y_test, self.y_pred, labels=["ham", "spam"])
        return self.cm



