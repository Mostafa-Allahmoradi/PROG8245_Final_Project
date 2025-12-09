from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

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
        # max_features=3000: Limit vocabulary to top 3000 terms
        self.vectorizer = TfidfVectorizer(min_df=5, max_features=3000)
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
    
    def show_top_spam_words(self, n=10):
        # 1. Access the classifier directly from the object
        classifier = self.nb_classifier
        
        # 2. Use the helper method YOU defined in your class to get feature names
        # (Instead of calling .vectorizer.get_feature_names_out())
        feature_names = self.get_features()
        
        # 3. Get log probabilities of features given 'spam' class
        # Note: We must check where 'spam' is in the classes_ list
        spam_class_index = list(classifier.classes_).index('spam')
        spam_prob = classifier.feature_log_prob_[spam_class_index]
        
        # 4. Sort indices by probability
        top_indices = sorted(range(len(spam_prob)), key=lambda i: spam_prob[i], reverse=True)[:n]
        
        print(f"\nTop {n} indicators of SPAM:")
        for i in top_indices:
            print(f"- {feature_names[i]}")

    def analyze_errors(self):
    # Create a DataFrame to match messages with their true/pred labels
        results = pd.DataFrame({
            'Message': self.X_test,
            'Actual': self.y_test,
            'Predicted': self.y_pred
        })
        
        # 1. False Positives: Safe messages (Ham) blocked as Spam
        fp = results[(results['Actual'] == 'ham') & (results['Predicted'] == 'spam')]
        
        print(f"\n--- False Positives (Ham labeled as Spam): {len(fp)} found ---")
        print("These are dangerous because the user might miss important messages!")
        for i, row in fp.head(3).iterrows():
            print(f" > \"{row['Message']}\"")
            
        # 2. False Negatives: Spam messages that slipped into the Inbox
        fn = results[(results['Actual'] == 'spam') & (results['Predicted'] == 'ham')]
        
        print(f"\n--- False Negatives (Spam labeled as Ham): {len(fn)} found ---")
        print("These are annoying, but usually less critical than blocking real mail.")
        for i, row in fn.head(3).iterrows():
            print(f" > \"{row['Message']}\"")


    def classification_report(self):
        """
        Generates a classification report.
        """
        return classification_report(self.y_test, self.y_pred, target_names=['ham', 'spam'])