import pandas as pd
import numpy as np
import string

import nltk
from nltk.corpus import stopwords
# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Train-test split
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.train_data = None
        self.test_data = None

    def load_data(self, sep='\t', header=None, names=['label', 'message']):
        """Load data from a CSV file."""
        self.data = pd.read_csv(self.data_path, sep=sep, header=header, names=names)
        print("Data loaded successfully.")

    def preprocess_text(self, message):
        """
        1. Removes punctuation
        2. Converts to lowercase
        3. Removes stopwords (common words like 'the', 'is', 'and')
        """
        # Remove Punctuation
        no_punc = [char for char in message if char not in string.punctuation]
        no_punc = ''.join(no_punc)
        
        # Convert to lower case and split into words
        words = no_punc.lower().split()
        
        # Remove Stopwords
        stop_words = set(stopwords.words('english'))
        clean_words = [word for word in words if word not in stop_words]
        
        # Join back into a string (optional, depending on next steps)
        return " ".join(clean_words)
    
    def feature_engineering(self):
        """Apply text preprocessing to the entire dataset."""

        # Apply text preprocessing
        self.data['cleaned_message'] = self.data['message'].apply(self.preprocess_text)

        # Add a feature for 'message length' to see if it varies by class
        self.data['message_len'] = self.data['message'].apply(len)

    def data_balancing(self):
        """Check and handle class imbalance if necessary."""
        
        ham_indices = self.data[self.data['label'] == 'ham'].index
        self.data = self.data.drop(ham_indices[:3572])  # Drop first 3572 ham messages
          