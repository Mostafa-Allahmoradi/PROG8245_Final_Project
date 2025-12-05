# Data handling
from pyclbr import Class
import pandas as pd
import numpy as np

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

    def preprocess_data(self):
        """Preprocess the data by handling missing values, handling duplicate rows, and encoding categorical variables."""
        # Handle missing values
        self.data.fillna(self.data.mean(), inplace=True)

        # Handle duplicate rows
        self.data.drop_duplicates(inplace=True)

        # Encode categorical variables
        # categorical_cols = self.data.select_dtypes(include=['object']).columns
        # self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        
        print("Data preprocessing completed.")

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        print(f"Data split into training and testing sets with test size = {test_size} and random state = {random_state}.")
        
        return self.train_data, self.test_data