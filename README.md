# Machine Learning Programming (PROG8245) Final_Project

## 👥 Group Members

    1. Jarius Bedward - 8841640
    2. Mostafa Allahmoradi - 9087818
    3. Oluwafemi Lawal - 8967308

## 📌 Project Overview

This project explores text classification using multiple machine learning techniques combined with dimensionality reduction. You will convert raw text into numerical features, apply TF-IDF, reduce dimensionality using SVD and PCA, and evaluate model performance through metrics and visualizations.
The goal is to understand how feature representation and dimensionality reduction impact classification accuracy and computational efficiency.

## Project Setup:

## Database Description

    - Type: Text-based dataset
    - Task: Binary classification (e.g., Spam vs Ham, Positive vs Negative)
    - Labels: Supervised learning
    - Size: ~2,000 documents/reviews/emails
    - Split: 75% Training, 25% Testing
    - Reference/Citation: (Almeida & Hidalgo, 2011)

## Model Architecture Summary

| Model       | Features    | Algorithm           | Purpose                       |
| ----------- | ----------- | ------------------- | ----------------------------- |
| **Model 1** | TF-IDF      | Naive Bayes         | Baseline with sparse features |
| **Model 2** | SVD-Reduced | Logistic Regression | Dense semantic features       |
| **Model 3** | PCA-Reduced | Logistic Regression | Compare PCA vs SVD            |

🎯 Learning Objectives

By completing this project, you will learn to: - Convert text to numerical features using TF-IDF - Evaluate model performance using a confusion matrix - Apply dimensionality reduction using:
_ SVD (Truncated SVD / LSA)
_ PCA (Principal Component Analysis) - Compare baseline and reduced-feature models - Analyze how dimensionality reduction affects accuracy and speed

## Requirements:

    - pip install -r requirements.txt

## 🎯 How to Run:

1. **Clone this repository:**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Install Required Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebooks in Order:**

   Navigate to the `Notebooks/` directory and run the notebooks in the following order:

   | Order | Notebook                 | Description                             | Author              |
   | ----- | ------------------------ | --------------------------------------- | ------------------- |
   | 1     | `TF_IDF_Demo.ipynb`      | Baseline: Naive Bayes with TF-IDF       | Mostafa Allahmoradi |
   | 2     | `SVD-Reduced_Demo.ipynb` | Logistic Regression with SVD            | Jarius Bedward      |
   | 3     | `PCA-Reduced_Demo.ipynb` | Logistic Regression with PCA            | Oluwafemi Lawal     |
   | 4     | `Model_Comparison.ipynb` | Step 8: Visual comparison of all models | Oluwafemi Lawal     |

4. **View Generated Visualizations:**

   After running the comparison notebook, visualizations will be saved in `Visualization/`:

   - `all_confusion_matrices.png` - Side-by-side confusion matrices
   - `metrics_comparison.png` - Performance metrics bar chart
   - `pca_vs_svd_variance.png` - Variance comparison
   - `dimensionality_impact.png` - Feature reduction impact

## 📁 Project Structure:

```
PROG8245_Final_Project/
├── data/
│   └── raw/
│       └── SMSSpamCollection    # SMS Spam dataset
├── Data_Processing/
│   └── EDA.py                   # Data loading and preprocessing
├── Models/
│   ├── TF_IDF_Model.py         # TF-IDF + Naive Bayes model
│   ├── SVD_Reduced_Model.py    # SVD + Logistic Regression model
│   └── PCA_Reduced_Model.py    # PCA + Logistic Regression model
├── Notebooks/
│   ├── TF_IDF_Demo.ipynb       # Step 3: Baseline model (Mostafa)
│   ├── SVD-Reduced_Demo.ipynb  # Steps 4-5: SVD reduction (Jarius)
│   ├── PCA-Reduced_Demo.ipynb  # Steps 6-7: PCA reductio(Oluwafemi)
│   └── Model_Comparison.ipynb  # Step 8: Visual comparison (Oluwafemi)
├── Visualization/
│   └── Visualizer.py           # Visualization utilities
├── requirements.txt
├── instructions.md
└── README.md
```

## Code Explanation/Workflow:

### Step 1-2: Data Loading and TF-IDF Feature Extraction

- Load SMS Spam Collection dataset
- Apply TF-IDF vectorization to convert text to numerical features

### Step 3: Baseline Model (Mostafa Allahmoradi)

- Train Naive Bayes classifier on TF-IDF features
- Generate confusion matrix and calculate metrics

### Steps 4-5: SVD Dimensionality Reduction (Jarius Bedward)

- Apply Truncated SVD to reduce TF-IDF dimensions
- Train Logistic Regression on SVD-reduced features
- Compare with baseline model

### Steps 6-7: PCA Dimensionality Reduction (Oluwafemi Lawal)

- Standardize TF-IDF features
- Apply PCA to reduce dimensions
- Train Logistic Regression on PCA-reduced features
- Compare with both previous models

### Step 8: Visual Comparison (Oluwafemi Lawal)

- Display all three confusion matrices side-by-side
- Compare performance metrics across all models
- Analyze which dimensionality reduction works better for text

### Final Comparison:

The `Model_Comparison.ipynb` notebook provides:

- Side-by-side confusion matrices for all three models
- Performance metrics comparison chart
- PCA vs SVD variance curves
- Dimensionality reduction impact analysis
- Comprehensive written analysis

### 🤝 Contributing

This is a Final Project Protocol developed for PROG8245. If any questions arise do not hesitate to contact the project member.

### References:

    1. Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
