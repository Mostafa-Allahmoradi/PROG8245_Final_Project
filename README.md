# Machine Learning Programming (PROG8245) Final_Project

##  👥 Group Members
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

By completing this project, you will learn to:
    - Convert text to numerical features using TF-IDF
    - Evaluate model performance using a confusion matrix
    - Apply dimensionality reduction using:
        * SVD (Truncated SVD / LSA)
        * PCA (Principal Component Analysis)
    - Compare baseline and reduced-feature models
    - Analyze how dimensionality reduction affects accuracy and speed

## Requirements:
    - pip install -r requirements.txt

##  🎯  How to Run:

1. Clone this repo (git clone <repo-url> cd <repo-folder>)
2. Install Required Dependencies: "pip install -r requirements.txt"
3. 


## Code Explanation/Workflow:


### Final Comparison: 


### 🤝 Contributing
This is a Final Project Protocol developed for PROG8245. If any questions arise do not hesitate to contact the project member.


### References:
    1. Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.