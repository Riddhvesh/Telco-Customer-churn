# Telco-Customer-churn



## Project overview

This project builds a machine learning model to predict which telecom customers are likely to stop using the service (churn).  
It follows the complete ML workflow from data loading to model evaluation using the popular Telco Customer Churn dataset.  
The goal is to identify at-risk customers so the company can take retention actions.

## Dataset source

Telco Customer Churn dataset from Kaggle: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn  ](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
The dataset contains 7043 customer records with demographic info, services used, account details, and a binary Churn label.

## Steps to run the project

1. Clone the repository and ensure you have Python 3.8+ with required libraries.
2. Install dependencies: `pip install pandas numpy scikit-learn`
3. Place the dataset file `Telco-Customer-Churn-2.csv` in the same folder as the notebook.
4. Open `churn_prediction.ipynb` in Jupyter Notebook or Google Colab.
5. Run all cells sequentially. The model will train and show evaluation results.

## Model used

Logistic Regression implemented through a scikit-learn Pipeline that includes:
- ColumnTransformer for handling categorical (OneHotEncoder) and numeric features
- Logistic Regression classifier (max_iter=1000 for convergence)

## Final result summary

Test accuracy: 79.49%  
Confusion Matrix: [[918 117], [172 202]]  
F1-score: 0.86 for non-churn, 0.58 for churn  
The model performs well on non-churners but has room for improvement on identifying churners (lower recall).
