# Bank Marketing ML Classification

## a. Problem Statement

The goal of this project is to build and compare multiple machine learning models for the Bank Marketing classification problem.

The task is to predict whether a client will subscribe to a term deposit based on demographic, financial, and campaign-related information. This is a binary classification problem where the target column `y` indicates `yes` or `no`.

The project also provides a Streamlit web application that allows users to upload a test dataset, select a trained model, and view evaluation metrics and prediction results.

## b. Dataset Description

This project uses the Bank Marketing dataset, a well-known dataset for binary classification.

Dataset highlights:

- Source: Bank marketing campaign data
- Target variable: `y`
- Classes: `yes` and `no`
- Type of problem: Binary classification
- Input features include customer and campaign attributes such as:
  - age
  - job
  - marital status
  - education
  - balance
  - housing loan
  - personal loan
  - contact type
  - day and month of contact
  - call duration
  - campaign counts
  - previous outcomes

The test dataset used in this project contains 9,044 rows and 17 columns.

## c. GitHub Repository Link

Repository: [https://github.com/DSanjana-20/bank-marketing-ml](https://github.com/DSanjana-20/bank-marketing-ml)

The repository includes all required project files:

- `app.py`
- `Models/` with trained model files
- `requirements.txt`
- `test_data.csv`
- `README.md`

## d. Models Used

The following machine learning models were trained and evaluated on the dataset:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors
- Naive Bayes
- Random Forest Ensemble

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9013 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| kNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest (Ensemble) | 0.9045 | 0.9263 | 0.6506 | 0.3960 | 0.4924 | 0.4597 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Performs strongly overall with high accuracy and good AUC. Precision is good, but recall is low, which means it misses some positive cases. |
| Decision Tree | Produces balanced recall and F1 compared to some other models, but its AUC is the weakest among the models, showing limited class separation ability. |
| kNN | Gives decent accuracy and precision, but recall is low. It performs better than the decision tree in accuracy and AUC, but is still not the best overall. |
| Naive Bayes | Has the lowest accuracy and MCC. It achieves the highest recall among the non-ensemble models, but precision is weaker, so it produces more false positives. |
| Random Forest (Ensemble) | Best overall performer. It has the highest accuracy, AUC, precision, F1, and MCC, making it the most reliable model for this dataset. |

### Overall Winner

**Random Forest (Ensemble)** is the overall winner for this dataset because it gives the best balance of performance metrics and the strongest discrimination ability.

