# Sonar Data Classification Project

This project focuses on building a logistic regression model to classify sonar signals into two categories. Using the **Sonar Dataset**, the model predicts whether the sonar signal was bounced off a rock or a metal cylinder.

## Project Overview

This project utilizes Python and its machine learning libraries to preprocess, train, and evaluate a logistic regression model. It includes essential steps such as data preprocessing, feature scaling, and model validation to ensure robust performance.

### Key Features

- **Dataset Loading**: The dataset used is the Sonar Dataset (stored as `Copy of sonar data.csv`), which contains 60 features and a target column.
- **Data Preprocessing**: Includes feature scaling with `StandardScaler` and splitting the dataset into training and testing sets.
- **Model Training**: A logistic regression model is trained on the dataset.
- **Model Evaluation**: Evaluation metrics include accuracy scores for both training and testing data, along with cross-validation for robustness.
- **Prediction**: The project supports predicting the class of new sonar data inputs.

---

### Technologies Used

- **Languages**: Python
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical computations
  - `sklearn` for machine learning and data preprocessing
