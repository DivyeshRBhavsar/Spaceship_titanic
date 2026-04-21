# Spaceship Titanic - Machine Learning Project

## Overview

This project is based on the Spaceship Titanic dataset, where the goal is to predict whether a passenger was transported to an alternate dimension after a spaceship collision.

The project involves data cleaning, feature engineering, exploratory data analysis (EDA), and model building to achieve accurate predictions.

## Objective

Predict the target variable:
Transported (True/False)

## Current Performance
✅ Latest Accuracy: 80.209%

🔄 Status: Work in Progress (Improving performance further)

## Approach
### Data Preprocessing
Handled missing values using:
Group-based imputation (e.g., using Group column)
Mode/median filling strategies
Cleaned and transformed categorical features
Created meaningful derived features

### Feature Engineering

Some important engineered features:
Group extracted from PassengerId
Spending-related features (e.g., total spend, no spend flags)
CryoSleep-related transformations
Deck and Cabin decomposition

### Exploratory Data Analysis (EDA)

Checked distributions of key variables
Analyzed correlation between features and target
Identified patterns such as:
CryoSleep passengers more likely transported
Spending behavior impact

### Model Building
Models explored:

Logistic Regression
Random Forest
Gradient Boosting (or similar ensemble models)
Xgboost
Catboost

Current best model achieves(Catboost):
Accuracy: 80.209%

### Tech Stack
Python 🐍
Pandas
NumPy
Scikit-learn
Matplotlib / Seaborn (for visualization)

### Current Limitations
Model accuracy can still be improved
Feature engineering can be expanded further
Hyperparameter tuning not fully optimized


### Contribution
This project is currently under active development. Suggestions and improvements are welcome!