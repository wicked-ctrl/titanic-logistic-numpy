# Titanic Survival Prediction with Pure NumPy Logistic Regression

A reproducible, end-to-end implementation of logistic regression built entirely with NumPy. This project demonstrates how to preprocess real-world data, engineer features, and optimize a classification model from first principles—without relying on high-level machine learning libraries.

## Methodology

The pipeline begins by loading raw CSV files into pandas DataFrames and inspecting missing values and basic distributions. Missing ages are imputed based on passenger class and sex, while cabin entries are consolidated into deck identifiers. Honorific titles are extracted from names, and family relationships are encoded via family‐size and “is alone” indicators. Continuous variables (age, fare, family size) are standardized manually, and categorical attributes are one‐hot encoded. Core algorithmic components—sigmoid activation, log-loss cost, gradient computation, and gradient-descent optimization—are implemented from scratch in NumPy. Model convergence is monitored through cost‐versus‐iteration diagnostics, yielding approximately 83.6% training accuracy. Final predictions for the test set are thresholded at 0.5 and exported for submission. An overlay of the learned sigmoid function with shaded decision regions provides a visual confirmation of the classification boundary.

## Provided Data

- `train.csv` — Training features and survival labels  
- `test.csv` — Passenger features for prediction  
- `gender_submission.csv` — Example submission format  

Clone this repository and execute the Jupyter notebook in a Python 3 environment to reproduce the entire workflow, from data preparation through to submission and visualization.  
