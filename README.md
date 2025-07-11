# Titanic Survival Prediction with Pure NumPy Logistic Regression

A self-contained, math-focused pipeline to predict Titanic survival using only NumPy. The notebook loads the dataset, handles missing ages by passenger class and sex, extracts honorific titles, computes family-size indicators, and bins age and fare into meaningful categories. Categorical features are one-hot encoded and numerical features are standardized manually. Core logistic regression components—sigmoid activation, log-loss cost, gradient computation, and gradient-descent optimization—are implemented from scratch, yielding about 83.6% training accuracy. Predictions for the test set are exported in `titanic_submission.csv`, and the learned sigmoid curve is visualized with shaded decision regions to illustrate the model’s classification boundary.

Run the notebook in JupyterLab (Python 3) to reproduce data preprocessing, model training, evaluation, submission generation, and visualization—all without high-level ML libraries.  

