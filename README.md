# titanic-logistic-numpy
This notebook implements a pure-NumPy logistic regression for Titanic survival. It loads data, imputes ages by class/sex, extracts titles, computes family size, and bins age/fare. Categorical features are one-hot encoded; numerics are standardized. Sigmoid, log-loss, gradient, and gradient descent are hand-coded, achieving ~83.6% accuracy.
