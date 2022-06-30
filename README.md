# Sucide-prediction-using-linear-regression
Our model follows Supervised Learning, which consists in learning the link between two datasets: the observed data X and an external variable y that we are trying to predict, usually called “target” or “labels”. Most often, y is a 1D array of length n_samples.

All supervised estimators in scikit-learn implement a fit(X, y) method to fit the model and a predict(X) method that, given unlabeled observations X, returns the predicted labels y.

While assigning values to X, we drop some columns which we do not require or which are less relevant to our model while predicting the output.

#Dataset
In our problem, the data that should be feeded for the machine to decide and predict effectively has to be measure of variability in depressive symptoms along with other relevant factors such as younger age, mood disorders, childhood abuse, and personal and parental history of suicide attempts, etc.

#Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output).
