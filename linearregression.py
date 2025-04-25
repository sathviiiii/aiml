import numpy as np

def gradient_descent(X, y, alpha=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta -= alpha * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
    return theta, cost_history
# Sample data
X = np.array([[1, 1], [1, 2], [1, 3]])  # Including bias term (1)
y = np.array([1, 2, 3])                # Target values

# Run gradient descent
theta, cost_history = gradient_descent(X, y)

# Print results
print("Learned parameters (theta):", theta)
print("Final cost:", cost_history[-1])


''' LEAST SQUARES '''

import numpy as np

def least_squares(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Sample data
X = np.array([[1, 1], [1, 2], [1, 3]])  # with bias term
y = np.array([1, 2, 3])

# Compute theta using least squares
theta_ls = least_squares(X, y)

# Print the result
print("Parameters (theta) from least squares:", theta_ls)


''' POLYNOMIAL REGRESSION '''

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    theta = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    return theta, poly

# Sample data
X = np.array([[1], [2], [3]])
y = np.array([1, 4, 9])  # Perfect square relation

# Train polynomial regression of degree 2
theta, poly = polynomial_regression(X, y, degree=2)

# Predict for new data
X_new = np.array([[4]])
X_new_poly = poly.transform(X_new)
y_pred = X_new_poly.dot(theta)

print("Learned theta:", theta)
print("Prediction for X=4:", y_pred[0])


''' LASSO REGRESSION '''

from sklearn.linear_model import Lasso

def lasso_regression(X, y, alpha=1.0):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    return lasso.coef_

import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([1, 2, 3, 4])

# Scaling is often important for Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run Lasso regression
coefficients = lasso_regression(X_scaled, y, alpha=0.1)

print("Lasso coefficients:", coefficients)


''' RIDGE REGRESSION '''

from sklearn.linear_model import Ridge

def ridge_regression(X, y, alpha=1.0):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    return ridge.coef_

import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([1, 2, 3, 4])

# Scale the data (optional but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run Ridge regression
coefficients = ridge_regression(X_scaled, y, alpha=0.1)

print("Ridge coefficients:", coefficients)

'''
Method          Advantages          Disadvantages
Use Cases

Gradient Descent

1. Efficient for large datasets, no need for matrix inversion.
2. Requires tuning of learning rate, may converge slowly.
3. Large datasets, online learning.

Least Squares

1. Simple, direct computation.
2. Infeasible for large datasets, sensitive to multicollinearity.
3. Small datasets, baseline model.

Polynomial Regression

1. Captures non-linear relationships.
2. Prone to overfitting, complex model interpretation.
3. When relationship between variables is non-linear.

LASSO Regression

1. Performs feature selection, interpretable model.
2. May introduce bias, depends heavily on regularization term.
3. Sparse models, feature selection.

Ridge Regression

1. Reduces overfitting, handles multicollinearity well.
2. Does not perform feature selection.
3. When multicollinearity is a concern.'''