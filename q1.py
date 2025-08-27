import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

X = np.array([5,12,18,22,28,35,42,48,55,63,72,85]).reshape(-1, 1)
y = np.array([0,0,0,0,0,0,0,1,1,1,1,1])

model = LogisticRegression(solver='liblinear')
model.fit(X, y)

print("intercept_:", model.intercept_)
print("coef_:", model.coef_)

b0 = model.intercept_[0]
b1 = model.coef_[0][0]
p = 0.70
x_boundary = (np.log(p/(1-p)) - b0) / b1
print("Decision boundary for p=0.70:", x_boundary)

X_test = np.array([16, 27, 33, 49, 67, 90]).reshape(-1, 1)
probs = model.predict_proba(X_test)[:,1]
print("Predicted probabilities:", probs)



b0 = model.intercept_[0]
b1, b2 = model.coef_[0]
print(f"Regression equation: logit(p) = {b0:.4f} + {b1:.4f}*x1 + {b2:.4f}*x2")

x_test = np.array([[6.5, 7.5]])
prob = model.predict_proba(x_test)[0,1]
print("Predicted probability for [6.5, 7.5]:", prob)