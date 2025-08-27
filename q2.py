import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [2, 5.0],
    [3, 5.5],
    [4, 5.0],
    [4, 6.0],
    [5, 5.5],
    [5, 6.5],
    [6, 6.0],
    [6, 7.0],
    [7, 6.5],
    [7, 7.5],
    [8, 6.0],
    [8, 7.0],
    [9, 7.0],
    [10, 7.5]
])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,1])

model = LogisticRegression(solver='liblinear')
model.fit(X, y)

print("intercept_:", model.intercept_)
print("coef_:", model.coef_)
