from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.DataFrame({
    "square_meters": [50, 60, 70, 80, 90, 100, 120, 150],
    "rooms": [2, 3, 3, 4, 4, 5, 5, 6],
    "age": [30, 25, 20, 15, 10, 8, 5, 2],
    "price": [150, 180, 200, 220, 250, 280, 320, 400]
})

X=data[["square_meters", "rooms", "age"]]
y=data["price"]

X_train, X_test, y_train, y_test=train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)

model=LinearRegression()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print("RSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(y_test, y_pred, color="green")
plt.xlabel("Real prices")
plt.ylabel("Predicted prices")
plt.title("Multi-variable Linear Regression")
plt.show()
