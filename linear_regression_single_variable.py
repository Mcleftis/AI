from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#fortwsh dataset k spasimo metavlhtn
X=np.array([50, 60, 70, 80, 90, 100]).reshape(-1,1)
y=np.array([150, 180, 200, 220, 250, 280])

#split
X_train, X_test, y_train, y_test=train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)

#dhmiourgia montelou
model=LinearRegression()
model.fit(X_train, y_train)

#problepsi
y_pred=model.predict(X_test)

#evaluation
print("MSE:",mean_squared_error(y_test, y_pred))
print("R^2:\n", r2_score(y_test, y_pred))

#diagramma
plt.scatter(X_test, y_test, color="blue", label="Real data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Prediction line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression (Single Variable)")
plt.legend()
plt.show()