from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#Fortosi dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.head())
print(y.head())  # deixnei tis protes times

feature_names = data.feature_names
class_names = data.target_names

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)

#Rixo Dentro (Shallow Model)
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)
y_pred_shallow = dt_shallow.predict(X_test)
acc_shallow = accuracy_score(y_test, y_pred_shallow)

#Vathy Dentro (Deep Model)
dt_deep = DecisionTreeClassifier(random_state=42)  
dt_deep.fit(X_train, y_train)
y_pred_deep = dt_deep.predict(X_test)
acc_deep = accuracy_score(y_test, y_pred_deep)


print("Accuracy Shallow Tree (max_depth=3):", acc_shallow)
print("Confusion Matrix Shallow:\n", confusion_matrix(y_test, y_pred_shallow))

print("\nAccuracy Deep Tree (xoris orio vathous):", acc_deep)
print("Confusion Matrix Deep:\n", confusion_matrix(y_test, y_pred_deep))

#Simantikotita Xaraktiristikwn
importances = dt_shallow.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,8))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Simantikotita Xaraktiristikou")
plt.title("Feature Importances (Decision Tree, max_depth=3)")
plt.show()

#Optikopoihsh Dentrou
plt.figure(figsize=(20,10))
plot_tree(dt_shallow, 
          feature_names=feature_names, 
          class_names=class_names, 
          filled=True, 
          rounded=True)
plt.show()
