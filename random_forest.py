from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#fortwsh dataset k spasimo metavlhtn
data=load_breast_cancer()

X=pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target)
print(X.head())
print(y.head())

feature_names=data.feature_names
class_names=data.target_names

#split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.4,
    random_state=42
)

#train
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

#predict
y_pred=model.predict(X_test)

#evaluate 
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#interpret
importances=model.feature_importances_
indices=np.argsort(importances)

#sxediasmos grafhmatos
plt.figure(figsize=(10,8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)),[feature_names[i] for i in indices])
plt.xlabel("Simantikothta xarakthristikou")
plt.title("Feature Importances(Decision Tree, max_depth=3)")
plt.show()