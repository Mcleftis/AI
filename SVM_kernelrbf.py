from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


data=load_breast_cancer()
X=pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target)

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)



print("Megethos trainiing set:", X_train.shape)
print("Megethos test set:", X_test.shape)
gnb=GaussianNB()

model=gnb.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

#scaling gia rbf
scaler = preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#SVM

svm_rbf=SVC(kernel='rbf', C=1.0, gamma='scale')

svm_rbf.fit(X_train_scaled, y_train)

y_pred=svm_rbf.predict(X_test_scaled)

accuracy=accuracy_score(y_test, y_pred)
print("Accuracy SVM (Scaled):", accuracy)
#confusion matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)