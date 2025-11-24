from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

#fortwsh dataset
data = load_breast_cancer()

# diaxwrismos se xarakthristika x kai etiketes y
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.head())   #deixnei tis prwtes grammes twn xarakthristikwn
print(y.value_counts())  #posa deigmata einai kaloithi kai posa kakoithi

# Mean removal / Standardization

#Mesos Oros, Typikh Apoklish
print("Prin to mean removal:")
print("Mesos oros:", X.mean(axis=0).values)
print("Typikh apoklish:", X.std(axis=0).values)

#Standardization,Mean Removal
X_standardized = preprocessing.scale(X)

#Meta to scaling
print("\nMeta to Mean Removal:")
print("Mesos oros:", X_standardized.mean(axis=0))
print("Typikh apoklish:", X_standardized.std(axis=0))

#prosarmosmena dedomena
print("\nProsarmosmena Dedomena:")
print(X_standardized)

#Spasimo set se training kai test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4,   
    random_state=42  #stathero seed gia anaparagwgimothta
)

print("Μέγεθος Training Set:", X_train.shape)
print("Μέγεθος Test Set:", X_test.shape)

#dhmiourgia montelou
gnb = GaussianNB()

#ekpaidefsh montelou
model = gnb.fit(X_train, y_train)

print("To montelo ekpaidefthke epityxws!")

#provlepsh sto test set
y_pred = model.predict(X_test)

#ypologismos akriveias
accuracy = accuracy_score(y_test, y_pred)
print("Akriveia montelou:", accuracy)

