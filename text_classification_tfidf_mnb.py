from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_20newsgroups 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score

# epilegmenes katigories
categories = ["alt.atheism", "rec.autos", "sci.space"]

# fortwsi dedomenwn
train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print("Arithmos training samples:", len(train_data.data))
print("Arithmos test samples:", len(test_data.data))

# vectorization
vectorizer = CountVectorizer()

# Fit sto training set (mathei to lexilogio)
X_train_counts = vectorizer.fit_transform(train_data.data)

# Transform sto test set (metatrepei ta keimena se pinaka sixnothtwn)
X_test_counts = vectorizer.transform(test_data.data)

print("Sxhma Training Matrix:", X_train_counts.shape)
print("Sxhma Test Matrix:", X_test_counts.shape)

# TF-IDF Transformer
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_test_tfidf = tfidf.transform(X_test_counts)

print("Sxhma Training TF-IDF:", X_train_tfidf.shape)
print("Sxhma Test TF-IDF:", X_test_tfidf.shape)

# Dimiourgia montelou
mnb = MultinomialNB()

# Ekpaidefsi me TF-IDF xaraktiristika
mnb.fit(X_train_tfidf, train_data.target)

# Provlepsi sto test set
y_pred = mnb.predict(X_test_tfidf)

# Axiologisi
acc = accuracy_score(test_data.target, y_pred)
print("Akribeia MultinomialNB:", acc)
