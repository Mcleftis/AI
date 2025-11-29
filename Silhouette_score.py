import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import  Kmeans
from sklearn.metrics import silhouette_score

X=np.random.rand(100, 2)

kmeans=Kmeans(n_clusters=3, random_state=42)

kmeans.fit(X)

labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

score=silhouette_score(X, labels)
print("Silhouette Score:", score)

for k in range(2, 7):
    kmeans=KMeans(n_clusters=k, random_state=42)
    labels=kmeans.fit_predict(X)
    score=silhouette_score(X, labels)
    print(f"K={k}, Silhouette Score={score:.3f}")