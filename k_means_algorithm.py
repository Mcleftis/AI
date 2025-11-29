import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X=np.random.rand(100, 2)*10

#arithmoi clusters
kmeans= KMeans(n_clusters=3,  random_state=42)

#ekpaidefsh
kmeans.fit(X)

#provlepsh
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

#diagramma
plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,1], c='black', marker='X', s=200, label='Centroids')
plt.title("Kmeans clustering")
plt.legend()
plt.show()