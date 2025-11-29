import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import  MeanShift

X=np.random.rand(100,2)*10

ms=MeanShift()

ms.fit(X)

labels=ms.labels_
cluster_centers=ms.cluster_centers_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='black', marker='X', s=200, label='Centers')
plt.title("Mean Shift Clustering")
plt.legend()
plt.show()