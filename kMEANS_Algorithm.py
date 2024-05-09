

import numpy as np
import matplotlib.pyplot as plt


X = np.loadtxt("jain_feats.txt")
centroid_old = np.loadtxt("jain_centers.txt")
N, _ = X.shape
K = centroid_old.shape[0]


centroid_new = np.zeros_like(centroid_old)


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='b', s=7)
plt.scatter(centroid_old[:, 0], centroid_old[:, 1], c='r', s=70, marker='*')
plt.title("Initial Centroids")
plt.show()


label = np.zeros(N, dtype=np.int32)
for e in range(100):

    for i in range(N):
        dist = np.linalg.norm(X[i] - centroid_old, axis=1)
        label[i] = np.argmin(dist)


    for j in range(K):
        centroid_new[j] = np.mean(X[label == j], axis=0)


    diff = np.linalg.norm(centroid_new - centroid_old, axis=1)
    if np.max(diff) < 1e-7:
        break

    centroid_old = centroid_new.copy()


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=label, s=7)
plt.scatter(centroid_old[:, 0], centroid_old[:, 1], c='r', s=70, marker='*')
plt.title("Final Centroids")
plt.show()