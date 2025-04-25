import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
# Create standard datasets
blobs, _ = make_blobs(n_samples=300, centers=4, random_state=42)
moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
# Normalize datasets
scaler = StandardScaler()
blobs = scaler.fit_transform(blobs)
moons = scaler.fit_transform(moons)
# Initialize clustering models
kmeans = KMeans(n_clusters=4, random_state=42)
agglo = AgglomerativeClustering(n_clusters=4)
dbscan = DBSCAN(eps=0.3, min_samples=5)
# Fit and predict
kmeans_labels_blobs = kmeans.fit_predict(blobs)
agglo_labels_blobs = agglo.fit_predict(blobs)
dbscan_labels_blobs = dbscan.fit_predict(blobs)
kmeans_labels_moons = kmeans.fit_predict(moons)
agglo_labels_moons = agglo.fit_predict(moons)
dbscan_labels_moons = dbscan.fit_predict(moons)
# Plot results
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
ax[0, 0].scatter(blobs[:, 0], blobs[:, 1], c=kmeans_labels_blobs, cmap='viridis')
ax[0, 0].set_title('k-Means on Blobs')
ax[1, 0].scatter(blobs[:, 0], blobs[:, 1], c=agglo_labels_blobs, cmap='viridis')
ax[1, 0].set_title('Agglomerative Clustering on Blobs')
ax[2, 0].scatter(blobs[:, 0], blobs[:, 1], c=dbscan_labels_blobs, cmap='viridis')
ax[2, 0].set_title('DBSCAN on Blobs')
ax[0, 1].scatter(moons[:, 0], moons[:, 1], c=kmeans_labels_moons, cmap='viridis')
ax[0, 1].set_title('k-Means on Moons')
ax[1, 1].scatter(moons[:, 0], moons[:, 1], c=agglo_labels_moons, cmap='viridis')
ax[1, 1].set_title('Agglomerative Clustering on Moons')
ax[2, 1].scatter(moons[:, 0], moons[:, 1], c=dbscan_labels_moons, cmap='viridis')
ax[2, 1].set_title('DBSCAN on Moons')
plt.tight_layout()
plt.show()