import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/spotify_churn_cleaned.csv')

X = df[['listening_time', 'skip_rate']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute KNN to find eps for DBSCAN
'''neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)

dists, indices = neighbors_fit.kneighbors(X_scaled)
dists = np.sort(dists[:, 4])

plt.plot(dists)
plt.title('k-distances Graph')
plt.show()'''

# DBSCAN
db = DBSCAN(eps=0.075, min_samples=15)
hdb = HDBSCAN(min_cluster_size=50, min_samples=15)

db_labels = db.fit_predict(X_scaled)
hdb_labels = hdb.fit_predict(X_scaled)

labels = hdb_labels # change to the model you're testing

df = df.loc[X.index]
df['Cluster'] = labels

cluster_summary = df.groupby('Cluster')['is_churned'].agg(['mean', 'count'])
print(cluster_summary)

cluster_profiles = df.groupby("Cluster")[['listening_time', 'skip_rate']].mean()
print(cluster_profiles)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow', s=20)
plt.title('HDBSCAN Clustering')
plt.xlabel('Listening Time')
plt.ylabel('Skip Rate')
plt.show()