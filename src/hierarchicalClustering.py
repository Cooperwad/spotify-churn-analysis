import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

#Load the model matrix and clean data
data = pd.read_csv("data\processed\spotify_churn_model_matrix.csv")

user_ids = data["user_id"]
is_churned = data["is_churned"]

data = data.drop(columns=["user_id", "is_churned"])
data.head()

# Perform hierarchical clustering
linked_euclidean = linkage(data, method='ward', metric='euclidean')

plt.figure(figsize=(12, 6))
dendrogram(linked_euclidean, truncate_mode='lastp', p=30)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()

# Fit basic Agglomerative Clustering model
agg_clustering_basic = AgglomerativeClustering(linkage='ward')
cluster_labels = agg_clustering_basic.fit_predict(data)
pd.Series(cluster_labels).value_counts()

# Evaluate silhouette scores for different numbers of clusters
n_clusters = []
silhouette_scores = []

for n in range(2, 25):
    model = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)

    n_clusters.append(n)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(n_clusters)
plt.show()

# Create the final model with the chosen number of clusters
model = AgglomerativeClustering(n_clusters=7, linkage='ward')
new_cluster_labels = model.fit_predict(data)

# Add the new cluster labels, churn status, and user IDs back to the original cleaned DataFrame (not the model matrix)
data_cleaned = pd.read_csv("data\processed\spotify_churn_cleaned.csv")
data_cleaned["cluster"] = new_cluster_labels

# Analyze the distribution of churned users across clusters
churn_distribution = data_cleaned.groupby("cluster")["is_churned"].value_counts().unstack()
churn_distribution.plot(kind="bar", stacked=False, figsize=(10, 6))
plt.title("Distribution of Churned Users Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Churned", labels=["No", "Yes"])
plt.show()

# Analyse the churn rate of each cluster
churn_rate = data_cleaned.groupby("cluster")["is_churned"].mean()
churn_rate.plot(kind="bar", figsize=(10, 6))
plt.title("Churn Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Churn Rate")
plt.xticks(rotation=0)
plt.show()

# Explore the distribution of other features across clusters to understand what differentiates them
CATEGORICAL_COLS = [
    "gender",
    "country",
    "subscription_type",
    "device_type",
]
NUMERIC_COLS = [
    "age",
    "listening_time",
    "songs_played_per_day",
    "skip_rate",
    "ads_listened_per_week",
]
BINARY_COLS = [
    "offline_listening",
]

stats_num = data_cleaned.groupby("cluster")[NUMERIC_COLS].mean()
stats_cat = data_cleaned.groupby("cluster")[CATEGORICAL_COLS].describe()
stats_binary = data_cleaned.groupby("cluster")[BINARY_COLS].value_counts().unstack()

stats_num
stats_cat
stats_binary

# Visualize the distribution of device types across clusters
device_stats = data_cleaned.groupby("cluster")["device_type"].value_counts().unstack()
device_stats.plot(kind="bar", stacked=False, figsize=(10, 6))
plt.title("Device Type Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Device Type")
plt.show()

# Visualize the distribution of subscription types across clusters
subscription_stats = data_cleaned.groupby("cluster")["subscription_type"].value_counts().unstack()
subscription_stats.plot(kind="bar", stacked=False, figsize=(10, 6))
plt.title("Subscription Type Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Subscription Type")
plt.show()

# Visualize the average numeric features across clusters
num_stats = data_cleaned.groupby("cluster")[NUMERIC_COLS].mean()
num_stats.plot(kind="bar", figsize=(12, 8))
plt.show()