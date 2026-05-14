# Hierarchical Clustering Notes

## --- Methodology ---
- Used Aggregative Clustering with Ward's Linkage metric to generate a total of 7 clusters
    - Compared silhouette scores from models ranging from 2-24 clusters, and choosing the model with the most clusters while maintaining reasonable silhouette.
    - Settled on 7 clusters after noticing a sharp jump in silhouette between 7 and 8 clusters.

## --- Results ---
- Silhouette score of 0.048, suggesting high overlap of clusters in feature space.
- Analysis of churn rates within each cluster stayed very consistent, only varying between 23.98% and 27.1%, showing no great risk of churn between clusters.
- Cluster 0 contained exclusively free users, while clusters 1-5 contained only subscribed users, with only cluster 6 featuring both.
- User demographics such as gender, age, and country showed little importance for determining clustering
- 'Listening time' and 'songs played per day varied largely between clusters, suggesting further study may be warranted.
- Cluster 4 seemed to feature unusually low skip rates, while featuring the highest mean mobile users and lowest mean listening time among clusters containing premium users, possibly warranting further study.