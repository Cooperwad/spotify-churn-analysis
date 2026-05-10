from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_churn_model_matrix.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_churn_cleaned.csv"
KMEANS_DATA_PATH = PROJECT_ROOT / "models" / "spotify_churn_kmeans.csv"


def load_model_matrix() -> pd.DataFrame:
    """Load the model_matrix of the Spotify churn dataset."""
    if not MODEL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"model matrix not found at: {MODEL_DATA_PATH}\n"
            "run python src/clean_data.py first"
        )
    return pd.read_csv(MODEL_DATA_PATH)

def load_clean_data() -> pd.DataFrame:
    """Load the clean Spotify churn dataset."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"model matrix not found at: {CLEAN_DATA_PATH}\n"
            "run python src/clean_data.py first"
        )

    return pd.read_csv(CLEAN_DATA_PATH)

def main() -> None:
    #Import data
    data = load_clean_data()
    matrix = load_model_matrix()
    print("Data imported")

    #Initialize and fit the model
    kmeans = KMeans()
    kmeans.fit(matrix.drop(columns=['user_id', 'is_churned']))
    print("kmeans initialized")

    #Get the cluster assignments
    data['k-means cluster'] = kmeans.labels_
    data = data.sort_values(by='k-means cluster')
    print(data)

    #Save the data
    data.to_csv(KMEANS_DATA_PATH, index=False)
    print(f"\nSaved cleaned data to: {KMEANS_DATA_PATH}")

   #Generate a comparison between churn across clusters
    tally = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    for row in data.iterrows():
        cluster = row[1]['k-means cluster']
        tally[cluster][0] += 1
        if row[1]['is_churned']:
           tally[cluster][1] += 1
    
    for cluster in tally:
        cluster[2] = cluster[1]/cluster[0]
    print(tally)


main()