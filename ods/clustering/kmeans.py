import pandas as pd
from sklearn.cluster import KMeans


def apply_kmeans(data: pd.DataFrame, k: int, random_state: int | None=None) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(data)
    return kmeans.labels_