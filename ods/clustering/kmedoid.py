import pandas as pd
from kmedoids import KMedoids
from sklearn.metrics.pairwise import euclidean_distances


def apply_kmedoids(data: pd.DataFrame, k: int) -> pd.DataFrame:
    distances = euclidean_distances(data)
    km = KMedoids(n_clusters=k)
    km.fit(distances)
    return km.labels_
