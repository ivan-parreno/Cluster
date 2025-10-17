import pandas as pd
from mst_clustering import MSTClustering


def run_mst_clustering(data: pd.DataFrame, cutoff: int, approximate=False):
    model = MSTClustering(
        cutoff=cutoff,
        metric="euclidean",
        approximate=False,
    )

    labels = model.fit_predict(data.values)

    return labels
