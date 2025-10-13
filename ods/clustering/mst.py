import pandas as pd
from mst_clustering import MSTClustering

def run_mst_clustering(csv_path, cutoff=1, approximate=False):
    
    # Load distance matrix
    df_distances = pd.read_csv(csv_path, index_col=0)
    distances = df_distances.values

    # Build MST clustering model
    model = MSTClustering(
        cutoff=cutoff,
        metric="precomputed",  # use distance matrix directly
        approximate=approximate
    )

    # Fit and predict clusters
    labels = model.fit_predict(distances)

    # Return results
    result = pd.DataFrame({
        "id": df_distances.index,
        "cluster": labels
    }).sort_values("cluster").reset_index(drop=True)

    return result

resultado = run_mst_clustering("puntos_mst.csv", cutoff=1)
print(resultado)
