import pandas as pd
from mst_clustering import MSTClustering

df_distancias = pd.read_csv("puntos_mst.csv", index_col=0)
distancias = df_distancias.values

model = MSTClustering(
    cutoff=1,  # Number of clusters
    metric="precomputed",  # Precomputed matrix of distances
    approximate=False,  # MST exact
)
labels = model.fit_predict(distancias)

resultado = pd.DataFrame({"id": df_distancias.index, "cluster": labels})

print(resultado.sort_values("cluster"))
