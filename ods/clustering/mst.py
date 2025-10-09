import pandas as pd
from mst_clustering import MSTClustering

df = pd.read_csv("puntos_mst.csv")
X = df[['x', 'y']].values

model = MSTClustering(cutoff_scale=1.5, min_cluster_size=2)

labels = model.fit_predict(X)
df['cluster'] = labels

print(df.sort_values('cluster'))

