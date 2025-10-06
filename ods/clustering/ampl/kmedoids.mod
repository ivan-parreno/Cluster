# Conjunto de puntos
set I;          # Total points
set J := I;     # Possible medoids (all points)

# Parámetros
param d {i in I, j in J};    # Distance from point i to medoid j
param k >= 0;   # Number of clusters

# Variables
var y{j in J} binary;        # 1 if point j is selected as a medoid
var x{i in J,j in J} binary;      # 1 if point i is assigned to medoid j

# Función objetivo: minimizar la suma de distancias
minimize clustering:
    sum {i in I, j in J} x[i,j] * d[i,j];

# Restricciones

# Cada punto se asigna a exactamente un medoid
subject to OneAssignment {i in I}:
    sum {j in J} x[i,j] = 1;

# Exactamente k medoids
subject to Kclusters:
    sum {j in J} y[j] = k;

# Un punto solo puede asignarse a un medoid activo
subject to ClusterExists {i in I, j in J}:
    x[i,j] <= y[j];
