# Set of points
set I;          # Total points

# Parameters
param d {i in I, j in I};    # Distance from point i to medoid j
param k >= 0;                # Number of clusters

# Variables
var y{j in I} binary;        # 1 if point j is selected as a medoid, else 0
var x{i in I, j in I} binary;# 1 if point i is assigned to medoid j, else 0

# Objective function: minimize the sum of distances
minimize clustering:
    sum {i in I, j in I} x[i,j] * d[i,j];

# Constraints

# Each point must be assigned to exactly one medoid
subject to OneAssignment {i in I}:
    sum {j in I} x[i,j] = 1;

# Exactly k medoids
subject to Kclusters:
    sum {j in I} y[j] = k;

# A point can only be assigned to an active medoid
subject to ClusterExists {i in I, j in I}:
    x[i,j] <= y[j];