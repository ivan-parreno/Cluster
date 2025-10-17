from amplpy import AMPL
import pandas as pd
import numpy as np


def compute_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[0]
    d_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d_matrix[i, j] = np.linalg.norm(df.iloc[i] - df.iloc[j])
    return d_matrix


def write_ampl_dat(d_matrix: np.ndarray, k: int, path: str = "./ampl/kmedoids.dat") -> None:
    n = d_matrix.shape[0]
    with open(path, "w") as f:
        f.write(f"set I := {' '.join(map(str, range(1, n + 1)))};\n\n")
        f.write(f"param k := {k};\n\n")
        f.write("param d : " + " ".join(map(str, range(1, n + 1))) + " :=\n")
        for i in range(n):
            row = " ".join(f"{d_matrix[i, j]:.4f}" for j in range(n))
            f.write(f"{i + 1} {row}\n")
        f.write(";\n")


def run_ampl_and_get_assignments(
    mod_path: str = "./ampl/kmedoids.mod", dat_path: str = "./ampl/kmedoids.dat", solver: str = "cplex"
) -> pd.DataFrame:
    ampl = AMPL()
    ampl.reset()

    ampl.read(mod_path)

    ampl.read_data(dat_path)

    ampl.option["solver"] = solver
    ampl.solve()

    assignments = []
    for i in ampl.set["I"]:
        for j in ampl.set["I"]:
            val = ampl.var["x"][i, j].value()
            if val is not None and float(val) > 0.5:
                assignments.append({"punto": int(i), "AMPL": int(j)})
                break
    return pd.DataFrame(assignments)
