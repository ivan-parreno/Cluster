import pandas as pd
from mst_clustering import MSTClustering
import numpy as np
from typing import List, Tuple

def run_mst_clustering(data: pd.DataFrame, cutoff: int, approximate=False):
    model = MSTClustering(
        cutoff=cutoff,
        metric="euclidean",
        approximate=approximate,
    )

    labels = model.fit_predict(data.values)

    return labels

def run_mst_clustering_heuristic(
    data: pd.DataFrame,
    cutoff: int,
    approximate: bool = False,
    max_iter: int = 100,
    min_cluster_size: int = 10,
) -> Tuple[pd.Series, List[int], pd.DataFrame, int, int]:
    original_index = data.index.to_numpy()
    current_idx = original_index.copy()
    current_df = data.loc[current_idx].copy()

    n_original = len(original_index)
    labels_full = np.full(n_original, -1, dtype=int)
    removed_indices: List[int] = []

    it = 0
    current_cutoff = int(cutoff)

    while it < max_iter and len(current_df) > 0:
        it += 1
        labels = run_mst_clustering(
            current_df,
            cutoff=current_cutoff,
            approximate=approximate,
        )
        labels = np.asarray(labels)

        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))

        small_labels = unique[(unique != -1) & (counts < min_cluster_size)]
        outlier_present = -1 in unique

        labels_to_remove = list(small_labels)
        if outlier_present:
            labels_to_remove.append(-1)

        if len(labels_to_remove) > 0:
            mask_remove = np.isin(labels, labels_to_remove)
            positions_to_remove = np.nonzero(mask_remove)[0]
            orig_inds_to_remove = current_idx[positions_to_remove].tolist()
            removed_indices.extend(orig_inds_to_remove)

            keep_mask = ~mask_remove
            current_idx = current_idx[keep_mask]
            current_df = current_df.loc[current_idx].copy()
            if len(current_df) == 0:
                break
            continue

        valid_clusters = [l for l, c in label_counts.items() if l != -1 and c >= min_cluster_size]
        n_valid = len(valid_clusters)

        if n_valid == cutoff + 1:
            for pos, orig_idx in enumerate(current_idx):
                labels_full[np.where(original_index == orig_idx)[0][0]] = int(labels[pos])
            break

        current_cutoff += 1

    if len(current_df) > 0:
        try:
            final_labels = run_mst_clustering(
                current_df,
                cutoff=current_cutoff,
                approximate=approximate,
            )
            for pos, orig_idx in enumerate(current_idx):
                labels_full[np.where(original_index == orig_idx)[0][0]] = int(final_labels[pos])
        except Exception:
            pass

    removed_indices = sorted(set(removed_indices))

    labels_series = pd.Series(labels_full, index=original_index, name="label")

    return labels_series