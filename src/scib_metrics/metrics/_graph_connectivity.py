import warnings

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from scib_metrics.nearest_neighbors import NeighborsResults


def graph_connectivity(X: NeighborsResults, labels: np.ndarray) -> float:
    """Quantify the connectivity of the subgraph per cell type label.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values
        representing distances to exactly each cell's k nearest neighbors.
    labels
        Array of shape (n_cells,) representing label values
        for each cell.
    """
    # TODO(adamgayoso): Utils for validating inputs
    labels = np.asarray(labels)
    nan_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in labels], dtype=bool)
    if nan_mask.any():
        warnings.warn(
            f"Found {nan_mask.sum()} cells with NaN labels. These cells will be excluded from graph connectivity computation.",
            UserWarning,
        )
        valid = ~nan_mask
        labels = labels[valid]
        graph = X.knn_graph_distances[valid][:, valid]
    else:
        graph = X.knn_graph_distances

    clust_res = []

    for label in np.unique(labels):
        mask = np.asarray(labels == label)
        if hasattr(mask, "values"):
            mask = mask.values
        graph_sub = graph[mask]
        graph_sub = graph_sub[:, mask]
        _, comps = connected_components(graph_sub, connection="strong")
        tab = pd.value_counts(comps)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)
