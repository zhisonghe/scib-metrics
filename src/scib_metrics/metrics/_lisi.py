import numpy as np
import pandas as pd

from scib_metrics.nearest_neighbors import NeighborsResults
from scib_metrics.utils import compute_simpson_index


def lisi_knn(X: NeighborsResults, labels: np.ndarray, perplexity: float = None) -> np.ndarray:
    """Compute the local inverse simpson index (LISI) for each cell :cite:p:`korsunsky2019harmony`.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
    labels
        Array of shape (n_cells,) representing label values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.

    Returns
    -------
    lisi
        Array of shape (n_cells,) with the LISI score for each cell.
    """
    labels = np.asarray(pd.Categorical(labels).codes)
    knn_dists, knn_idx = X.distances, X.indices
    row_idx = np.arange(X.n_samples)[:, np.newaxis]

    if perplexity is None:
        perplexity = np.floor(knn_idx.shape[1] / 3)

    n_labels = len(np.unique(labels))

    simpson = compute_simpson_index(
        knn_dists=knn_dists, knn_idx=knn_idx, row_idx=row_idx, labels=labels, n_labels=n_labels, perplexity=perplexity
    )
    return 1 / simpson


def ilisi_knn(X: NeighborsResults, batches: np.ndarray, perplexity: float = None, scale: bool = True) -> float:
    """Compute the integration local inverse simpson index (iLISI) for each cell :cite:p:`korsunsky2019harmony`.

    Returns a scaled version of the iLISI score for each cell, by default :cite:p:`luecken2022benchmarking`.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
    batches
        Array of shape (n_cells,) representing batch values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.
    scale
        Scale lisi into the range [0, 1]. If True, higher values are better.

    Returns
    -------
    ilisi
        iLISI score.
    """
    batches = np.asarray(pd.Categorical(batches).codes)
    lisi = lisi_knn(X, batches, perplexity=perplexity)
    ilisi = np.nanmedian(lisi)
    if scale:
        nbatches = len(np.unique(batches))
        ilisi = (ilisi - 1) / (nbatches - 1)
    return ilisi


def clisi_knn(X: NeighborsResults, labels: np.ndarray, perplexity: float = None, scale: bool = True) -> float:
    """Compute the cell-type local inverse simpson index (cLISI) for each cell :cite:p:`korsunsky2019harmony`.

    Returns a scaled version of the cLISI score for each cell, by default :cite:p:`luecken2022benchmarking`.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
    labels
        Array of shape (n_cells,) representing cell type label values
        for each cell.
    perplexity
        Parameter controlling effective neighborhood size. If None, the
        perplexity is set to the number of neighbors // 3.
    scale
        Scale lisi into the range [0, 1]. If True, higher values are better.

    Returns
    -------
    clisi
        cLISI score.
    """
    labels = np.asarray(labels)
    nan_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in labels], dtype=bool)
    if nan_mask.any():
        import warnings
        warnings.warn(
            f"Found {nan_mask.sum()} cells with NaN labels. These cells will be excluded from cLISI computation.",
            UserWarning,
        )
    labels_coded = np.asarray(pd.Categorical(labels).codes)
    lisi = lisi_knn(X, labels_coded, perplexity=perplexity)
    # Exclude NaN-labeled cells from the summary statistic
    lisi_valid = lisi[~nan_mask]
    clisi = np.nanmedian(lisi_valid)
    if scale:
        nlabels = len(np.unique(labels[~nan_mask]))
        clisi = (nlabels - clisi) / (nlabels - 1)
    return clisi
