import logging
import random
import warnings
from typing import Literal

import igraph
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import check_array

from scib_metrics.nearest_neighbors import NeighborsResults
from scib_metrics.utils import KMeans

logger = logging.getLogger(__name__)


def _nan_mask(labels: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True where labels are NaN/None."""
    try:
        return np.isnan(labels.astype(float))
    except (ValueError, TypeError):
        # Object arrays (strings, etc.): check for None / float NaN entries
        return np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in labels], dtype=bool)


def _compute_clustering_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def _compute_clustering_leiden(connectivity_graph: spmatrix, resolution: float, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    igraph.set_random_number_generator(rng)
    # The connectivity graph with the umap method is symmetric, but we need to first make it directed
    # to have both sets of edges as is done in scanpy. See test for more details.
    g = igraph.Graph.Weighted_Adjacency(connectivity_graph, mode="directed")
    g.to_undirected(mode="each")
    clustering = g.community_leiden(objective_function="modularity", weights="weight", resolution=resolution)
    clusters = clustering.membership
    return np.asarray(clusters)


def _compute_nmi_ari_cluster_labels(
    X: spmatrix,
    labels: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> tuple[float, float]:
    labels_pred = _compute_clustering_leiden(X, resolution, seed)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)
    return nmi, ari


# ---------------------------------------------------------------------------
# GPU availability helpers
# ---------------------------------------------------------------------------


def _leiden_gpu_available() -> bool:
    """Return True when cugraph + cudf + torch CUDA + torchmetrics are all importable."""
    try:
        import cudf  # noqa: F401
        import cugraph  # noqa: F401
        import torch
        from torchmetrics.functional.clustering import normalized_mutual_info_score as _  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


def _kmeans_gpu_available() -> bool:
    """Return True when cuml + torch CUDA + torchmetrics are all importable."""
    try:
        from cuml.cluster import KMeans as _  # noqa: F401
        import torch
        from torchmetrics.functional.clustering import normalized_mutual_info_score as _  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# GPU clustering helpers
# ---------------------------------------------------------------------------


def _compute_clustering_leiden_gpu(connectivity_graph: spmatrix, resolution: float, seed: int) -> np.ndarray:
    """GPU-accelerated Leiden clustering via cugraph.

    Parameters
    ----------
    connectivity_graph
        Symmetric sparse connectivity matrix (n_cells × n_cells).
    resolution
        Leiden resolution parameter.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Cluster membership array of length n_cells.
    """
    import cudf
    import cugraph

    n = connectivity_graph.shape[0]
    cx = connectivity_graph.tocoo()
    edge_df = cudf.DataFrame(
        {
            "src": cudf.Series(cx.row.astype("int32")),
            "dst": cudf.Series(cx.col.astype("int32")),
            "weight": cudf.Series(cx.data.astype("float32")),
        }
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight")
    parts, _ = cugraph.leiden(G, resolution=float(resolution), random_state=seed)
    # Merge against full vertex range to handle any isolated vertex edge-cases
    all_v = cudf.DataFrame({"vertex": cudf.Series(np.arange(n, dtype="int32"))})
    parts = all_v.merge(parts, on="vertex", how="left").sort_values("vertex").reset_index(drop=True)
    # to_numpy() on an int column with NaN entries yields float64; fix in numpy
    arr = parts["partition"].to_numpy()  # float64, NaN where vertex was isolated
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        next_id = int(np.nanmax(arr)) + 1
        arr[nan_mask] = np.arange(next_id, next_id + int(nan_mask.sum()))
    return arr.astype("int32")


def _compute_clustering_kmeans_gpu(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """GPU-accelerated k-means clustering via cuml.

    Parameters
    ----------
    X
        2-D feature matrix.
    n_clusters
        Number of clusters (k).

    Returns
    -------
    Cluster label array of length n_cells.
    """
    from cuml.cluster import KMeans as cuKMeans

    kmeans = cuKMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    # cuml returns cupy arrays; convert to numpy
    if hasattr(labels, "get"):
        return labels.get()
    return np.asarray(labels)


def _compute_nmi_ari_gpu(labels_true: np.ndarray, labels_pred: np.ndarray) -> tuple[float, float]:
    """Compute NMI and ARI on GPU via torchmetrics.

    Parameters
    ----------
    labels_true
        Ground-truth label array (may contain strings).
    labels_pred
        Predicted integer cluster label array.

    Returns
    -------
    (nmi, ari) as Python floats.
    """
    import torch
    from torchmetrics.functional.clustering import normalized_mutual_info_score as tm_nmi

    # Encode ground-truth labels to contiguous integers (handles string labels)
    labels_true_int = np.asarray(pd.Categorical(labels_true).codes, dtype="int64")
    device = torch.device("cuda")
    t_true = torch.tensor(labels_true_int, dtype=torch.long, device=device)
    t_pred = torch.tensor(np.asarray(labels_pred, dtype="int64"), dtype=torch.long, device=device)

    nmi = tm_nmi(t_pred, t_true, average_method="arithmetic").item()
    # torchmetrics ARI uses int64 arithmetic internally; for large datasets (≥~100k cells)
    # products of O(n²) values overflow int64, producing values outside [-1, 1].
    # sklearn uses float64 throughout and is correct regardless of dataset size.
    ari = adjusted_rand_score(labels_true, np.asarray(labels_pred))
    return float(nmi), float(ari)


def _compute_nmi_ari_cluster_labels_gpu(
    X: spmatrix,
    labels: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> tuple[float, float]:
    labels_pred = _compute_clustering_leiden_gpu(X, resolution, seed)
    return _compute_nmi_ari_gpu(labels, labels_pred)


def nmi_ari_cluster_labels_kmeans(
    X: np.ndarray,
    labels: np.ndarray,
    flavor: Literal["auto", "cpu", "gpu"] = "auto",
) -> dict[str, float]:
    """Compute nmi and ari between k-means clusters and labels.

    This deviates from the original implementation in scib by using k-means
    with k equal to the known number of cell types/labels. This leads to
    a more efficient computation of the nmi and ari scores.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values
    flavor
        Compute backend to use.

        - ``'cpu'``: scikit-learn KMeans + sklearn NMI/ARI (default CPU path).
        - ``'gpu'``: cuML KMeans + torchmetrics NMI/ARI on CUDA. Requires
          ``cuml``, ``torch`` with CUDA, and ``torchmetrics``.
        - ``'auto'``: use GPU when all required packages are available,
          otherwise fall back to CPU silently.

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    X = check_array(X, accept_sparse=False, ensure_2d=True)
    labels = np.asarray(labels)
    valid_mask = ~_nan_mask(labels)
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        warnings.warn(
            f"Found {n_nan} cells with NaN labels. These cells will be excluded from NMI/ARI computation.",
            UserWarning,
        )
        labels = labels[valid_mask]
        X = X[valid_mask]

    if flavor == "gpu" and not _kmeans_gpu_available():
        raise RuntimeError(
            "flavor='gpu' requested for nmi_ari_cluster_labels_kmeans but one or more required packages "
            "(cuml, torch with CUDA, torchmetrics) are not available."
        )
    use_gpu = (flavor == "gpu") or (flavor == "auto" and _kmeans_gpu_available())

    n_clusters = len(np.unique(labels))
    if use_gpu:
        labels_pred = _compute_clustering_kmeans_gpu(X, n_clusters)
        nmi, ari = _compute_nmi_ari_gpu(labels, labels_pred)
    else:
        labels_pred = _compute_clustering_kmeans(X, n_clusters)
        nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
        ari = adjusted_rand_score(labels, labels_pred)

    return {"nmi": float(nmi), "ari": float(ari)}


def nmi_ari_cluster_labels_leiden(
    X: NeighborsResults,
    labels: np.ndarray,
    optimize_resolution: bool = True,
    resolution: float = 1.0,
    n_jobs: int = 1,
    seed: int = 42,
    flavor: Literal["auto", "cpu", "gpu"] = "auto",
) -> dict[str, float]:
    """Compute nmi and ari between leiden clusters and labels.

    This deviates from the original implementation in scib by using leiden instead of
    louvain clustering. Installing joblib allows for parallelization of the leiden
    resoution optimization.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
    labels
        Array of shape (n_cells,) representing label values
    optimize_resolution
        Whether to optimize the resolution parameter of leiden clustering by searching over
        10 values
    resolution
        Resolution parameter of leiden clustering. Only used if optimize_resolution is False.
    n_jobs
        Number of jobs for parallelizing resolution optimization via joblib. If -1, all CPUs
        are used. Ignored when ``flavor='gpu'``.
    seed
        Seed used for reproducibility of clustering.
    flavor
        Compute backend to use.

        - ``'cpu'``: igraph Leiden + sklearn NMI/ARI (default CPU path, supports joblib).
        - ``'gpu'``: cugraph Leiden + torchmetrics NMI/ARI on CUDA. Requires
          ``cugraph``, ``cudf``, ``torch`` with CUDA, and ``torchmetrics``.
          Resolution optimisation runs serially (CUDA fork-safety).
        - ``'auto'``: use GPU when all required packages are available,
          otherwise fall back to CPU silently.

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    conn_graph = X.knn_graph_connectivities
    labels = np.asarray(labels)
    valid_mask = ~_nan_mask(labels)
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        warnings.warn(
            f"Found {n_nan} cells with NaN labels. These cells will be excluded from NMI/ARI computation.",
            UserWarning,
        )
        labels = labels[valid_mask]
        conn_graph = conn_graph[valid_mask][:, valid_mask]

    if flavor == "gpu" and not _leiden_gpu_available():
        raise RuntimeError(
            "flavor='gpu' requested for nmi_ari_cluster_labels_leiden but one or more required packages "
            "(cugraph, cudf, torch with CUDA, torchmetrics) are not available."
        )
    use_gpu = (flavor == "gpu") or (flavor == "auto" and _leiden_gpu_available())

    if use_gpu:
        # GPU path: serial resolution search (CUDA context is not fork-safe with joblib)
        try:
            if optimize_resolution:
                n = 10
                resolutions = np.array([2 * x / n for x in range(1, n + 1)])
                out = [
                    _compute_nmi_ari_cluster_labels_gpu(conn_graph, labels, r, seed=seed) for r in resolutions
                ]
                nmi_ari = np.array(out)
                nmi_ind = np.argmax(nmi_ari[:, 0])
                nmi, ari = nmi_ari[nmi_ind, :]
            else:
                nmi, ari = _compute_nmi_ari_cluster_labels_gpu(conn_graph, labels, resolution, seed=seed)
            return {"nmi": float(nmi), "ari": float(ari)}
        except RuntimeError as e:
            if "out_of_memory" in str(e) or "bad_alloc" in str(e) or "cudaErrorMemoryAllocation" in str(e):
                # Free GPU memory from all known pools before falling back to CPU
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                warnings.warn(
                    f"GPU out-of-memory during Leiden NMI/ARI computation ({e}). "
                    "Falling back to CPU path.",
                    RuntimeWarning,
                )
                use_gpu = False
            else:
                raise

    # CPU path
    if optimize_resolution:
        n = 10
        resolutions = np.array([2 * x / n for x in range(1, n + 1)])
        try:
            from joblib import Parallel, delayed

            out = Parallel(n_jobs=n_jobs)(
                delayed(_compute_nmi_ari_cluster_labels)(conn_graph, labels, r, seed=seed) for r in resolutions
            )
        except ImportError:
            warnings.warn("Using for loop over clustering resolutions. `pip install joblib` for parallelization.")
            out = [_compute_nmi_ari_cluster_labels(conn_graph, labels, r, seed=seed) for r in resolutions]
        nmi_ari = np.array(out)
        nmi_ind = np.argmax(nmi_ari[:, 0])
        nmi, ari = nmi_ari[nmi_ind, :]
        return {"nmi": float(nmi), "ari": float(ari)}
    else:
        nmi, ari = _compute_nmi_ari_cluster_labels(conn_graph, labels, resolution, seed=seed)

    return {"nmi": float(nmi), "ari": float(ari)}
