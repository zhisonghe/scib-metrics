from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._dist import cdist
from ._utils import get_ndarray


# ---------------------------------------------------------------------------
# GPU (PyTorch) availability helper
# ---------------------------------------------------------------------------


def _silhouette_torch_available() -> bool:
    """Return True when PyTorch with CUDA support is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PyTorch silhouette kernel
# ---------------------------------------------------------------------------


def _silhouette_reduce_torch(
    D_chunk,  # torch.Tensor (n_chunk, n_samples) on GPU
    start: int,
    labels,  # torch.Tensor (n_samples,) int64
    label_freqs,  # torch.Tensor (n_clusters,) float32
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
):
    """PyTorch equivalent of _silhouette_reduce."""
    import torch

    n_chunk = D_chunk.shape[0]
    n_clusters = label_freqs.shape[0]

    # Accumulate distances from each chunk sample to each cluster via scatter_add
    # clust_dists[i, c] = sum of D_chunk[i, j] for all j where labels[j] == c
    clust_dists = torch.zeros(n_chunk, n_clusters, dtype=D_chunk.dtype, device=D_chunk.device)
    label_col = labels.unsqueeze(0).expand(n_chunk, -1)  # (n_chunk, n_samples)
    clust_dists.scatter_add_(1, label_col, D_chunk)

    chunk_labels = labels[start : start + n_chunk]  # (n_chunk,)
    intra_index = (torch.arange(n_chunk, device=D_chunk.device), chunk_labels)
    intra_clust_dists = clust_dists[intra_index]  # (n_chunk,)

    if between_cluster_distances == "furthest":
        clust_dists[intra_index] = -torch.inf
        clust_dists = clust_dists / label_freqs
        inter_clust_dists = clust_dists.max(dim=1).values
    elif between_cluster_distances == "mean_other":
        clust_dists[intra_index] = float("nan")
        total_other_dists = torch.nansum(clust_dists, dim=1)
        total_other_count = label_freqs.sum() - label_freqs[chunk_labels]
        inter_clust_dists = total_other_dists / total_other_count
    elif between_cluster_distances == "nearest":
        clust_dists[intra_index] = torch.inf
        clust_dists = clust_dists / label_freqs
        inter_clust_dists = clust_dists.min(dim=1).values
    else:
        raise ValueError("Parameter 'between_cluster_distances' must be one of ['nearest', 'mean_other', 'furthest'].")

    return intra_clust_dists, inter_clust_dists


def _pairwise_distances_chunked_torch(
    X,  # torch.Tensor (n_samples, n_features) on GPU
    labels,  # torch.Tensor (n_samples,) int64
    label_freqs,  # torch.Tensor (n_clusters,) float32
    chunk_size: int,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
):
    """Chunked pairwise distances + silhouette reduce, all on GPU via PyTorch."""
    import torch

    n_samples = X.shape[0]
    intra_all = []
    inter_all = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = X[start:end]  # (n_chunk, n_features)
        if metric == "cosine":
            D_chunk = 1.0 - torch.nn.functional.normalize(chunk, dim=1) @ torch.nn.functional.normalize(X, dim=1).T
            D_chunk = D_chunk.clamp(0.0, 2.0)
        else:  # euclidean
            D_chunk = torch.cdist(chunk, X, p=2)
        intra, inter = _silhouette_reduce_torch(
            D_chunk, start, labels, label_freqs, between_cluster_distances
        )
        intra_all.append(intra)
        inter_all.append(inter)
    return torch.cat(intra_all), torch.cat(inter_all)


def _silhouette_samples_torch(
    X: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> np.ndarray:
    """GPU silhouette via PyTorch."""
    import torch

    device = torch.device("cuda")
    codes = pd.Categorical(labels).codes
    t_labels = torch.tensor(np.ascontiguousarray(codes), dtype=torch.long, device=device)
    t_freqs = torch.bincount(t_labels).float()
    t_X = torch.tensor(np.ascontiguousarray(X.astype(np.float32)), device=device)

    intra_clust_dists, inter_clust_dists = _pairwise_distances_chunked_torch(
        t_X, t_labels, t_freqs, chunk_size=chunk_size, metric=metric,
        between_cluster_distances=between_cluster_distances,
    )

    denom = (t_freqs - 1)[t_labels].clamp(min=1.0)
    intra_clust_dists = intra_clust_dists / denom
    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples = sil_samples / torch.maximum(intra_clust_dists, inter_clust_dists)
    sil_samples = torch.nan_to_num(sil_samples)
    return sil_samples.cpu().numpy()


# ---------------------------------------------------------------------------
# JAX silhouette kernel (original)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=["between_cluster_distances"])
def _silhouette_reduce(
    D_chunk: jnp.ndarray,
    start: int,
    labels: jnp.ndarray,
    label_freqs: jnp.ndarray,
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accumulate silhouette statistics for vertical chunk of X.

    Follows scikit-learn implementation with default parameter usage ('nearest').

    Additional options enable BRAS compatible usage, addressing specific limitations of using silhouette in the context
     of evaluating data integration (see :func:`~scib_metrics.metrics.bras` documentation).


    Parameters
    ----------
    D_chunk
        Array of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    start
        First index in the chunk.
    labels
        Array of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs
        Distribution of cluster labels in ``labels``.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'nearest': Standard silhouette (distance to nearest cluster)
        - 'mean_other': BRAS-specific (mean distance to all other clusters)
        - 'furthest': BRAS-specific (distance to furthest cluster)

    """
    # accumulate distances from each sample to each cluster
    D_chunk_len = D_chunk.shape[0]

    clust_dists = jax.vmap(partial(jnp.bincount, length=label_freqs.shape[0]), in_axes=(None, 0))(labels, D_chunk)

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (jnp.arange(D_chunk_len), jax.lax.dynamic_slice(labels, (start,), (D_chunk_len,)))
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]

    if between_cluster_distances == "furthest":
        # of the remaining distances we normalise and extract the maximum
        clust_dists = clust_dists.at[intra_index].set(-jnp.inf)
        clust_dists /= label_freqs
        inter_clust_dists = clust_dists.max(axis=1)
    elif between_cluster_distances == "mean_other":
        clust_dists = clust_dists.at[intra_index].set(jnp.nan)
        total_other_dists = jnp.nansum(clust_dists, axis=1)
        total_other_count = jnp.sum(label_freqs) - label_freqs[jax.lax.dynamic_slice(labels, (start,), (D_chunk_len,))]
        inter_clust_dists = total_other_dists / total_other_count
    elif between_cluster_distances == "nearest":
        # of the remaining distances we normalise and extract the minimum
        clust_dists = clust_dists.at[intra_index].set(jnp.inf)
        clust_dists /= label_freqs
        inter_clust_dists = clust_dists.min(axis=1)
    else:
        raise ValueError("Parameter 'between_cluster_distances' must be one of ['nearest', 'mean_other', 'furthest'].")
    return intra_clust_dists, inter_clust_dists


def _pairwise_distances_chunked(
    X: jnp.ndarray, chunk_size: int, reduce_fn: callable, metric: Literal["euclidean", "cosine"] = "euclidean"
) -> jnp.ndarray:
    """Compute pairwise distances in chunks to reduce memory usage."""
    n_samples = X.shape[0]
    n_chunks = jnp.ceil(n_samples / chunk_size).astype(int)
    intra_dists_all = []
    inter_dists_all = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_samples)
        intra_cluster_dists, inter_cluster_dists = reduce_fn(cdist(X[start:end], X, metric=metric), start=start)
        intra_dists_all.append(intra_cluster_dists)
        inter_dists_all.append(inter_cluster_dists)
    return jnp.concatenate(intra_dists_all), jnp.concatenate(inter_dists_all)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
    flavor: Literal["auto", "torch", "jax"] = "auto",
) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation.

    Implements :func:`sklearn.metrics.silhouette_samples`.

    Default parameters ('euclidean', 'nearest') match scIB implementation.

    Additional options enable BRAS compatible usage (see `bras()` documentation).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features) representing a
        feature array.
    labels
        Array of shape (n_cells,) representing label values
        for each observation.
    chunk_size
        Number of samples to process at a time for distance computation.
    metric
        The distance metric to use. The distance function can be 'euclidean' (default) or 'cosine'.
    between_cluster_distances
        Method for computing inter-cluster distances.
        - 'nearest': Standard silhouette (distance to nearest cluster)
        - 'mean_other': BRAS-specific (mean distance to all other clusters)
        - 'furthest': BRAS-specific (distance to furthest cluster)
    flavor
        Which backend to use.  ``"auto"`` (default) selects ``"torch"`` when
        PyTorch with CUDA is available, and falls back to ``"jax"`` otherwise.
        ``"torch"`` forces the PyTorch CUDA backend (raises ``RuntimeError``
        if no GPU is found).  ``"jax"`` forces the original JAX backend.

    Returns
    -------
    silhouette scores array of shape (n_cells,)
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels should have the same number of samples")

    if flavor == "torch" and not _silhouette_torch_available():
        raise RuntimeError(
            "flavor='torch' requested but PyTorch CUDA is not available. "
            "Install torch with CUDA support or use flavor='auto'/'jax'."
        )

    use_torch = (flavor == "torch") or (flavor == "auto" and _silhouette_torch_available())

    if use_torch:
        return _silhouette_samples_torch(
            X, labels, chunk_size=chunk_size, metric=metric,
            between_cluster_distances=between_cluster_distances,
        )

    # JAX path
    labels_jax = jnp.asarray(pd.Categorical(labels).codes)
    label_freqs = jnp.bincount(labels_jax)
    reduce_fn = partial(
        _silhouette_reduce, labels=labels_jax, label_freqs=label_freqs,
        between_cluster_distances=between_cluster_distances,
    )
    results = _pairwise_distances_chunked(X, chunk_size=chunk_size, reduce_fn=reduce_fn, metric=metric)
    intra_clust_dists, inter_clust_dists = results

    denom = jnp.take(label_freqs - 1, labels_jax, mode="clip")
    intra_clust_dists /= denom
    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= jnp.maximum(intra_clust_dists, inter_clust_dists)
    return get_ndarray(jnp.nan_to_num(sil_samples))
