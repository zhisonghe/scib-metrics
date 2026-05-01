import logging
import warnings
from functools import partial
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy

from scib_metrics._types import NdArray
from scib_metrics.nearest_neighbors import NeighborsResults
from scib_metrics.utils import diffusion_nn, get_ndarray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU (PyTorch) availability helper
# ---------------------------------------------------------------------------


def _kbet_gpu_available() -> bool:
    """Return True when PyTorch with CUDA support is importable and a GPU is present."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # torch.special.gammainc was added in PyTorch 1.8
        if not hasattr(torch.special, "gammainc"):
            return False
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PyTorch GPU kBET kernel
# ---------------------------------------------------------------------------


def _kbet_torch(neigh_batch_ids: np.ndarray, batches: np.ndarray, n_batches: int):
    """Compute kBET chi-square statistics and p-values on GPU via PyTorch.

    Parameters
    ----------
    neigh_batch_ids
        Integer array of shape (n_cells, k) — batch ID for each neighbor.
    batches
        Integer array of shape (n_cells,) — batch ID per cell.
    n_batches
        Total number of distinct batches.

    Returns
    -------
    test_statistics
        Chi-square statistic per cell, shape (n_cells,).
    p_values
        p-value per cell, shape (n_cells,).
    """
    import torch

    device = torch.device("cuda")
    t_neigh = torch.tensor(np.ascontiguousarray(neigh_batch_ids), dtype=torch.long, device=device)  # (n_cells, k)
    t_batches = torch.tensor(np.ascontiguousarray(batches), dtype=torch.long, device=device)  # (n_cells,)

    n_cells, k = t_neigh.shape

    # Global expected frequency of each batch
    expected_freq = torch.bincount(t_batches, minlength=n_batches).float()
    expected_freq = expected_freq / expected_freq.sum()  # (n_batches,)

    # Observed counts: for each cell, how many of its k neighbors belong to each batch
    observed = torch.zeros(n_cells, n_batches, dtype=torch.float32, device=device)
    observed.scatter_add_(1, t_neigh, torch.ones(n_cells, k, dtype=torch.float32, device=device))

    # Expected counts per cell
    expected_counts = expected_freq * k  # (n_batches,) broadcast over cells

    # Chi-squared statistic (cells × batches summed to cells)
    dof = n_batches - 1
    test_statistics = ((observed - expected_counts) ** 2 / expected_counts).sum(dim=1)  # (n_cells,)

    # p-value = 1 - chi2_cdf(dof, stat) = 1 - regularized_lower_gamma(dof/2, stat/2)
    a = torch.tensor(dof / 2.0, dtype=torch.float32, device=device)
    p_values = 1.0 - torch.special.gammainc(a, test_statistics / 2.0)

    return test_statistics.cpu().numpy(), p_values.cpu().numpy()


def _chi2_cdf(df: int | NdArray, x: NdArray) -> float:
    """Chi2 cdf.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtr.html
    for explanation of gammainc.
    """
    return jax.scipy.special.gammainc(df / 2, x / 2)


@partial(jax.jit, static_argnums=2)
def _kbet(neigh_batch_ids: jnp.ndarray, batches: jnp.ndarray, n_batches: int) -> float:
    expected_freq = jnp.bincount(batches, length=n_batches)
    expected_freq = expected_freq / jnp.sum(expected_freq)
    dof = n_batches - 1

    observed_counts = jax.vmap(partial(jnp.bincount, length=n_batches))(neigh_batch_ids)
    expected_counts = expected_freq * neigh_batch_ids.shape[1]
    test_statistics = jnp.sum(jnp.square(observed_counts - expected_counts) / expected_counts, axis=1)
    p_values = 1 - jax.vmap(_chi2_cdf, in_axes=(None, 0))(dof, test_statistics)

    return test_statistics, p_values


def kbet(
    X: NeighborsResults,
    batches: np.ndarray,
    alpha: float = 0.05,
    flavor: Literal["auto", "jax", "torch"] = "auto",
) -> float:
    """Compute kbet :cite:p:`buttner2018`.

    This implementation is inspired by the implementation in Pegasus:
    https://pegasus.readthedocs.io/en/stable/index.html

    A higher acceptance rate means more mixing of batches. This implementation does
    not exactly mirror the default original implementation, as there is currently no
    `adapt` option.

    Note that this is also not equivalent to the kbet used in the original scib package,
    as that one computes kbet for each cell type label. To achieve this, use
    :func:`scib_metrics.kbet_per_label`.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
    batches
        Array of shape (n_cells,) representing batch values
        for each cell.
    alpha
        Significance level for the statistical test.
    flavor
        Which backend to use for computation.  ``"auto"`` (default) selects
        ``"torch"`` when a CUDA-capable GPU is available (via PyTorch), and
        falls back to ``"jax"`` otherwise.  ``"jax"`` forces the JAX backend.
        ``"torch"`` forces the PyTorch CUDA backend (raises ``RuntimeError``
        if no GPU is found).

    Returns
    -------
    acceptance_rate
        Kbet acceptance rate of the sample.
    stat_mean
        Mean Kbet chi-square statistic over all cells.
    pvalue_mean
        Mean Kbet p-value over all cells.
    """
    if len(batches) != len(X.indices):
        raise ValueError("Length of batches does not match number of cells.")
    knn_idx = X.indices
    batches = np.asarray(pd.Categorical(batches).codes)
    neigh_batch_ids = batches[knn_idx]
    chex.assert_equal_shape([neigh_batch_ids, knn_idx])
    n_batches = len(np.unique(batches))

    use_gpu = (flavor == "torch") or (flavor == "auto" and _kbet_gpu_available())
    if flavor == "torch" and not _kbet_gpu_available():
        raise RuntimeError(
            "flavor='torch' requested but PyTorch CUDA is not available. "
            "Install torch with CUDA support or use flavor='auto'/'jax'."
        )

    if use_gpu:
        test_statistics, p_values = _kbet_torch(neigh_batch_ids, batches, n_batches)
    else:
        test_statistics, p_values = _kbet(
            jnp.array(neigh_batch_ids), jnp.array(batches), n_batches
        )
        test_statistics = get_ndarray(test_statistics)
        p_values = get_ndarray(p_values)

    acceptance_rate = (p_values >= alpha).mean()
    return acceptance_rate, test_statistics, p_values


def kbet_per_label(
    X: NeighborsResults,
    batches: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    diffusion_n_comps: int = 100,
    return_df: bool = False,
    flavor: Literal["auto", "jax", "torch"] = "auto",
) -> float | tuple[float, pd.DataFrame]:
    """Compute kBET score per cell type label as in :cite:p:`luecken2022benchmarking`.

    This approximates the method used in the original scib package. Notably, the underlying
    kbet might have some inconsistencies with the R implementation. Furthermore, to equalize
    the neighbor graphs of cell type subsets we use diffusion distance approximated with diffusion
    maps. Increasing `diffusion_n_comps` will increase the accuracy of the approximation.

    Parameters
    ----------
    X
        A :class:`~scib_metrics.utils.nearest_neighbors.NeighborsResults` object.
    batches
        Array of shape (n_cells,) representing batch values
        for each cell.
    labels
        Array of shape (n_cells,) representing cell type labels.
    alpha
        Significance level for the statistical test.
    diffusion_n_comps
        Number of diffusion components to use for diffusion distance approximation.
    return_df
        Return dataframe of results in addition to score.
    flavor
        Which backend to use for computation.  ``"auto"`` (default) selects
        ``"torch"`` when a CUDA-capable GPU is available (via PyTorch), and
        falls back to ``"jax"`` otherwise.  Forwarded to each internal call
        of :func:`kbet`.

    Returns
    -------
    kbet_score
        Kbet score over all cells. Higher means more integrated, as in the kBET acceptance rate.
    df
        Dataframe with kBET score per cell type label.

    Notes
    -----
    This function requires X to be cell-cell connectivities, not distances.
    """
    if len(batches) != len(X.indices):
        raise ValueError("Length of batches does not match number of cells.")
    if len(labels) != len(X.indices):
        raise ValueError("Length of labels does not match number of cells.")
    # set upper bound for k0
    size_max = 2**31 - 1
    batches = np.asarray(pd.Categorical(batches).codes)
    labels = np.asarray(labels)

    conn_graph = X.knn_graph_connectivities

    # Drop cells with NaN labels
    nan_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in labels], dtype=bool)
    if nan_mask.any():
        warnings.warn(
            f"Found {nan_mask.sum()} cells with NaN labels. These cells will be excluded from kBET computation.",
            UserWarning,
        )
        valid = ~nan_mask
        labels = labels[valid]
        batches = batches[valid]
        conn_graph = conn_graph[valid][:, valid]

    # prepare call of kBET per cluster
    kbet_scores = {"cluster": [], "kBET": []}
    for clus in np.unique(labels):
        # subset by label
        mask = labels == clus
        conn_graph_sub = conn_graph[mask, :][:, mask]
        conn_graph_sub.sort_indices()
        n_obs = conn_graph_sub.shape[0]
        batches_sub = batches[mask]

        # check if neighborhood size too small or only one batch in subset
        if np.logical_or(n_obs < 10, len(np.unique(batches_sub)) == 1):
            logger.info(f"{clus} consists of a single batch or is too small. Skip.")
            score = np.nan
        else:
            quarter_mean = np.floor(np.mean(pd.Series(batches_sub).value_counts()) / 4).astype("int")
            k0 = np.min([70, np.max([10, quarter_mean])])
            # check k0 for reasonability
            if k0 * n_obs >= size_max:
                k0 = np.floor(size_max / n_obs).astype("int")

            n_comp, labs = scipy.sparse.csgraph.connected_components(conn_graph_sub, connection="strong")

            if n_comp == 1:  # a single component to compute kBET on
                try:
                    diffusion_n_comps = np.min([diffusion_n_comps, n_obs - 1])
                    nn_graph_sub = diffusion_nn(conn_graph_sub, k=k0, n_comps=diffusion_n_comps)
                    # call kBET
                    score, _, _ = kbet(
                        nn_graph_sub,
                        batches=batches_sub,
                        alpha=alpha,
                        flavor=flavor,
                    )
                except ValueError:
                    logger.info("Diffusion distance failed. Skip.")
                    score = 0  # i.e. 100% rejection

            else:
                # check the number of components where kBET can be computed upon
                comp_size = pd.Series(labs).value_counts()
                # check which components are small
                comp_size_thresh = 3 * k0
                idx_nonan = np.flatnonzero(np.isin(labs, comp_size[comp_size >= comp_size_thresh].index))

                # check if 75% of all cells can be used for kBET run
                if len(idx_nonan) / len(labs) >= 0.75:
                    # create another subset of components, assume they are not visited in a diffusion process
                    conn_graph_sub_sub = conn_graph_sub[idx_nonan, :][:, idx_nonan]
                    conn_graph_sub_sub.sort_indices()

                    try:
                        diffusion_n_comps = np.min([diffusion_n_comps, conn_graph_sub_sub.shape[0] - 1])
                        nn_results_sub_sub = diffusion_nn(conn_graph_sub_sub, k=k0, n_comps=diffusion_n_comps)
                        # call kBET
                        score, _, _ = kbet(
                            nn_results_sub_sub,
                            batches=batches_sub[idx_nonan],
                            alpha=alpha,
                            flavor=flavor,
                        )
                    except ValueError:
                        logger.info("Diffusion distance failed. Skip.")
                        score = 0  # i.e. 100% rejection
                else:  # if there are too many too small connected components, set kBET score to 0
                    score = 0  # i.e. 100% rejection

        kbet_scores["cluster"].append(clus)
        kbet_scores["kBET"].append(score)

    kbet_scores = pd.DataFrame.from_dict(kbet_scores)
    kbet_scores = kbet_scores.reset_index(drop=True)

    final_score = np.nanmean(kbet_scores["kBET"])
    if not return_df:
        return final_score
    else:
        return final_score, kbet_scores
