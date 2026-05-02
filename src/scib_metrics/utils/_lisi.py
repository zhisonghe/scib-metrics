import math
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ._utils import get_ndarray

NdArray = np.ndarray | jnp.ndarray


@chex.dataclass
class _NeighborProbabilityState:
    H: float
    P: chex.ArrayDevice
    Hdiff: float
    beta: float
    betamin: float
    betamax: float
    tries: int


@jax.jit
def _Hbeta(knn_dists_row: jnp.ndarray, row_self_mask: jnp.ndarray, beta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    P = jnp.exp(-knn_dists_row * beta)
    # Mask out self edges to be zero
    P = jnp.where(row_self_mask, P, 0)
    sumP = jnp.nansum(P)
    H = jnp.where(sumP == 0, 0, jnp.log(sumP) + beta * jnp.nansum(knn_dists_row * P) / sumP)
    P = jnp.where(sumP == 0, jnp.zeros_like(knn_dists_row), P / sumP)
    return H, P


@jax.jit
def _get_neighbor_probability(
    knn_dists_row: jnp.ndarray, row_self_mask: jnp.ndarray, perplexity: float, tol: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    beta = 1
    betamin = -jnp.inf
    betamax = jnp.inf
    H, P = _Hbeta(knn_dists_row, row_self_mask, beta)
    Hdiff = H - jnp.log(perplexity)

    def _get_neighbor_probability_step(state):
        Hdiff = state.Hdiff
        beta = state.beta
        betamin = state.betamin
        betamax = state.betamax
        tries = state.tries

        new_betamin = jnp.where(Hdiff > 0, beta, betamin)
        new_betamax = jnp.where(Hdiff > 0, betamax, beta)
        new_beta = jnp.where(
            Hdiff > 0,
            jnp.where(betamax == jnp.inf, beta * 2, (beta + betamax) / 2),
            jnp.where(betamin == -jnp.inf, beta / 2, (beta + betamin) / 2),
        )
        new_H, new_P = _Hbeta(knn_dists_row, row_self_mask, new_beta)
        new_Hdiff = new_H - jnp.log(perplexity)
        return _NeighborProbabilityState(
            H=new_H, P=new_P, Hdiff=new_Hdiff, beta=new_beta, betamin=new_betamin, betamax=new_betamax, tries=tries + 1
        )

    def _get_neighbor_probability_convergence(state):
        Hdiff, tries = state.Hdiff, state.tries
        return jnp.logical_and(jnp.abs(Hdiff) >= tol, tries < 50)

    init_state = _NeighborProbabilityState(H=H, P=P, Hdiff=Hdiff, beta=beta, betamin=betamin, betamax=betamax, tries=0)
    final_state = jax.lax.while_loop(_get_neighbor_probability_convergence, _get_neighbor_probability_step, init_state)
    return final_state.H, final_state.P


def _compute_simpson_index_cell(
    knn_dists_row: jnp.ndarray,
    knn_labels_row: jnp.ndarray,
    row_self_mask: jnp.ndarray,
    n_batches: int,
    perplexity: float,
    tol: float,
) -> jnp.ndarray:
    H, P = _get_neighbor_probability(knn_dists_row, row_self_mask, perplexity, tol)

    def _non_zero_H_simpson():
        sumP = jnp.bincount(knn_labels_row, weights=P, length=n_batches)
        return jnp.where(knn_labels_row.shape[0] == P.shape[0], jnp.dot(sumP, sumP), 1)

    return jnp.where(H == 0, -1, _non_zero_H_simpson())


def _lisi_torch_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_lisi_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("out_of_memory", "bad_alloc", "cudaerrormemorya"))


def _flush_gpu_memory_lisi() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _compute_simpson_index_torch(
    knn_dists: np.ndarray,
    knn_idx: np.ndarray,
    row_idx: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    perplexity: float,
    tol: float,
) -> np.ndarray:
    """Vectorised PyTorch implementation of the Simpson index (runs on GPU)."""
    import torch

    device = torch.device("cuda")
    knn_dists_t = torch.tensor(np.ascontiguousarray(knn_dists), dtype=torch.float32, device=device)
    knn_idx_t = torch.tensor(np.ascontiguousarray(knn_idx), dtype=torch.long, device=device)
    row_idx_t = torch.tensor(np.ascontiguousarray(row_idx.ravel()), dtype=torch.long, device=device)
    labels_t = torch.tensor(np.ascontiguousarray(labels), dtype=torch.long, device=device)

    n_cells = knn_dists_t.shape[0]
    knn_labels_t = labels_t[knn_idx_t]  # (n_cells, n_neighbors)
    # Float mask so we can multiply directly: 1 = keep, 0 = self-edge
    self_mask = (knn_idx_t != row_idx_t.unsqueeze(1)).float()

    log_perp = math.log(max(float(perplexity), 1e-10))

    beta = torch.ones(n_cells, dtype=torch.float32, device=device)
    betamin = torch.full((n_cells,), float("-inf"), dtype=torch.float32, device=device)
    betamax = torch.full((n_cells,), float("inf"), dtype=torch.float32, device=device)

    for _ in range(50):
        P = torch.exp(-knn_dists_t * beta.unsqueeze(1)) * self_mask  # (n_cells, k)
        sumP = P.sum(dim=1)  # (n_cells,)

        nz = sumP > 0
        H = torch.zeros_like(beta)
        H[nz] = torch.log(sumP[nz]) + beta[nz] * (knn_dists_t[nz] * P[nz]).sum(dim=1) / sumP[nz]

        Hdiff = H - log_perp
        converged = Hdiff.abs() < tol

        new_betamin = torch.where(Hdiff > 0, beta, betamin)
        new_betamax = torch.where(Hdiff > 0, betamax, beta)
        new_beta = torch.where(
            Hdiff > 0,
            torch.where(betamax.isinf(), beta * 2, (beta + betamax) / 2),
            torch.where(betamin.isinf(), beta / 2, (beta + betamin) / 2),
        )
        betamin = torch.where(converged, betamin, new_betamin)
        betamax = torch.where(converged, betamax, new_betamax)
        beta = torch.where(converged, beta, new_beta)

    # Compute final P and H with the converged beta values
    P = torch.exp(-knn_dists_t * beta.unsqueeze(1)) * self_mask
    sumP = P.sum(dim=1)
    nz = sumP > 0
    H_final = torch.zeros_like(beta)
    H_final[nz] = torch.log(sumP[nz]) + beta[nz] * (knn_dists_t[nz] * P[nz]).sum(dim=1) / sumP[nz]
    P = P / sumP.clamp(min=1e-30).unsqueeze(1)

    # Simpson index: sum_c ( sum_{j: label==c} P_j )^2
    clust_P = torch.zeros(n_cells, n_labels, dtype=torch.float32, device=device)
    clust_P.scatter_add_(1, knn_labels_t, P)
    simpson = (clust_P**2).sum(dim=1)

    # Cells with H==0 are treated as invalid → return -1 (matches JAX behaviour)
    result = torch.where(H_final == 0, torch.tensor(-1.0, device=device), simpson)
    return result.cpu().numpy()


def compute_simpson_index(
    knn_dists: NdArray,
    knn_idx: NdArray,
    row_idx: NdArray,
    labels: NdArray,
    n_labels: int,
    perplexity: float = 30,
    tol: float = 1e-5,
    flavor: str = "auto",
) -> np.ndarray:
    """Compute the Simpson index for each cell.

    Parameters
    ----------
    knn_dists
        KNN distances of size (n_cells, n_neighbors).
    knn_idx
        KNN indices of size (n_cells, n_neighbors) corresponding to distances.
    row_idx
        Idx of each row (n_cells, 1).
    labels
        Cell labels of size (n_cells,).
    n_labels
        Number of labels.
    perplexity
        Measure of the effective number of neighbors.
    tol
        Tolerance for binary search.
    flavor
        Backend to use. ``"auto"`` (default) uses the PyTorch GPU backend when
        CUDA is available and falls back to JAX; ``"torch"`` forces the PyTorch
        backend (GPU if available, otherwise CPU); ``"jax"`` forces JAX.

    Returns
    -------
    simpson_index
        Simpson index of size (n_cells,).
    """
    use_torch = flavor == "torch" or (flavor == "auto" and _lisi_torch_available())
    if use_torch:
        try:
            return _compute_simpson_index_torch(
                np.asarray(knn_dists),
                np.asarray(knn_idx),
                np.asarray(row_idx),
                np.asarray(labels),
                n_labels,
                perplexity,
                tol,
            )
        except ImportError:
            pass  # torch not installed — fall through to JAX
        except RuntimeError as e:
            if _is_lisi_oom(e):
                _flush_gpu_memory_lisi()
                warnings.warn(
                    "CUDA out-of-memory during Simpson index computation. "
                    "Falling back to JAX CPU backend.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise

    # JAX (CPU) path
    knn_dists = jnp.array(knn_dists)
    knn_idx = jnp.array(knn_idx)
    labels = jnp.array(labels)
    row_idx = jnp.array(row_idx)
    knn_labels = labels[knn_idx]
    self_mask = knn_idx != row_idx
    simpson_fn = partial(_compute_simpson_index_cell, n_batches=n_labels, perplexity=perplexity, tol=tol)
    out = jax.vmap(simpson_fn)(knn_dists, knn_labels, self_mask)
    return get_ndarray(out)
