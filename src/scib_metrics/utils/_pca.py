import jax.numpy as jnp
import numpy as np
import warnings
from chex import dataclass
from jax import jit

from scib_metrics._types import NdArray

from ._utils import get_ndarray


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("out_of_memory", "bad_alloc", "cudaerrormemorya"))


def _flush_gpu_memory() -> None:
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


@dataclass
class _SVDResult:
    """SVD result.

    Attributes
    ----------
    u
        Array of shape (n_cells, n_components) containing the left singular vectors.
    s
        Array of shape (n_components,) containing the singular values.
    v
        Array of shape (n_components, n_features) containing the right singular vectors.
    """

    u: NdArray
    s: NdArray
    v: NdArray


@dataclass
class _PCAResult:
    """PCA result.

    Attributes
    ----------
    coordinates
        Array of shape (n_cells, n_components) containing the PCA coordinates.
    components
        Array of shape (n_components, n_features) containing the PCA components.
    variance
        Array of shape (n_components,) containing the explained variance of each PC.
    variance_ratio
        Array of shape (n_components,) containing the explained variance ratio of each PC.
    svd
        Dataclass containing the SVD data.
    """

    coordinates: NdArray
    components: NdArray
    variance: NdArray
    variance_ratio: NdArray
    svd: _SVDResult | None = None


def _svd_flip(
    u: NdArray,
    v: NdArray,
    u_based_decision: bool = True,
):
    """Sign correction to ensure deterministic output from SVD.

    Jax implementation of :func:`~sklearn.utils.extmath.svd_flip`.

    Parameters
    ----------
    u
        Left singular vectors of shape (M, K).
    v
        Right singular vectors of shape (K, N).
    u_based_decision
        If True, use the columns of u as the basis for sign flipping.
    """
    if u_based_decision:
        max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
        signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
    else:
        max_abs_rows = jnp.argmax(jnp.abs(v), axis=1)
        signs = jnp.sign(v[jnp.arange(v.shape[0]), max_abs_rows])
    u_ = u * signs
    v_ = v * signs[:, None]
    return u_, v_


def _svd_flip_torch(u, v, u_based_decision: bool = True):
    """Sign correction for deterministic SVD output — torch version."""
    import torch

    if u_based_decision:
        max_abs_cols = torch.abs(u).argmax(dim=0)  # (K,)
        signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1], device=u.device)])
    else:
        max_abs_rows = torch.abs(v).argmax(dim=1)  # (K,)
        signs = torch.sign(v[torch.arange(v.shape[0], device=v.device), max_abs_rows])
    return u * signs, v * signs.unsqueeze(1)


def _pca_torch(X: np.ndarray):
    """Compute PCA via torch.linalg.svd (GPU-accelerated when CUDA is available)."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    X_centered = X_t - X_t.mean(dim=0)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    U, Vh = _svd_flip_torch(U, Vh)
    variance = (S**2) / (X_t.shape[0] - 1)
    total_variance = variance.sum().clamp(min=1e-30)
    variance_ratio = variance / total_variance
    return (
        U.cpu().numpy(),
        S.cpu().numpy(),
        Vh.cpu().numpy(),
        variance.cpu().numpy(),
        variance_ratio.cpu().numpy(),
    )


def pca(
    X: NdArray,
    n_components: int | None = None,
    return_svd: bool = False,
    flavor: str = "auto",
) -> _PCAResult:
    """Principal component analysis (PCA).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    n_components
        Number of components to keep. If None, all components are kept.
    return_svd
        If True, also return the results from SVD.
    flavor
        Backend to use. ``"auto"`` (default) and ``"torch"`` use
        :func:`torch.linalg.svd` (GPU if CUDA is available, else CPU).
        ``"jax"`` uses JAX.

    Returns
    -------
    results: _PCAData
    """
    max_components = min(X.shape)
    if n_components and n_components > max_components:
        raise ValueError(f"n_components = {n_components} must be <= min(n_cells, n_features) = {max_components}")
    n_components = n_components or max_components

    use_torch = flavor in ("torch", "auto")
    if use_torch:
        try:
            u, s, v, variance_, variance_ratio_ = _pca_torch(np.asarray(X))
        except ImportError:
            use_torch = False
        except RuntimeError as e:
            if _is_oom_error(e):
                _flush_gpu_memory()
                warnings.warn(
                    "CUDA out-of-memory during PCA. Falling back to JAX CPU backend.",
                    UserWarning,
                    stacklevel=2,
                )
                use_torch = False
            else:
                raise

    if not use_torch:
        u, s, v, variance_, variance_ratio_ = _pca(X)
        u, s, v = get_ndarray(u), get_ndarray(s), get_ndarray(v)
        variance_, variance_ratio_ = get_ndarray(variance_), get_ndarray(variance_ratio_)

    # Select n_components
    coordinates = u[:, :n_components] * s[:n_components]
    components = v[:n_components]
    variance_out = variance_[:n_components]
    variance_ratio_out = variance_ratio_[:n_components]

    results = _PCAResult(
        coordinates=coordinates,
        components=components,
        variance=variance_out,
        variance_ratio=variance_ratio_out,
        svd=_SVDResult(u=u, s=s, v=v) if return_svd else None,
    )
    return results


@jit
def _pca(
    X: NdArray,
) -> tuple[NdArray, NdArray, NdArray, NdArray, NdArray]:
    """Principal component analysis.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).

    Returns
    -------
    u: NdArray
        Left singular vectors of shape (M, K).
    s: NdArray
        Singular values of shape (K,).
    v: NdArray
        Right singular vectors of shape (K, N).
    variance: NdArray
        Array of shape (K,) containing the explained variance of each PC.
    variance_ratio: NdArray
        Array of shape (K,) containing the explained variance ratio of each PC.
    """
    X_ = X - jnp.mean(X, axis=0)
    u, s, v = jnp.linalg.svd(X_, full_matrices=False)
    u, v = _svd_flip(u, v)

    variance = (s**2) / (X.shape[0] - 1)
    total_variance = jnp.sum(variance)
    variance_ratio = variance / total_variance

    return u, s, v, variance, variance_ratio
