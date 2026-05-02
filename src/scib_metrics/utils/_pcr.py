import jax.numpy as jnp
import numpy as np
import pandas as pd
import warnings
from jax import jit

from scib_metrics._types import NdArray

from ._pca import _flush_gpu_memory, _is_oom_error, pca
from ._utils import one_hot


def _pcr_torch(X_pca: np.ndarray, covariate: np.ndarray, var: np.ndarray) -> float:
    """PCR via torch.linalg.lstsq (GPU-accelerated when CUDA is available)."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(np.ascontiguousarray(X_pca), dtype=torch.float32, device=device)
    cov_t = torch.tensor(np.ascontiguousarray(covariate), dtype=torch.float32, device=device)
    var_t = torch.tensor(np.ascontiguousarray(var), dtype=torch.float32, device=device)

    solution = torch.linalg.lstsq(cov_t, X_t).solution
    predicted = cov_t @ solution
    residual_sum = ((X_t - predicted) ** 2).sum(dim=0)
    total_sum = ((X_t - X_t.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = torch.clamp(1.0 - residual_sum / total_sum.clamp(min=1e-30), min=0.0)

    pcr = (r2.ravel() @ var_t) / var_t.sum().clamp(min=1e-30)
    return float(pcr.cpu().item())


def principal_component_regression(
    X: NdArray,
    covariate: NdArray,
    categorical: bool = False,
    n_components: int | None = None,
    flavor: str = "auto",
) -> float:
    """Principal component regression (PCR) :cite:p:`buttner2018`.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    covariate
        Array of shape (n_cells,) or (n_cells, 1) representing batch/covariate values.
    categorical
        If True, batch will be treated as categorical and one-hot encoded.
    n_components:
        Number of components to compute, passed into :func:`~scib_metrics.utils.pca`.
        If None, all components are used.
    flavor
        Backend to use. ``"auto"`` (default) and ``"torch"`` use PyTorch (GPU if
        CUDA is available, otherwise CPU). ``"jax"`` uses JAX.

    Returns
    -------
    pcr: float
        Principal component regression using the first n_components principal components.
    """
    if len(X.shape) != 2:
        raise ValueError("Dimension mismatch: X must be 2-dimensional.")
    if X.shape[0] != covariate.shape[0]:
        raise ValueError("Dimension mismatch: X and batch must have the same number of samples.")
    if categorical:
        covariate = np.asarray(pd.Categorical(covariate).codes)
    else:
        covariate = np.asarray(covariate)

    use_torch = flavor in ("torch", "auto")

    if use_torch:
        try:
            if categorical:
                n_classes = int(covariate.max()) + 1
                covariate_np = np.eye(n_classes, dtype=np.float32)[covariate]
            else:
                covariate_np = covariate.astype(np.float32).reshape((covariate.shape[0], 1))

            pca_results = pca(X, n_components=n_components, flavor=flavor)
            covariate_np = covariate_np - covariate_np.mean(axis=0)
            return _pcr_torch(pca_results.coordinates, covariate_np, pca_results.variance)
        except ImportError:
            pass  # torch not installed — fall through to JAX
        except RuntimeError as e:
            if _is_oom_error(e):
                _flush_gpu_memory()
                warnings.warn(
                    "CUDA out-of-memory during PCR. Falling back to JAX CPU backend.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise

    # JAX path
    covariate = one_hot(covariate) if categorical else covariate.reshape((covariate.shape[0], 1))

    pca_results = pca(X, n_components=n_components, flavor="jax")

    # Center inputs for no intercept
    covariate = covariate - jnp.mean(covariate, axis=0)
    pcr = _pcr(pca_results.coordinates, covariate, pca_results.variance)
    return float(pcr)


@jit
def _pcr(
    X_pca: NdArray,
    covariate: NdArray,
    var: NdArray,
) -> NdArray:
    """Principal component regression.

    Parameters
    ----------
    X_pca
        Array of shape (n_cells, n_components) containing PCA coordinates. Must be standardized.
    covariate
        Array of shape (n_cells, 1) or (n_cells, n_classes) containing batch/covariate values. Must be standardized
        if not categorical (one-hot).
    var
        Array of shape (n_components,) containing the explained variance of each PC.
    """
    residual_sum = jnp.linalg.lstsq(covariate, X_pca)[1]
    total_sum = jnp.sum((X_pca - jnp.mean(X_pca, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = jnp.maximum(0, 1 - residual_sum / total_sum)

    return jnp.dot(jnp.ravel(r2), var) / jnp.sum(var)
