import numpy as np

from ._dataclass import NeighborsResults


def cuml_available() -> bool:
    """Return True when cuML's NearestNeighbors is importable."""
    try:
        import cuml.neighbors  # noqa: F401

        return True
    except ImportError:
        return False


def cuml_nndescent(X: np.ndarray, n_neighbors: int) -> NeighborsResults:
    """GPU nearest-neighbor search via cuML's NearestNeighbors.

    Parameters
    ----------
    X
        Data matrix of shape (n_cells, n_features).
    n_neighbors
        Number of neighbors to find (including self).

    Returns
    -------
    NeighborsResults
        Indices and distances arrays of shape (n_cells, n_neighbors).
    """
    import cuml.neighbors

    nn = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", output_type="numpy")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # cuML may return cupy arrays depending on global_output_type; coerce to numpy
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.intp)
    return NeighborsResults(indices=indices, distances=distances)
