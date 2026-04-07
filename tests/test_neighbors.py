import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from scib_metrics.nearest_neighbors import jax_approx_min_k, pynndescent
from scib_metrics.utils import convert_knn_graph_to_idx
from tests.utils.data import dummy_benchmarker_adata


def test_jax_neighbors():
    ad, emb_keys, _, _ = dummy_benchmarker_adata()
    output = jax_approx_min_k(ad.obsm[emb_keys[0]], 10)
    assert output.distances.shape == (ad.n_obs, 10)


@pytest.mark.parametrize("n", [5, 10, 20, 21])
def test_neighbors_results(n):
    adata, embedding_keys, *_ = dummy_benchmarker_adata()
    neigh_result = pynndescent(adata.obsm[embedding_keys[0]], n_neighbors=n)
    neigh_result = neigh_result.subset_neighbors(n=n)
    new_connect = neigh_result.knn_graph_connectivities

    sc_connect = sc.neighbors._connectivity.umap(
        neigh_result.indices[:, :n], neigh_result.distances[:, :n], n_obs=adata.n_obs, n_neighbors=n
    )

    np.testing.assert_allclose(new_connect.toarray(), sc_connect.toarray())


def test_convert_knn_graph_to_idx_variable_neighbors_trimmed():
    # Variable row-wise neighbor counts: [3, 2, 4, 2].
    X = csr_matrix(
        (
            np.array([1e-12, 0.2, 0.5, 1e-12, 0.3, 1e-12, 0.4, 0.6, 0.9, 1e-12, 0.1]),
            np.array([0, 1, 2, 1, 0, 2, 3, 1, 0, 3, 2]),
            np.array([0, 3, 5, 9, 11]),
        ),
        shape=(4, 4),
    )

    with pytest.warns(UserWarning, match="variable per-cell neighbor counts"):
        distances, indices = convert_knn_graph_to_idx(X)

    # The minimum row-wise neighbor count is 2, so all rows should be trimmed to 2 neighbors.
    assert distances.shape == (4, 2)
    assert indices.shape == (4, 2)

    # Row 0: keep nearest two distances (self and neighbor 1).
    np.testing.assert_allclose(distances[0], np.array([1e-12, 0.2]))
    np.testing.assert_array_equal(indices[0], np.array([0, 1]))

    # Row 2 originally has 4 neighbors; verify it was trimmed to nearest 2.
    np.testing.assert_allclose(distances[2], np.array([1e-12, 0.4]))
    np.testing.assert_array_equal(indices[2], np.array([2, 3]))
