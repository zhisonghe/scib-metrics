# scib-metrics

[![Stars][badge-stars]][link-stars]
[![PyPI][badge-pypi]][link-pypi]
[![PyPIDownloads][badge-downloads]][link-downloads]
[![Docs][badge-docs]][link-docs]
[![Build][badge-build]][link-build]
[![Coverage][badge-cov]][link-cov]
[![Discourse][badge-discourse]][link-discourse]
[![Chat][badge-zulip]][link-zulip]

[badge-stars]: https://img.shields.io/github/stars/YosefLab/scib-metrics?logo=GitHub&color=yellow
[link-stars]: https://github.com/YosefLab/scib-metrics/stargazers
[badge-pypi]: https://img.shields.io/pypi/v/scib-metrics.svg
[link-pypi]: https://pypi.org/project/scib-metrics
[badge-downloads]: https://static.pepy.tech/badge/scib-metrics
[link-downloads]: https://pepy.tech/project/scib-metrics
[badge-docs]: https://readthedocs.org/projects/scib-metrics/badge/?version=latest
[link-docs]: https://scib-metrics.readthedocs.io/en/latest/?badge=latest
[badge-build]: https://github.com/YosefLab/scib-metrics/actions/workflows/build.yaml/badge.svg
[link-build]: https://github.com/YosefLab/scib-metrics/actions/workflows/build.yaml/
[badge-cov]: https://codecov.io/gh/YosefLab/scib-metrics/branch/main/graph/badge.svg
[link-cov]: https://codecov.io/gh/YosefLab/scib-metrics
[badge-discourse]: https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org
[link-discourse]: https://discourse.scverse.org/
[badge-zulip]: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
[link-zulip]: https://scverse.zulipchat.com/

Accelerated and Python-only metrics for benchmarking single-cell integration outputs.

This repository is a fork of the original `scib-metrics` project and is adapted to support benchmarking from precomputed neighbor graphs (including sparse distance graphs) as input, in addition to embedding-based inputs.

This package contains implementations of metrics for evaluating the performance of single-cell omics data integration methods. The implementations of these metrics use [JAX](https://jax.readthedocs.io/en/latest/) when possible for jit-compilation and hardware acceleration. All implementations are in Python.

Currently we are porting metrics used in the scIB [manuscript](https://www.nature.com/articles/s41592-021-01336-8) (and [code](https://github.com/theislab/scib)). Deviations from the original implementations are documented. However, metric values from this repository should not be compared to the scIB repository.

## Fork-specific changes

This fork focuses on adapting benchmarking workflows to accept precomputed neighbor graphs as first-class inputs.

1. The benchmarker can run from precomputed graphs stored in `adata.uns`, instead of requiring only embedding matrices in `adata.obsm`.
2. Each precomputed graph key is treated as one benchmarked method/embedding.
3. Supported precomputed graph input types include `NeighborsResults` and sparse distance matrices.
4. In precomputed-graph mode, `prepare()` is skipped and only neighbor-graph-based metrics are run.
5. Metrics computed in precomputed-graph mode:
  - Bio conservation: `clisi_knn`, `nmi_ari_cluster_labels_leiden`.
  - Batch correction: `ilisi_knn`, `kbet_per_label`, `graph_connectivity`.
6. Metrics skipped in precomputed-graph mode (because they require embedding matrices):
  - Bio conservation: `isolated_labels`, `nmi_ari_cluster_labels_kmeans`, `silhouette_label`.
  - Batch correction: `bras`, `pcr_comparison`.

## Getting started

Please refer to the [documentation][link-docs].

## Benchmarker Input Modes

`Benchmarker` supports two input modes:

1. Standard embedding mode (existing behavior):
  Provide `embedding_obsm_keys`, and `prepare()` reconstructs neighbor graphs per embedding.

2. Precomputed neighbor-graph mode:
  Provide `precomputed_neighbor_uns_keys`, where each key in `adata.uns` points to one precomputed
  graph for one embedding. Each value can be either:
  - a `NeighborsResults` object, or
  - a sparse distance matrix.

In precomputed mode, `prepare()` is skipped, only neighbor-graph-based metrics are run, and each input
graph is internally reused for the `15_neighbor_res`, `50_neighbor_res`, and `90_neighbor_res` slots.

Specifically, precomputed mode runs only:
- Bio conservation: `clisi_knn`, `nmi_ari_cluster_labels_leiden`.
- Batch correction: `ilisi_knn`, `kbet_per_label`, `graph_connectivity`.

It skips:
- Bio conservation: `isolated_labels`, `nmi_ari_cluster_labels_kmeans`, `silhouette_label`.
- Batch correction: `bras`, `pcr_comparison`.

Example:

```python
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

bm = Benchmarker(
   adata,
   batch_key="batch",
   label_key="cell_type",
   precomputed_neighbor_uns_keys=["graph_emb1", "graph_emb2"],
   bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_leiden=True),
   batch_correction_metrics=BatchCorrection(),
)
bm.benchmark()
results = bm.get_results()
```

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several options to install scib-metrics:

1. Install the latest release on PyPI:

```bash
pip install scib-metrics
```

2. Install the latest development version:

```bash
pip install git+https://github.com/yoseflab/scib-metrics.git@main
```

To leverage hardware acceleration (e.g., GPU) please install the apprpriate version of [JAX](https://github.com/google/jax#installation) separately. Often this can be easier by using conda-distributed versions of JAX.

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse Discourse][link-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

Please cite:

```
Adam Gayoso, Martin Kim, Ori Kronfeld, Justin Hong, & Yosef, N. (2026). YosefLab/scib-metrics: scib-metrics 0.5.8 (v0.5.8). Zenodo. https://doi.org/10.5281/zenodo.18504367
```

In addition, please cite the original single-cell integration benchmarking work:

```
@article{luecken2022benchmarking,
  title={Benchmarking atlas-level data integration in single-cell genomics},
  author={Luecken, Malte D and B{\"u}ttner, Maren and Chaichoompu, Kridsadakorn and Danese, Anna and Interlandi, Marta and M{\"u}ller, Michaela F and Strobl, Daniel C and Zappia, Luke and Dugas, Martin and Colom{\'e}-Tatch{\'e}, Maria and others},
  journal={Nature methods},
  volume={19},
  number={1},
  pages={41--50},
  year={2022},
  publisher={Nature Publishing Group}
}
```

References for individual metrics can be also found in the corresponding documentation.

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/YosefLab/scib-metrics/issues
[changelog]: https://scib-metrics.readthedocs.io/en/latest/changelog.html
[link-api]: https://scib-metrics.readthedocs.io/en/latest/api.html
