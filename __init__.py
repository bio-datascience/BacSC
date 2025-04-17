"""
BacSC: A general workflow for bacterial single-cell RNA sequencing data analysis
"""

__version__ = "0.1.0"

from .tools.ClusterDE import (
    call_de,
    construct_synthetic_null,
    generate_nb_data_copula,
)

from .tools.NB_est import (
    estimate_overdisp_nb,
    fit_nbinom,
    negbin_mean_to_numpy,
)

from .tools.scTransform import SCTransform
from .tools.countsplit import countsplit_adata, select_n_pcs_countsplit
from .tools.clustering_opt import (
    modularity,
    cpm,
    mod_weighted,
    cluster_train_test,
    find_optimal_clustering_resolution,
)
from .tools.scDEED import (
    scDEED,
    create_permuted_data_scdeed,
    embed_data_scDEED,
    calculate_reliability_scores,
    scdeed_parameter_selection,
)
from .tools.util import (
    convert_to_dense_counts,
    filter_outliers,
    is_outlier,
    rotate_umap,
    find_opt_umap_rotation,
)
from .tools.util_probe import (
    prep_probe_BacSC_data,
    grouped_var_agg,
)
