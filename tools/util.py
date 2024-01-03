import scipy.sparse as sps
import numpy as np
import scanpy as sc
from scipy.stats import median_abs_deviation


def convert_to_dense_counts(adata, layer=None):

    if layer is None:
        count_data = adata.X
    else:
        count_data = adata.layers[layer]

    if type(count_data) == sps._csr.csr_matrix:
        count_data = count_data.toarray()
    count_data = count_data.astype(int)

    return count_data


def filter_outliers(adata, type="mad", nmads=5, min_cells=2, min_genes=2, max_counts=None):

    if not (("log1p_n_genes_by_counts" in adata.obs.index) and ("log1p_total_counts" in adata.obs.index)):
        sc.pp.calculate_qc_metrics(adata, var_type="genes", percent_top=None, log1p=True, inplace=True)

    if type == "mad":
        adata.obs["outlier"] = (
                is_outlier(adata, "log1p_total_counts", nmads=nmads)
                | is_outlier(adata, "log1p_n_genes_by_counts", nmads=nmads)
        )
    elif type == "max":
        adata.obs["outlier"] = (
            adata.obs["total_counts"] > max_counts
        )

    data_out = adata.copy()
    data_out = data_out[(~data_out.obs.outlier)].copy()

    sc.pp.filter_cells(data_out, min_genes=min_genes)
    sc.pp.filter_genes(data_out, min_cells=min_cells)
    sc.pp.calculate_qc_metrics(data_out, var_type="genes", percent_top=None, log1p=True, inplace=True)

    return data_out


def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


def rotate_umap(umap, theta, mirror=np.array([[1, 0], [0, 1]])):
    umap_mean = umap.mean(axis=0)
    X_umap_2 = umap - umap_mean
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X_umap_2 = np.matmul(np.matmul(X_umap_2, rotation_matrix), mirror) + umap_mean

    return X_umap_2


def find_opt_umap_rotation(umap_1, umap_2):
    ssd = np.sum((umap_1 - umap_2) ** 2)
    theta_opt = 0
    mirror_opt = np.array([[1, 0], [0, 1]])
    umap_2_opt = umap_2

    mirror_matrices = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, -1]]),
        np.array([[-1, 0], [0, 1]]),
    ]

    for theta in np.linspace(0, 2 * np.pi, 360):
        for m in mirror_matrices:
            umap_2_rot = rotate_umap(umap_2, theta, m)
            ssd_ = np.sum((umap_1 - umap_2_rot) ** 2)
            if ssd_ < ssd:
                ssd = ssd_
                theta_opt = theta
                mirror_opt = m
                umap_2_opt = umap_2_rot

    return theta_opt, mirror_opt, umap_2_opt, ssd
