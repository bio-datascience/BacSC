import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sps

import tools.util as ut


def countsplit_adata(adata, data_dist="NB", beta_key="nb_overdisp", mean_key="nb_mean", epsilon=0.5, min_cells=2, min_genes=2, max_counts=100000, min_counts=1,
                        layer=None, seed=None):

    count_data = ut.convert_to_dense_counts(adata, layer)

    count_data = count_data.astype(int)
    n, p = count_data.shape

    if data_dist == "NB":
        # Create train and test data with Dirichlet-Multinomial count splitting
        beta = adata.var[beta_key]
        dir = np.concatenate(
            [np.random.default_rng(seed).dirichlet(alpha=[epsilon * beta[j], (1 - epsilon) * beta[j]], size=(n, 1)) for j in
             range(p)], axis=1)
        X_res = np.random.default_rng(seed).multinomial(count_data, dir)
        X_train = X_res[:, :, 0]
        X_test = X_res[:, :, 1]
    elif data_dist == "Poi":
        # Create train and test data with Poisson count splitting
        X_train = np.random.default_rng(seed).binomial(count_data, epsilon)
        X_test = count_data - X_train
    else:
        raise NotImplementedError("Only 'NB' and 'Poi' are implemented for 'data_dist'")

    # Make anndata object for training data and filter
    adata_train = ad.AnnData(X=sps.csr_matrix(X_train), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.filter_cells(adata_train, min_genes=min_genes)
    sc.pp.filter_genes(adata_train, min_cells=min_cells)
    sc.pp.filter_cells(adata_train, min_counts=min_counts)
    sc.pp.filter_cells(adata_train, max_counts=max_counts)

    # filter test data to include same cells/features as training data
    adata_test = ad.AnnData(X=sps.csr_matrix(np.array(X_test)), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.filter_cells(adata_test, min_genes=min_genes)
    sc.pp.filter_genes(adata_test, min_cells=min_cells)

    cells_ind = adata_train.obs.index.intersection(adata_test.obs.index)
    features_ind = adata_train.var.index.intersection(adata_test.var.index)

    adata_test = adata_test[cells_ind, features_ind]
    adata_train = adata_train[cells_ind, features_ind]

    sc.pp.calculate_qc_metrics(adata_train, var_type="PCs", percent_top=None, log1p=True, inplace=True)
    adata_train.layers["counts"] = adata_train.X.copy()

    sc.pp.calculate_qc_metrics(adata_test, var_type="PCs", percent_top=None, log1p=True, inplace=True)
    adata_test.layers["counts"] = adata_test.X.copy()



    if data_dist == "NB":
        adata_train.var["nb_mean"] = adata_train.var[mean_key] * epsilon
        adata_train.var["nb_overdisp"] = adata_train.var[beta_key] * epsilon
        adata_test.var["nb_mean"] = adata_test.var[mean_key] * (1 - epsilon)
        adata_test.var["nb_overdisp"] = adata_test.var[beta_key] * (1 - epsilon)

    return adata_train, adata_test


def select_n_pcs_countsplit(train_data, test_data, max_k=20):

    def approx_k(U, s, V, k):
        s_ = s[:k]
        u_ = U[:, :k]
        v_ = V[:k, :]
        ret = u_ @ np.diag(s_) @ v_
        return ret

    X_train = ut.convert_to_dense_counts(train_data, convert_to_integer=False)
    X_test = ut.convert_to_dense_counts(test_data, convert_to_integer=False)

    u_train, s_train, v_train = np.linalg.svd(X_train, full_matrices=False)
    k_devs = [np.linalg.norm((X_test - approx_k(u_train, s_train, v_train, k+1)), ord='fro') ** 2 for k in
              range(max_k)]

    opt_k = np.where(k_devs == np.min(k_devs))[0][0] + 1

    return k_devs, opt_k
