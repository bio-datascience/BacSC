import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sps

import tools.util as ut

import tools.NMD as nmd
from scipy.sparse.linalg import svds


def countsplit_adata(
    adata,
    data_dist="NB",
    beta_key="nb_overdisp",
    mean_key="nb_mean",
    epsilon=0.5,
    min_cells=2,
    min_genes=2,
    max_counts=100000,
    min_counts=1,
    layer=None,
    seed=None,
):

    count_data = ut.convert_to_dense_counts(adata, layer)

    count_data = count_data.astype(int)
    n, p = count_data.shape

    if data_dist == "NB":
        # Create train and test data with Dirichlet-Multinomial count splitting
        beta = adata.var[beta_key]
        dir = np.concatenate(
            [
                np.random.default_rng(seed).dirichlet(
                    alpha=[epsilon * beta[j], (1 - epsilon) * beta[j]], size=(n, 1)
                )
                for j in range(p)
            ],
            axis=1,
        )
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
    adata_train = ad.AnnData(
        X=sps.csr_matrix(X_train), obs=adata.obs.copy(), var=adata.var.copy()
    )
    sc.pp.filter_cells(adata_train, min_genes=min_genes)
    sc.pp.filter_genes(adata_train, min_cells=min_cells)
    sc.pp.filter_cells(adata_train, min_counts=min_counts)
    sc.pp.filter_cells(adata_train, max_counts=max_counts)

    # filter test data to include same cells/features as training data
    adata_test = ad.AnnData(
        X=sps.csr_matrix(np.array(X_test)), obs=adata.obs.copy(), var=adata.var.copy()
    )
    sc.pp.filter_cells(adata_test, min_genes=min_genes)
    sc.pp.filter_genes(adata_test, min_cells=min_cells)

    cells_ind = adata_train.obs.index.intersection(adata_test.obs.index)
    features_ind = adata_train.var.index.intersection(adata_test.var.index)

    adata_test = adata_test[cells_ind, features_ind]
    adata_train = adata_train[cells_ind, features_ind]

    sc.pp.calculate_qc_metrics(
        adata_train, var_type="PCs", percent_top=None, log1p=True, inplace=True
    )
    adata_train.layers["counts"] = adata_train.X.copy()

    sc.pp.calculate_qc_metrics(
        adata_test, var_type="PCs", percent_top=None, log1p=True, inplace=True
    )
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

    X_train = ut.convert_to_dense_counts(train_data)
    X_test = ut.convert_to_dense_counts(test_data)

    u_train, s_train, v_train = np.linalg.svd(X_train, full_matrices=False)
    k_devs = [
        np.linalg.norm((X_test - approx_k(u_train, s_train, v_train, k + 1)), ord="fro")
        ** 2
        for k in range(max_k)
    ]

    opt_k = np.where(k_devs == np.min(k_devs))[0][0] + 1

    return k_devs, opt_k


def select_3b_latentdim_countsplit(
    train_data, test_data, potential_ks=[20, 10, 5, 3], do_warmstart=True
):
    X_train = ut.convert_to_dense_counts(train_data)
    X_test = ut.convert_to_dense_counts(test_data)

    m, n = X_train.shape
    W0, H0 = nuclear_norm_init(X_train, m, n, potential_ks[0])

    k_devs = []

    for k in potential_ks:
        print(f"################## LATENT DIM {k}")
        _, W0, H0, _, _, _ = nmd.nmd_3b(
            X_train, r=k, W0=W0, H0=H0, tol_over_10iters=1.0e-3
        )

        k_devs.append(np.linalg.norm(X_test - np.maximum(0, W0 @ H0), ord="fro"))

        if not do_warmstart:
            W0, H0 = nuclear_norm_init(X_train, m, n, k)

    opt_k = potential_ks[np.argmin(k_devs)]

    return k_devs, opt_k


def select_anmd_latentdim_countsplit(
    train_data,
    test_data,
    potential_ks=[20, 10, 5, 3],
    do_warmstart=True,
):
    X_train = ut.convert_to_dense_counts(train_data)
    X_test = ut.convert_to_dense_counts(test_data)

    m, n = X_train.shape
    W0, H0 = nuclear_norm_init(X_train, m, n, potential_ks[0])
    Theta0 = W0 @ H0

    k_devs = []

    for k in potential_ks:
        print(f"################## LATENT DIM {k}")
        Theta0, _, _, _ = nmd.a_nmd(
            X_train,
            r=k,
            Theta0=Theta0,
            tol_over_10iters=1.0e-4,
            gamma=1.2,
            gamma_bar=1.1,
            eta=0.4,
            beta=0.7,
        )
        k_devs.append(np.linalg.norm(X_test - np.maximum(0, Theta0), ord="fro"))

        if not do_warmstart:
            # intialize new theta
            W0, H0 = nuclear_norm_init(X_train, m, n, k)
            Theta0 = W0 @ H0

    opt_k = potential_ks[np.argmin(k_devs)]

    return k_devs, opt_k


def nuclear_norm_init(
    X: np.ndarray, m: int, n: int, r: int, seed: int = 1293871, verbose: bool = False
) -> (np.ndarray, np.ndarray):
    rng = np.random.default_rng(seed=seed)
    Theta1 = rng.standard_normal(size=(m, n))
    Theta2, _ = nmd_nuclear_bt(X, Theta1, 3, verbose=verbose)
    ua, sa, va = np.linalg.svd(Theta2, full_matrices=False)
    sa = np.diag(sa)[:r, :r]
    W0 = ua[:, :r]
    H0 = sa @ va[:r, :]
    return W0, H0


def nmd_nuclear_bt(
    X: np.typing.ArrayLike,
    Theta: np.typing.ArrayLike,
    max_iter: int,
    verbose: bool = False,
) -> (np.ndarray, list[float]):
    assert np.all(X >= 0)

    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    alpha = 1 / 1**0.1  # Initial choice for alpha
    Theta[x_is_pos] = X[x_is_pos]  # Set the fixed components of Theta
    Theta[x_is_zero] = np.minimum(0, Theta[x_is_zero])

    nuclear_norms = []

    for i in range(max_iter):
        if verbose:
            print(f"Iteration { i + 1 } out of { max_iter }")
        U, D, Vt = np.linalg.svd(Theta, full_matrices=False)
        nuclear_norms.append(np.sum(np.diag(D)))  # Nuclear norm eval

        # backtracking
        if i > 0 and nuclear_norms[i] < nuclear_norms[i - 1]:
            alpha *= 1.2
        else:
            alpha *= 0.7

        # Update Theta
        # Theta = Theta - alpha * (U @ Vt)
        Theta -= alpha * (U @ Vt)

        # Project Theta
        Theta[x_is_pos] = X[x_is_pos]
        Theta[x_is_zero] = np.minimum(0, Theta[x_is_zero])

    return Theta, nuclear_norms
