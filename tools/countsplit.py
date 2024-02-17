import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sps
import pandas as pd
from itertools import product

import tools.util as ut

import tools.NMD as nmd
from scipy.sparse.linalg import svds
from tools.nuclear_norm_init import nuclear_norm_init


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


def select_nmd_t_params_countsplit(
    train_data,
    test_data,
    potential_ks=[20, 10, 5, 3],
    layer="counts",
    potential_betas=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    tol_over_10iters=1.0e-4,
):
    X_train = ut.convert_to_dense_counts(train_data, layer=layer)
    X_test = ut.convert_to_dense_counts(test_data, layer=layer)

    m, n = X_train.shape

    results_df = generate_dataframe(
        k=potential_ks, beta=potential_betas, val_colname="loss"
    )

    for k in potential_ks:
        print(f"################## LATENT DIM {k}")
        W0, H0 = nuclear_norm_init(X_train, m, n, k)

        for beta in potential_betas:
            print(f"################## BETA {beta}")

            _, W0, H0, _, _, _ = nmd.nmd_t(
                X_train,
                r=k,
                W0=W0,
                H0=H0,
                tol_over_10iters=tol_over_10iters,
                beta1=beta,
                verbose=False,
            )
            # TODO: IndexError: index 4 is out of bounds for axis 0 with size 4
            results_df = add_result(
                results_df,
                val_col="loss",
                val=np.linalg.norm(X_test - np.maximum(0, W0 @ H0), ord="fro"),
                k=k,
                beta=beta,
            )

    return results_df


def select_3b_params_countsplit(
    train_data,
    test_data,
    potential_ks=[20, 10, 5, 3],
    layer="counts",
    potential_betas=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    tol_over_10iters=1.0e-4,
):
    X_train = ut.convert_to_dense_counts(train_data, layer=layer)
    X_test = ut.convert_to_dense_counts(test_data, layer=layer)

    m, n = X_train.shape

    results_df = generate_dataframe(
        k=potential_ks, beta=potential_betas, val_colname="loss"
    )

    for k in potential_ks:
        print(f"################## LATENT DIM {k}")

        for beta in potential_betas:
            print(f"################## BETA {beta}")
            W0, H0 = nuclear_norm_init(X_train, m, n, k)

            _, W0, H0, _, _, _ = nmd.nmd_3b(
                X_train,
                r=k,
                W0=W0,
                H0=H0,
                tol_over_10iters=tol_over_10iters,
                beta1=beta,
                verbose=False,
            )

            results_df = add_result(
                results_df,
                val_col="loss",
                val=np.linalg.norm(X_test - np.maximum(0, W0 @ H0), ord="fro"),
                k=k,
                beta=beta,
            )

    return results_df


def select_anmd_params_countsplit(
    train_data,
    test_data,
    potential_ks=[20, 10, 5, 3],
    do_warmstart=True,
    layer="counts",
    beta=0.7,
):
    X_train = ut.convert_to_dense_counts(train_data, layer=layer)
    X_test = ut.convert_to_dense_counts(test_data, layer=layer)

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
            beta=beta,
        )
        k_devs.append(np.linalg.norm(X_test - np.maximum(0, Theta0), ord="fro"))

        if not do_warmstart:
            # intialize new theta
            W0, H0 = nuclear_norm_init(X_train, m, n, k)
            Theta0 = W0 @ H0

    opt_k = potential_ks[np.argmin(k_devs)]

    return k_devs, opt_k


def generate_dataframe(val_colname="loss", **kwargs):
    """
    Generate a pandas DataFrame with all combinations of values from the input dictionaries.

    Parameters:
        val_colname (str): Name for the empty column. Default is 'loss'.
        **kwargs: Variable number of dictionaries containing column names as keys and values to generate combinations from.

    Returns:
        pd.DataFrame: DataFrame containing all combinations of values from the input dictionaries.
                      The DataFrame has columns specified by the keys of the dictionaries,
                      and an additional empty column with the specified name.
    """
    # Extract column names and values from kwargs
    columns = list(kwargs.keys())
    values = list(kwargs.values())

    # Generate all combinations of values
    combinations = list(product(*values))

    # Create DataFrame with columns based on column names
    df = pd.DataFrame(combinations, columns=columns)
    df[val_colname] = pd.Series(dtype=float)  # Creating an empty column

    return df


def add_result(df, val_col, val, **kwargs):
    """
    Assign a value to the specified column based on the values of the input lists.

    Parameters:
        df (pd.DataFrame): DataFrame to update.
        val_col (str): Name of the column to assign the value.
        val: Value to assign.
        **kwargs: Keyword arguments representing the values of the input lists.

    Returns:
        pd.DataFrame: Updated DataFrame with the specified value assigned to the column.
    """
    # Filter DataFrame based on provided values
    mask = pd.Series(True, index=df.index)
    for key, value in kwargs.items():
        mask &= df[key] == value

    # Assign value to specified column
    df.loc[mask, val_col] = val

    return df
