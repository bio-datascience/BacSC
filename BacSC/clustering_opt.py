import numpy as np
import scanpy as sc
import pandas as pd
from scipy.special import binom


def modularity(adata, partition_key, resolution, neighbors_key="connectivities"):
    conn_matrix = adata.obsp[neighbors_key]
    partition = adata.obs[partition_key]
    cluster_names = np.unique(partition)

    m = conn_matrix.count_nonzero() / 2

    cluster_mods = []

    for c in cluster_names:
        c_ids = np.where(partition == c)[0]
        e_c = conn_matrix[c_ids, :][:, c_ids].count_nonzero() / 2
        K_c = np.sum([conn_matrix[i, :].count_nonzero() for i in c_ids])
        cm = e_c - resolution * K_c ** 2 / (2 * m)
        cluster_mods.append(cm)

    mod = (1 / (2 * m)) * np.sum(cluster_mods)

    return mod


def cpm(adata, partition_key, resolution, neighbors_key="connectivities"):
    conn_matrix = adata.obsp[neighbors_key]
    partition = adata.obs[partition_key]
    cluster_names = np.unique(partition)

    cluster_mods = []

    for c in cluster_names:
        c_ids = np.where(partition == c)[0]
        e_c = conn_matrix[c_ids, :][:, c_ids].count_nonzero() / 2
        n_c = len(c_ids)
        cm = e_c - resolution * binom(n_c, 2)

        cluster_mods.append(cm)

    return np.sum(cluster_mods)


def mod_weighted(adata, partition_key, resolution, neighbors_key="connectivities"):
    conn_matrix = adata.obsp[neighbors_key]
    partition = adata.obs[partition_key]
    cluster_names = np.unique(partition)

    m = conn_matrix.sum() / 2

    cluster_mods = []

    for c in cluster_names:
        c_ids = np.where(partition == c)[0]
        e_c = conn_matrix[c_ids, :][:, c_ids].sum() / 2
        K_c = np.sum([conn_matrix[i, :].sum() for i in c_ids])
        cm = e_c - resolution * K_c ** 2 / (2 * m)
        cluster_mods.append(cm)

    mod = (1 / (2 * m)) * np.sum(cluster_mods)

    return np.sum(mod)


def cluster_train_test(data_train, data_test, resolutions, alg="leiden", random_state=None):

    for resolution in resolutions:
        if alg == "leiden":
            sc.tl.leiden(data_train, resolution=resolution, key_added=f"leiden_res{resolution}", random_state=random_state)
            data_test.obs[f"leiden_res{resolution}"] = data_train.obs[f"leiden_res{resolution}"]
        elif alg == "louvain":
            sc.tl.louvain(data_train, resolution=resolution, key_added=f"leiden_res{resolution}", random_state=random_state)
            data_test.obs[f"leiden_res{resolution}"] = data_train.obs[f"leiden_res{resolution}"]


def find_optimal_clustering_resolution(data_train, data_test, resolutions, res_key="leiden_res", measure=modularity, random_seed=None):

    rng = np.random.default_rng(random_seed)
    mod_scores = []
    for res in resolutions:
        res_key_full = f'{res_key}{res}'
        data_test.obs[f"random_res{res}"] = data_test.obs[res_key_full]
        rng.shuffle(data_test.obs[f"random_res{res}"])

        nclust = len(np.unique(data_train.obs[res_key_full]))
        train_score = measure(data_train, res_key_full, res)
        test_score = measure(data_test, res_key_full, res)
        random_score = measure(data_test, f'random_res{res}', res)

        mod_scores.append({"resolution": res, "n_clusters": nclust, "type": "train", "score": train_score})
        mod_scores.append({"resolution": res, "n_clusters": nclust, "type": "test", "score": test_score})
        mod_scores.append({"resolution": res, "n_clusters": nclust, "type": "random", "score": random_score})

        print(
            f"resolution: {res} - clusters: {nclust} - Train: {np.round(train_score, 3)} - Test: {np.round(test_score, 3)} - Random: {np.round(random_score, 3)}"
        )

    mod_df = pd.DataFrame(mod_scores)

    mod_df_wide = mod_df.pivot(index="resolution", columns="type", values="score")
    mod_df_wide["diff_train_test"] = mod_df_wide["train"] - mod_df_wide["test"]
    mod_df_wide["diff_train_rand"] = mod_df_wide["train"] - mod_df_wide["random"]
    mod_df_wide["diff_rand_test"] = mod_df_wide["test"] - mod_df_wide["random"]

    opt_setting = mod_df_wide.loc[mod_df_wide["diff_rand_test"] == np.max(mod_df_wide["diff_rand_test"])]
    res_opt = opt_setting.reset_index()["resolution"].values[0]

    return mod_df, mod_df_wide, res_opt
