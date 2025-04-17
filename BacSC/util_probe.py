import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sps


def prep_probe_BacSC_data(adata, gene_id_key="gene_ids", agg_fn=np.max):

    gene_ids = adata.var[gene_id_key].str.extract('(?P<gene>[\s\S]+)_(?P<id>[\s\S]+)', expand=True)
    adata.var["gene"] = gene_ids["gene"]
    adata.var["probe_id"] = gene_ids["id"]

    X_agg, agg_features = grouped_var_agg(adata, group_key="gene", agg_fn=agg_fn)

    data_pool = ad.AnnData(
        X=sps.csr_matrix(X_agg),
        obs=adata.obs,
        var=pd.DataFrame({"feature_types": "Gene Expression", "genome": "PA01"}, index=agg_features)
    )

    return data_pool


def grouped_var_agg(adata, group_key, layer=None, agg_fn=np.max):

    if layer is not None:
        getx = lambda x: x.layers[layer]
    else:
        getx = lambda x: x.X

    grouped = adata.var.groupby(group_key)
    out = np.zeros((adata.shape[0], len(grouped)), dtype=np.float64)

    tot = len(grouped.indices.keys())
    c = 0

    for group, idx in grouped.indices.items():
        c += 1
        if c % 100 == 0:
            print(f"Aggregating feature {c}/{tot}")
        X = getx(adata[:, idx])

        # print(type(X))
        try:
            out[:, c - 1] = agg_fn(X, axis=1).toarray().reshape(X.shape[0])
        except AttributeError:
            out[:, c - 1] = agg_fn(X, axis=1).reshape(X.shape[0])

    return sps.csr_array(out), list(grouped.groups.keys())
