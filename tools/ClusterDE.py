import pandas as pd
import anndata as ad
import numpy as np
import os
import tools.NB_est as nb
import tools.util as ut
from scipy.stats import nbinom, norm, poisson

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
r_path = "/Library/Frameworks/R.framework/Resources/bin"
os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

import rpy2.robjects as rp
from rpy2.robjects import numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()
import rpy2.robjects.packages as rpackages
PairedData = rpackages.importr("PairedData")
MASS = rpackages.importr("MASS")
from rpy2.robjects import Formula

import warnings

warnings.filterwarnings("ignore")


def construct_synthetic_null(adata, save_path):
    adata2 = ad.AnnData(X=adata.layers["counts"].toarray(), obs=adata.obs, var=adata.var)
    adata2.write(f"to_seurat.h5ad")

    rp.r(f"""
        library(Seurat)
        library(anndata)
        library(SummarizedExperiment)
        library(SingleCellExperiment)
        library(scDesign3)
        library(SeuratDisk)
        
        data <- read_h5ad('to_seurat.h5ad')
        data <- CreateSeuratObject(counts = t(data$X), meta.data = data$obs)
        mat <- GetAssayData(object = data, slot = 'counts')
        sce <- SingleCellExperiment::SingleCellExperiment(list(counts = mat))
        SummarizedExperiment::colData(sce)$cell_type <- '1'
        sce = sce
        
        newData <- scdesign3(
            sce,
            celltype = "cell_type",
            pseudotime = NULL,
            spatial = NULL,
            other_covariates = NULL,
            empirical_quantile = FALSE,
            mu_formula = "1",
            sigma_formula = "1",
            corr_formula = "1",
            family_use = "nb",
            nonzerovar = FALSE,
            n_cores = 1,
            parallelization = "pbmcmapply",
            important_feature = "auto",
            nonnegative = FALSE,
            copula = "gaussian",
            fastmvn = FALSE
        )
        
        null_data <- CreateSeuratObject(counts = newData$new_count)
        
        SaveH5Seurat(null_data, filename = '{save_path}.h5seurat', overwrite=TRUE)
    """)


def call_de(target_scores, null_scores, nlog=True, FDR=0.05, correct=False, threshold="BC", ordering=True):

    p_table = pd.merge(target_scores, null_scores, left_index=True, right_index=True)
    if nlog:
        p_table["pval_trafo_data"] = -1 * np.log10(p_table["pval_data"])
        p_table["pval_trafo_null"] = -1 * np.log10(p_table["pval_null"])
    else:
        p_table["pval_trafo_data"] = p_table["pval_data"]
        p_table["pval_trafo_null"] = p_table["pval_null"]

    p_table["cs"] = p_table["pval_trafo_data"] - p_table["pval_trafo_null"]

    if correct:
        if PairedData.yuen_t_test(x=p_table["pval_trafo_data"], y=p_table["pval_trafo_null"], alternative="greater",
                                  paired=True, tr=0.1)["p.value"] < 0.001:
            print("correcting...")
            fmla = Formula('y ~ x')
            env = fmla.environment
            env['y'] = p_table["pval_trafo_data"]
            env['x'] = p_table["pval_trafo_null"]
            fit = MASS.rlm(fmla, maxit=100)
            p_table["cs"] = fit["residuals"]

    p_table["q"] = cs2q(p_table["cs"], threshold=threshold)
    if ordering:
        p_table.sort_values("cs", inplace=True, ascending=False)

    DEGenes = p_table[p_table["q"] < FDR]

    return DEGenes, p_table


def cs2q(contrastScore, nnull=1, threshold="BC"):
    contrastScore = np.nan_to_num(contrastScore, nan=0)
    c_abs = np.abs(contrastScore[contrastScore != 0])
    c_abs = np.sort(np.unique(c_abs))

    i = 0
    emp_fdp = []

    if threshold == "BC":
        for t in c_abs:
            emp_fdp.append(
                np.min([(1 / nnull + 1 / nnull * np.sum(contrastScore <= -t)) / np.sum(contrastScore >= t), 1]))
            if i != 0:
                emp_fdp[i] = np.min([emp_fdp[i], emp_fdp[i - 1]])
            i += 1
    elif threshold == "DS":
        for t in c_abs:
            emp_fdp.append(np.min([(1 / nnull * np.sum(contrastScore <= -t)) / np.sum(contrastScore >= t), 1]))
            if i != 0:
                emp_fdp[i] = np.min([emp_fdp[i], emp_fdp[i - 1]])
            i += 1

    emp_fdp = np.array(emp_fdp)

    c_abs = c_abs[~np.isnan(emp_fdp)]
    emp_fdp = emp_fdp[~np.isnan(emp_fdp)]

    q_ind = [np.where(c_abs == x)[0] for x in contrastScore]
    q = [emp_fdp[x[0]] if len(x) > 0 else 1 for x in q_ind]
    return q


def dist_cdf_selector(X, intercept, overdisp, zinf_param):
    if overdisp == np.inf:
        cdf = poisson.cdf(X, intercept)
    else:
        r, q = nb.negbin_mean_to_numpy(intercept, overdisp)
        cdf = nbinom.cdf(X, r, q)

    if zinf_param != 0:
        cdf = zinf_param + (1 - zinf_param) * cdf

    return cdf


def dist_ppf_selector(X, intercept, overdisp, zinf_param):
    if zinf_param != 0:
        X_ = (X - zinf_param) / (1 - zinf_param)
    else:
        X_ = X

    if overdisp == np.inf:
        ppf = poisson.ppf(X_, intercept)
    else:
        r, q = nb.negbin_mean_to_numpy(intercept, overdisp)
        ppf = nbinom.ppf(X_, r, q)

    if zinf_param != 0:
        ppf[X < zinf_param] = 0

    return ppf


def generate_nb_data_copula(adata, rng_seed=1234, new_data_shape=None, nb_flavor="BFGS", auto_dist=False):

    """
    Generate synthetic null data with simplified copula approach from ClusterDE (cf. scDesign 2/3)

    :param adata: AnnData object with layer ["counts"]
    :return:
    """

    rng = np.random.default_rng(rng_seed)

    # Estimate Negative binomial parameters with BFGS implementation


    # Extract nb means, overdispersions and count data and convert parameters to scipy/numpy parametrization
    if auto_dist:
        if ("est_overdisp" not in adata.var.columns) or ("est_mean" not in adata.var.columns) or ("est_zero_inflation" not in adata.var.columns):
            nb.estimate_overdisp_nb(adata, layer="counts", flavor=nb_flavor)
        means = adata.var["est_mean"]
        overdisps = adata.var["est_overdisp"]
        zinfs = adata.var["est_zero_inflation"]
    else:
        if ("nb_overdisp" not in adata.var.columns) or ("nb_mean" not in adata.var.columns):
            nb.estimate_overdisp_nb(adata, layer="counts", flavor=nb_flavor)
        nb_means = adata.var["nb_mean"]
        nb_overdisps = adata.var["nb_overdisp"]
        r, q = nb.negbin_mean_to_numpy(nb_means, nb_overdisps)
        r = r.tolist()
        q = q.tolist()

    X = ut.convert_to_dense_counts(adata, layer="counts")

    n, p = X.shape
    if new_data_shape is None:
        new_data_shape = (n, p)

    # Do counts-to-uniform transforamation from scDesign
    if auto_dist:
        F = np.array([dist_cdf_selector(X[:, j], means[j], overdisps[j], zinfs[j]) for j in range(p)]).T
        F1 = np.array([dist_cdf_selector(X[:, j] + 1, means[j], overdisps[j], zinfs[j]) for j in range(p)]).T

    else:
        F = np.array([nbinom.cdf(X[:, j], r[j], q[j]) for j in range(p)]).T
        F1 = np.array([nbinom.cdf(X[:, j] + 1, r[j], q[j]) for j in range(p)]).T

    V = rng.uniform(0, 1, F.shape)
    U = V * F + (1 - V) * F1

    # Gaussian Copula
    U_inv = norm.ppf(U, 0, 1)

    # Estimate correlation matrix
    R_est = np.corrcoef(U_inv.T)
    R_est[R_est < 0] = 0
    R_est = np.nan_to_num(R_est, nan=0, posinf=1, neginf=-1)

    # Generate new data and do reverse copula transform
    Z = rng.multivariate_normal(mean=np.zeros(new_data_shape[1]), cov=R_est, size=new_data_shape[0])
    Z_cdf = norm.cdf(Z)
    if auto_dist:
        Y_gen = np.array([dist_ppf_selector(Z_cdf[:, j], means[j], overdisps[j], zinfs[j]) for j in
                          range(new_data_shape[1])]).T
    else:
        Y_gen = np.array([nbinom.ppf(Z_cdf[:, j], r[j], q[j]) for j in range(new_data_shape[1])]).T

    # Make return anndata object
    return_data = ad.AnnData(X=Y_gen)
    if new_data_shape == X.shape:
        return_data.obs = pd.DataFrame(index=adata.obs.index)
        return_data.var = pd.DataFrame(index=adata.var.index)

    return return_data
