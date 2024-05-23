import pandas as pd
import anndata as ad
import numpy as np
import os
import tools.NB_est as nb
import tools.util as ut
from scipy.stats import nbinom, norm, poisson, spearmanr
from scipy.optimize import golden

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
r_path = "/Library/Frameworks/R.framework/Resources/bin"
os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

"""
import rpy2.robjects as rp
from rpy2.robjects import numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()
import rpy2.robjects.packages as rpackages
PairedData = rpackages.importr("PairedData")
MASS = rpackages.importr("MASS")
from rpy2.robjects import Formula
"""

import warnings

# warnings.filterwarnings("ignore")


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

    if nlog:
        p_table["cs"] = p_table["pval_trafo_data"] - p_table["pval_trafo_null"]
    else:
        p_table["cs"] = p_table["pval_trafo_null"] - p_table["pval_trafo_data"]

    if correct:
        if PairedData.yuen_t_test(x=p_table["pval_trafo_data"], y=p_table["pval_trafo_null"], alternative="greater",
                                  paired=True, tr=0.1).rx2("p.value") < 0.001:
            print("correcting...")
            fmla = Formula('y ~ x')
            env = fmla.environment
            env['y'] = p_table["pval_trafo_data"]
            env['x'] = p_table["pval_trafo_null"]
            fit = MASS.rlm(fmla, maxit=100)
            p_table["cs"] = fit.rx2("residuals")

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


def dist_ppf_selector(X, intercept, overdisp, zinf_param, impute_zero_genes=False):
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

    if impute_zero_genes:
        if all(ppf == 0):
            print("Only zero counts!")
            ppf[X == np.max(X)] = 1
        if len(ppf[ppf != 0]) < 2:
            print("One nonzero count!")
            ppf[X == sorted(X)[-2]] = 1

    return ppf


def generate_nb_data_copula(
        adata,
        R_est=None,
        rng_seed=1234,
        new_data_shape=None,
        nb_flavor="BFGS",
        auto_dist=False,
        return_R=False,
        correct_var=False,
        R_metric="corr",
        corr_factor=1,
        check_pd=True,
        min_nonzero=2
):

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

    if R_est is None:
        # Do counts-to-uniform transforamation from scDesign
        if auto_dist:
            F = np.array([dist_cdf_selector(X[:, j], means.iloc[j], overdisps.iloc[j], zinfs.iloc[j]) for j in range(p)]).T
            F1 = np.array([dist_cdf_selector(X[:, j] + 1, means.iloc[j], overdisps.iloc[j], zinfs.iloc[j]) for j in range(p)]).T
        else:
            F = np.array([nbinom.cdf(X[:, j], r.iloc[j], q.iloc[j]) for j in range(p)]).T
            F1 = np.array([nbinom.cdf(X[:, j] + 1, r.iloc[j], q.iloc[j]) for j in range(p)]).T

        V = rng.uniform(0, 1, F.shape)
        U = V * F + (1 - V) * F1
        U[U == 1] = 0.99999

        # Gaussian Copula
        U_inv = norm.ppf(U, 0, 1)

        # Estimate correlation matrix
        if R_metric == "corr":
            if correct_var:
                R_est = schaefer_strimmer(U_inv, use_corr=True)
            else:
                R_est = np.corrcoef(U_inv.T)
        elif R_metric == "cov":
            R_est = np.cov(U_inv.T)

    R_est = np.maximum(np.minimum(R_est * corr_factor, 1), -1)
    np.fill_diagonal(R_est, 1)

    if check_pd:
        eigenvals, eigenvecs = np.linalg.eigh(R_est)
        min_ev = np.min(eigenvals)
        if min_ev < 0:
            warnings.warn("R_est is not positive definite! Adjusting eigenvalues...")
            new_ev = eigenvals - (min_ev - 1e-12)
            R_est = np.real(eigenvecs @ np.diag(new_ev) @ np.linalg.inv(eigenvecs))
            # Is = np.sqrt(1 / np.diag(R_est))
            # R_est = R_est * Is.reshape(-1, 1) * Is.reshape(1, -1)

    # Generate new data and do reverse copula transform
    Z = rng.multivariate_normal(mean=np.zeros(new_data_shape[1]), cov=R_est, size=new_data_shape[0], method="eigh", check_valid="warn")
    Z_cdf = norm.cdf(Z)
    if auto_dist:
        Y_gen = np.array([dist_ppf_selector(Z_cdf[:, j], means.iloc[j], overdisps.iloc[j], zinfs.iloc[j]) for j in
                          range(new_data_shape[1])]).T
    else:
        Y_gen = np.array([nbinom.ppf(Z_cdf[:, j], r.iloc[j], q.iloc[j]) for j in range(new_data_shape[1])]).T

    nonzero_ests = [i for i in range(p) if np.sum(Y_gen[:, i] != 0) >= min_nonzero]
    Y_gen = Y_gen[:, nonzero_ests]

    # Make return anndata object
    return_data = ad.AnnData(X=Y_gen)
    if new_data_shape[0] == X.shape[0]:
        return_data.obs = pd.DataFrame(index=adata.obs.index)
    if new_data_shape[1] == X.shape[1]:
        return_data.var = pd.DataFrame(index=adata.var.index[nonzero_ests])

    if return_R:
        return return_data, R_est
    else:
        return return_data


def schaefer_strimmer(X, use_corr=False):

    n, p = X.shape

    w = (X - X.mean(axis=0, keepdims=True))**2
    w_bar = np.mean(w, axis=0)
    var_unb = (n / (n - 1)) * w_bar
    var_s = (n / (n - 1) ** 3) * np.sum((w - w_bar) ** 2, axis=0)
    med_var = np.median(var_unb)
    lambda_var = np.min((1, np.sum(var_s) / (np.sum((var_unb - med_var) ** 2))))

    del (var_unb, var_s, w, w_bar)

    X_st = np.nan_to_num(X / np.std(X, axis=0), nan=0)
    X_c_st = X_st - X_st.mean(axis=0, keepdims=True)

    w_st = X_c_st.T.dot(X_c_st)
    w_st_sq = (X_c_st**2).T.dot((X_c_st**2))
    w_bar_st = w_st/n
    var_s_st = (n / (n - 1) ** 3) * (w_st_sq - 2 * w_bar_st * w_st + n * w_bar_st**2)

    del (X_st, X_c_st)

    corr_unb_st = (n / (n - 1)) * w_bar_st

    del (w_st, w_bar_st)

    lambda_corr = np.min((1, (np.sum(var_s_st) - np.sum(np.diag(var_s_st))) / (
                np.sum(corr_unb_st ** 2) - np.sum(np.diag(corr_unb_st) ** 2))))
    del (corr_unb_st, var_s_st)

    corr_X = np.nan_to_num(np.corrcoef(X.T), nan=0)
    var_X = np.var(X, axis=0, ddof=1)

    var_shrink = lambda_var * med_var + (1 - lambda_var) * var_X
    cov_shrink = ((1 - lambda_corr) * corr_X)
    if not use_corr:
        cov_shrink *= np.sqrt(np.outer(var_shrink, var_shrink))

        np.fill_diagonal(cov_shrink, var_shrink)
    else:
        np.fill_diagonal(cov_shrink, 1)

    return cov_shrink


def select_covariance_scaling(adata, cor_cutoff=0.1, min_scale=1, max_scale=2, maxiter=20, rng_seed=1234):

    data_gen_noscale, R_est_noscale = generate_nb_data_copula(adata, rng_seed=rng_seed, nb_flavor="statsmod_auto",
                                                  auto_dist=True, correct_var=True, return_R=True, corr_factor=1,
                                                  R_est=None, check_pd=True)

    cor_orig_old = schaefer_strimmer(adata.layers["counts"].toarray(), use_corr=True)

    def opt_fun(factor):
        
        factor_cor = (np.abs(cor_orig_old) > cor_cutoff)
        cf = factor_cor * factor
        cf[cf == 0] = 1
        np.fill_diagonal(cf, 1)

        data_null_gen2, R_est_new = generate_nb_data_copula(adata, rng_seed=rng_seed, nb_flavor="statsmod_auto",
                                                               auto_dist=True, correct_var=True, return_R=True,
                                                               corr_factor=cf, R_est=R_est_noscale, check_pd=False)
        
        data_gene_nonzero = adata[:, data_null_gen2.var_names].copy()
        cor_orig = schaefer_strimmer(data_gene_nonzero.layers["counts"].toarray(), use_corr=True)
        cor_gen = schaefer_strimmer(data_null_gen2.X, use_corr=True)

        large_cor = (np.abs(cor_orig) > cor_cutoff) | (np.abs(cor_gen) > cor_cutoff)
        frob = np.linalg.norm(cor_orig[large_cor] - cor_gen[large_cor])
        if np.isnan(frob):
            frob = np.inf

        print(f"Factor: {factor} - Error: {frob}")
        return frob

    xmin, fval, funcalls = golden(opt_fun, brack=(min_scale, max_scale), full_output=True, maxiter=maxiter)

    return xmin, fval, R_est_noscale

