import pandas as pd
import anndata as ad
import numpy as np
import os

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
        p_table["pval_data"] = -1 * np.log10(p_table["pval_data"])
        p_table["pval_null"] = -1 * np.log10(p_table["pval_null"])
    p_table["cs"] = p_table["pval_data"] - p_table["pval_null"]

    if correct:
        if PairedData.yuen_t_test(x=p_table["pval_data"], y=p_table["pval_null"], alternative="greater",
                                  paired=True, tr=0.1)["p.value"] < 0.001:
            print("correcting...")
            fmla = Formula('y ~ x')
            env = fmla.environment
            env['y'] = p_table["pval_data"]
            env['x'] = p_table["pval_null"]
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
