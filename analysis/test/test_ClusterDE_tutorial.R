library(Seurat)
library(anndata)
library(SeuratData)
library(SummarizedExperiment)
library(SingleCellExperiment)
library(SeuratDisk)
library(scDesign3)

data <- read_h5ad('/Users/johannes.ostner/Documents/PhD/BacSC/analysis/P_aero_S2S3/to_seurat.h5ad')
data <- CreateSeuratObject(counts = t(data$X), meta.data = data$obs)
mat <- GetAssayData(object = data, slot = 'counts')
sce <- SingleCellExperiment::SingleCellExperiment(list(counts = mat))
SummarizedExperiment::colData(sce)$cell_type <- '1'

newData <- scDesign3::scdesign3(sce,
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
                                n_cores = 4,
                                parallelization = "pbmcapply",
                                important_feature = "auto",
                                nonnegative = FALSE,
                                copula = "gaussian",
                                fastmvn = FALSE,
                                return_model=TRUE)


pbmc <- readRDS("~/Documents/PhD/data/pbmc.rds")
set.seed(123)
pbmc <- NormalizeData(object = pbmc)
pbmc <- FindVariableFeatures(object = pbmc)
pbmc <- ScaleData(object = pbmc)
pbmc <- RunPCA(object = pbmc)
pbmc <- FindNeighbors(object = pbmc)
pbmc <- FindClusters(object = pbmc, resolution = 0.3)
pbmc <- RunUMAP(object = pbmc, dims = 1:10)
p1 <- DimPlot(object = pbmc, reduction = "umap", label = TRUE) + ggtitle("Clustering result") + NoLegend()
p2 <- DimPlot(object = pbmc, reduction = "umap", group.by = "CellType", label = TRUE) + NoLegend()
p1 + p2
pbmc <- BuildClusterTree(pbmc)
pbmc_sub <- subset(x = pbmc, idents = c(2, 8))
original_markers <- FindMarkers(pbmc_sub, 
                                ident.1 = 2, 
                                ident.2 = 8, 
                                min.pct = 0, 
                                logfc.threshold = 0)
count_mat <- GetAssayData(object = pbmc_sub, slot = "counts")
set.seed(1234)
sce <- SingleCellExperiment::SingleCellExperiment(list(counts = count_mat))
SummarizedExperiment::colData(sce)$cell_type <- '1'
newData <- scDesign3::scdesign3(sce,
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
                                parallelization = "pbmcapply",
                                important_feature = "auto",
                                nonnegative = FALSE,
                                copula = "gaussian",
                                fastmvn = FALSE)
