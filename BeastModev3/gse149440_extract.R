# Step 2: Extract expression matrix and sample metadata

# Load the ExpressionSet object
data_file <- "gse149440_expressionSet.RData"
if (!file.exists(data_file)) stop("Run gse149440_download.R first!")
load(data_file)  # loads 'gse'


# Load Biobase for exprs()
if (!requireNamespace("Biobase", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install("Biobase")
}
library(Biobase)

# Extract expression matrix (genes x samples)
expr <- exprs(gse)

# Extract sample metadata (phenotype data)
meta <- pData(gse)

# Save for next steps
save(expr, meta, file = "gse149440_expr_meta.RData")
cat("Expression matrix and metadata extracted and saved as gse149440_expr_meta.RData\n")
