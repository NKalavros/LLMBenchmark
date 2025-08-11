# Step 1: Download and load the GSE149440 Series Matrix file in R using GEOquery

# Install GEOquery if not already installed
if (!requireNamespace("GEOquery", quietly = TRUE)) {
    install.packages("BiocManager")
    BiocManager::install("GEOquery")
}
library(GEOquery)

# Download the Series Matrix file (this may take a while due to size)
gse <- getGEO("GSE149440", GSEMatrix = TRUE, getGPL = FALSE)

# If multiple ExpressionSets are returned, use the first one
gse <- gse[[1]]

# Save for next steps
save(gse, file = "gse149440_expressionSet.RData")
cat("GSE149440 loaded and saved as gse149440_expressionSet.RData\n")
