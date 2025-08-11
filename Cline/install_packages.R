# Install required packages
# Set CRAN mirror
options(repos = c(CRAN = "https://cran.r-project.org"))

if (!require("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Install Bioconductor packages
BiocManager::install(c("GEOquery", "Biobase", "limma"))

# Install CRAN packages
install.packages(c("randomForest", "caret", "ggplot2", "dplyr", "glmnet", "e1071"))

cat("All packages installed successfully!\n")
