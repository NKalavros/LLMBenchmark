# Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440 from Gene Expression Omnibus
# Author: Generated for bioinformatics analysis
# Date: 2025-07-28

# Load required libraries
cat("Loading required libraries...\n")

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      if (pkg %in% c("GEOquery", "limma", "Biobase")) {
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
          install.packages("BiocManager")
        }
        BiocManager::install(pkg)
      } else {
        install.packages(pkg)
      }
    }
  }
}

# Required packages
required_packages <- c("GEOquery", "limma", "Biobase", "caret", 
                      "randomForest", "glmnet", "ggplot2", "dplyr")

install_if_missing(required_packages)

# Load libraries
library(GEOquery)
library(limma)
library(Biobase)
library(caret)
library(randomForest)
library(glmnet)
library(ggplot2)
library(dplyr)

cat("Libraries loaded successfully.\n")

# Download GSE149440 dataset
cat("Downloading GSE149440 dataset from GEO...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = TRUE)

# Extract expression set
if (length(gse) > 1) {
  eset <- gse[[1]]
} else {
  eset <- gse
}

cat("Dataset downloaded successfully.\n")
cat("Dataset dimensions:", dim(exprs(eset)), "\n")

# Extract metadata
metadata <- pData(eset)
cat("Metadata columns:", colnames(metadata), "\n")

# Check for required columns
if (!"gestational age:ch1" %in% colnames(metadata)) {
  stop("Gestational age column not found in metadata")
}

if (!"train:ch1" %in% colnames(metadata)) {
  stop("Train/test split column not found in metadata")
}

# Extract gestational age and training indicators
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_indicator <- metadata$`train:ch1`

cat("Gestational age range:", range(gestational_age, na.rm = TRUE), "\n")
cat("Training samples:", sum(train_indicator == "1", na.rm = TRUE), "\n")
cat("Test samples:", sum(train_indicator == "0", na.rm = TRUE), "\n")

# Remove samples with missing data
valid_samples <- !is.na(gestational_age) & !is.na(train_indicator)
eset_clean <- eset[, valid_samples]
gestational_age_clean <- gestational_age[valid_samples]
train_indicator_clean <- train_indicator[valid_samples]

cat("Clean dataset dimensions:", dim(exprs(eset_clean)), "\n")

# Split into training and test sets
train_indices <- which(train_indicator_clean == "1")
test_indices <- which(train_indicator_clean == "0")

train_expr <- exprs(eset_clean)[, train_indices]
test_expr <- exprs(eset_clean)[, test_indices]

train_age <- gestational_age_clean[train_indices]
test_age <- gestational_age_clean[test_indices]

cat("Training set:", dim(train_expr), "\n")
cat("Test set:", dim(test_expr), "\n")

# Preprocess expression data
cat("Preprocessing expression data...\n")

# Log2 transform if needed (check if data is already log-transformed)
expr_range <- range(train_expr, na.rm = TRUE)
if (expr_range[2] > 100) {
  cat("Applying log2 transformation...\n")
  train_expr <- log2(train_expr + 1)
  test_expr <- log2(test_expr + 1)
}

# Remove features with low variance
var_threshold <- 0.1
gene_vars <- apply(train_expr, 1, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > var_threshold)

train_expr_filtered <- train_expr[high_var_genes, ]
test_expr_filtered <- test_expr[high_var_genes, ]

cat("Genes after variance filtering:", nrow(train_expr_filtered), "\n")

# Feature selection using correlation with gestational age
cat("Performing feature selection...\n")
correlations <- apply(train_expr_filtered, 1, function(x) {
  cor(x, train_age, use = "complete.obs")
})

# Select top correlated genes
top_n_genes <- 1000
top_gene_indices <- order(abs(correlations), decreasing = TRUE)[1:min(top_n_genes, length(correlations))]

train_expr_selected <- train_expr_filtered[top_gene_indices, ]
test_expr_selected <- test_expr_filtered[top_gene_indices, ]

cat("Selected genes:", nrow(train_expr_selected), "\n")

# Transpose data for modeling (samples as rows, genes as columns)
train_data <- t(train_expr_selected)
test_data <- t(test_expr_selected)

# Create data frames
train_df <- data.frame(gestational_age = train_age, train_data)
test_df <- data.frame(gestational_age = test_age, test_data)

# Remove any remaining NA values
train_df <- train_df[complete.cases(train_df), ]
test_df <- test_df[complete.cases(test_df), ]

cat("Final training samples:", nrow(train_df), "\n")
cat("Final test samples:", nrow(test_df), "\n")

# Model training and optimization
cat("Training prediction models...\n")

# Set up cross-validation
ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# Try multiple models
models <- list()

# 1. Random Forest
cat("Training Random Forest...\n")
set.seed(42)
rf_model <- train(gestational_age ~ ., 
                  data = train_df,
                  method = "rf",
                  trControl = ctrl,
                  tuneGrid = expand.grid(mtry = c(10, 20, 50)),
                  ntree = 100)
models$rf <- rf_model

# 2. Elastic Net
cat("Training Elastic Net...\n")
set.seed(42)
glmnet_model <- train(gestational_age ~ ., 
                      data = train_df,
                      method = "glmnet",
                      trControl = ctrl,
                      tuneGrid = expand.grid(alpha = c(0.1, 0.5, 0.9),
                                           lambda = c(0.001, 0.01, 0.1)))
models$glmnet <- glmnet_model

# 3. Support Vector Machine (if data size allows)
if (nrow(train_df) < 500) {
  cat("Training SVM...\n")
  set.seed(42)
  svm_model <- train(gestational_age ~ ., 
                     data = train_df,
                     method = "svmRadial",
                     trControl = ctrl,
                     tuneGrid = expand.grid(sigma = c(0.001, 0.01),
                                           C = c(1, 10)))
  models$svm <- svm_model
}

# Select best model based on cross-validation RMSE
cv_results <- sapply(models, function(m) min(m$results$RMSE, na.rm = TRUE))
best_model_name <- names(cv_results)[which.min(cv_results)]
best_model <- models[[best_model_name]]

cat("Best model:", best_model_name, "with CV RMSE:", min(cv_results), "\n")

# Make predictions on test set
test_predictions <- predict(best_model, newdata = test_df)

# Calculate RMSE on test set
test_rmse <- sqrt(mean((test_predictions - test_df$gestational_age)^2))

cat("\n=== RESULTS ===\n")
cat("Test Set RMSE:", test_rmse, "\n")
cat("Test Set R-squared:", cor(test_predictions, test_df$gestational_age)^2, "\n")

# Create scatter plot
cat("Creating scatter plot...\n")

plot_data <- data.frame(
  Actual = test_df$gestational_age,
  Predicted = test_predictions
)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
  labs(title = paste("Predicted vs Actual Gestational Age (Test Set)\nRMSE =", 
                     round(test_rmse, 3)),
       x = "Actual Gestational Age (weeks)",
       y = "Predicted Gestational Age (weeks)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(p)

# Save the plot
ggsave("gestational_age_prediction_plot.png", plot = p, width = 8, height = 6, dpi = 300)

cat("Analysis completed successfully!\n")
cat("Plot saved as 'gestational_age_prediction_plot.png'\n")

# Additional model diagnostics
cat("\n=== MODEL DIAGNOSTICS ===\n")
cat("Best model details:\n")
print(best_model)

# Feature importance (if available)
if (best_model_name == "rf") {
  importance <- varImp(best_model)
  cat("\nTop 10 most important features:\n")
  print(importance$importance[1:10, , drop = FALSE])
}