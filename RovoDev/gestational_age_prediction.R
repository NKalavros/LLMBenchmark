# Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440 from Gene Expression Omnibus

# Load required libraries
library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)

# Set options
options(timeout = 300)  # Increase timeout for large downloads

# Download and load the GSE149440 dataset
cat("Downloading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)

# Extract the expression set
eset <- gse[[1]]

# Get expression data and metadata
expr_data <- exprs(eset)
metadata <- pData(eset)

cat("Dataset dimensions:\n")
cat("Genes:", nrow(expr_data), "\n")
cat("Samples:", ncol(expr_data), "\n")

# Examine the metadata structure
cat("\nMetadata columns:\n")
print(colnames(metadata))

# Check for the required columns
if(!"gestational age:ch1" %in% colnames(metadata)) {
  cat("Available columns containing 'age':\n")
  print(grep("age", colnames(metadata), value = TRUE, ignore.case = TRUE))
}

if(!"train:ch1" %in% colnames(metadata)) {
  cat("Available columns containing 'train':\n")
  print(grep("train", colnames(metadata), value = TRUE, ignore.case = TRUE))
}

# Extract gestational age and training indicators
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_indicator <- metadata$`train:ch1`

# Check for missing values
cat("\nMissing values in gestational age:", sum(is.na(gestational_age)), "\n")
cat("Unique values in train indicator:", unique(train_indicator), "\n")

# Create training and test sets
train_mask <- train_indicator == "1"
test_mask <- train_indicator == "0"

cat("Training samples:", sum(train_mask, na.rm = TRUE), "\n")
cat("Test samples:", sum(test_mask, na.rm = TRUE), "\n")

# Prepare data
train_expr <- t(expr_data[, train_mask])
train_age <- gestational_age[train_mask]
test_expr <- t(expr_data[, test_mask])
test_age <- gestational_age[test_mask]

# Remove samples with missing gestational age
train_complete <- complete.cases(train_age)
test_complete <- complete.cases(test_age)

train_expr <- train_expr[train_complete, ]
train_age <- train_age[train_complete]
test_expr <- test_expr[test_complete, ]
test_age <- test_age[test_complete]

cat("Final training samples:", nrow(train_expr), "\n")
cat("Final test samples:", nrow(test_expr), "\n")

# Feature selection: remove genes with low variance
gene_vars <- apply(train_expr, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > quantile(gene_vars, 0.75, na.rm = TRUE))

cat("Selected", length(high_var_genes), "high variance genes\n")

train_expr_filtered <- train_expr[, high_var_genes]
test_expr_filtered <- test_expr[, high_var_genes]

# Handle any remaining missing values
train_expr_filtered[is.na(train_expr_filtered)] <- 0
test_expr_filtered[is.na(test_expr_filtered)] <- 0

# Model 1: Elastic Net Regression
cat("\nTraining Elastic Net model...\n")
set.seed(42)

# Create validation split for hyperparameter tuning
val_indices <- createDataPartition(train_age, p = 0.8, list = FALSE)
train_val_expr <- train_expr_filtered[val_indices, ]
train_val_age <- train_age[val_indices]
val_expr <- train_expr_filtered[-val_indices, ]
val_age <- train_age[-val_indices]

# Train elastic net with cross-validation
cv_fit <- cv.glmnet(train_val_expr, train_val_age, alpha = 0.5, nfolds = 5)
elastic_model <- glmnet(train_expr_filtered, train_age, alpha = 0.5, lambda = cv_fit$lambda.min)

# Model 2: Random Forest
cat("Training Random Forest model...\n")
set.seed(42)
rf_model <- randomForest(train_expr_filtered, train_age, 
                        ntree = 500, 
                        mtry = sqrt(ncol(train_expr_filtered)),
                        importance = TRUE)

# Make predictions on test set
elastic_pred <- predict(elastic_model, test_expr_filtered, s = cv_fit$lambda.min)
rf_pred <- predict(rf_model, test_expr_filtered)

# Calculate RMSE for both models
rmse_elastic <- sqrt(mean((test_age - elastic_pred)^2))
rmse_rf <- sqrt(mean((test_age - rf_pred)^2))

cat("\nModel Performance (RMSE on test set):\n")
cat("Elastic Net RMSE:", round(rmse_elastic, 4), "\n")
cat("Random Forest RMSE:", round(rmse_rf, 4), "\n")

# Choose the best model
if(rmse_elastic < rmse_rf) {
  best_pred <- elastic_pred
  best_rmse <- rmse_elastic
  best_model <- "Elastic Net"
} else {
  best_pred <- rf_pred
  best_rmse <- rmse_rf
  best_model <- "Random Forest"
}

cat("\nBest model:", best_model, "with RMSE:", round(best_rmse, 4), "\n")

# Create scatter plot
plot_data <- data.frame(
  Actual = test_age,
  Predicted = as.numeric(best_pred),
  Model = best_model
)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(
    title = paste("Predicted vs Actual Gestational Age -", best_model),
    subtitle = paste("RMSE =", round(best_rmse, 4)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

print(p)

# Save the plot
ggsave("gestational_age_prediction_plot.png", p, width = 8, height = 6, dpi = 300)

# Additional model performance metrics
correlation <- cor(test_age, best_pred)
mae <- mean(abs(test_age - best_pred))

cat("\nAdditional Performance Metrics:\n")
cat("Correlation:", round(correlation, 4), "\n")
cat("Mean Absolute Error (MAE):", round(mae, 4), "\n")

# Summary statistics
cat("\nSummary Statistics:\n")
cat("Actual gestational age range:", round(min(test_age), 2), "-", round(max(test_age), 2), "weeks\n")
cat("Predicted gestational age range:", round(min(best_pred), 2), "-", round(max(best_pred), 2), "weeks\n")

cat("\nAnalysis complete!\n")