# Load required libraries
library(GEOquery)
library(Biobase)
library(limma)
library(randomForest)
library(caret)
library(ggplot2)
library(dplyr)

# Set options
options(timeout = 300)  # Increase timeout for large downloads

# Download GSE149440 dataset
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

# Check metadata columns
cat("\nMetadata columns:\n")
print(colnames(metadata))

# Extract relevant variables
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_indicator <- metadata$`train:ch1`

# Check for missing values
cat("\nMissing values in gestational age:", sum(is.na(gestational_age)), "\n")
cat("Missing values in train indicator:", sum(is.na(train_indicator)), "\n")

# Remove samples with missing gestational age or train indicator
valid_samples <- !is.na(gestational_age) & !is.na(train_indicator)
expr_data <- expr_data[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_indicator <- train_indicator[valid_samples]

cat("After removing missing values:\n")
cat("Samples:", ncol(expr_data), "\n")

# Split into training and test sets
train_idx <- train_indicator == "1"
test_idx <- train_indicator == "0"

cat("Training samples:", sum(train_idx), "\n")
cat("Test samples:", sum(test_idx), "\n")

# Prepare training data
X_train <- t(expr_data[, train_idx])
y_train <- gestational_age[train_idx]

# Prepare test data
X_test <- t(expr_data[, test_idx])
y_test <- gestational_age[test_idx]

# Remove genes with low variance (feature selection)
cat("\nPerforming feature selection...\n")
gene_vars <- apply(expr_data, 1, var)
high_var_genes <- gene_vars > quantile(gene_vars, 0.75)  # Top 25% most variable genes
cat("Selected", sum(high_var_genes), "high variance genes\n")

X_train_filtered <- X_train[, high_var_genes]
X_test_filtered <- X_test[, high_var_genes]

# Further feature selection using correlation with target
gene_cors <- cor(X_train_filtered, y_train)
high_cor_genes <- abs(gene_cors) > quantile(abs(gene_cors), 0.9, na.rm = TRUE)  # Top 10% correlated genes
cat("Selected", sum(high_cor_genes, na.rm = TRUE), "highly correlated genes\n")

X_train_final <- X_train_filtered[, high_cor_genes]
X_test_final <- X_test_filtered[, high_cor_genes]

# Model 1: Linear Regression with regularization (Elastic Net)
cat("\nTraining Elastic Net model...\n")
library(glmnet)

# Cross-validation to find optimal lambda
cv_model <- cv.glmnet(X_train_final, y_train, alpha = 0.5, nfolds = 5)
elastic_net_model <- glmnet(X_train_final, y_train, alpha = 0.5, lambda = cv_model$lambda.min)

# Predictions
pred_elastic <- predict(elastic_net_model, X_test_final, s = cv_model$lambda.min)
rmse_elastic <- sqrt(mean((pred_elastic - y_test)^2))
cat("Elastic Net RMSE:", rmse_elastic, "\n")

# Model 2: Random Forest
cat("\nTraining Random Forest model...\n")
rf_model <- randomForest(X_train_final, y_train, ntree = 500, mtry = sqrt(ncol(X_train_final)))
pred_rf <- predict(rf_model, X_test_final)
rmse_rf <- sqrt(mean((pred_rf - y_test)^2))
cat("Random Forest RMSE:", rmse_rf, "\n")

# Model 3: Support Vector Regression
cat("\nTraining Support Vector Regression model...\n")
library(e1071)
svm_model <- svm(X_train_final, y_train, kernel = "radial", cost = 1, gamma = 1/ncol(X_train_final))
pred_svm <- predict(svm_model, X_test_final)
rmse_svm <- sqrt(mean((pred_svm - y_test)^2))
cat("SVM RMSE:", rmse_svm, "\n")

# Choose the best model
models <- list(
  "Elastic Net" = list(pred = pred_elastic, rmse = rmse_elastic),
  "Random Forest" = list(pred = pred_rf, rmse = rmse_rf),
  "SVM" = list(pred = pred_svm, rmse = rmse_svm)
)

best_model_name <- names(models)[which.min(sapply(models, function(x) x$rmse))]
best_predictions <- models[[best_model_name]]$pred
best_rmse <- models[[best_model_name]]$rmse

cat("\n=== FINAL RESULTS ===\n")
cat("Best model:", best_model_name, "\n")
cat("Test set RMSE:", best_rmse, "\n")

# Create scatter plot
plot_data <- data.frame(
  Actual = y_test,
  Predicted = as.vector(best_predictions),
  Model = best_model_name
)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(
    title = paste("Predicted vs Actual Gestational Age -", best_model_name),
    subtitle = paste("Test Set RMSE:", round(best_rmse, 3)),
    x = "Actual Gestational Age",
    y = "Predicted Gestational Age"
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
ggsave("gestational_age_prediction_plot.png", plot = p, width = 8, height = 6, dpi = 300)
cat("Plot saved as 'gestational_age_prediction_plot.png'\n")

# Additional model performance metrics
correlation <- cor(y_test, best_predictions)
mae <- mean(abs(y_test - best_predictions))

cat("\nAdditional metrics:\n")
cat("Correlation:", round(correlation, 4), "\n")
cat("Mean Absolute Error (MAE):", round(mae, 4), "\n")

# Summary statistics
cat("\nSummary statistics:\n")
cat("Actual gestational age range:", range(y_test), "\n")
cat("Predicted gestational age range:", range(best_predictions), "\n")
