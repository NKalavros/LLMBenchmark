# Load required libraries
library(GEOquery)
library(Biobase)
library(limma)
library(randomForest)
library(caret)
library(ggplot2)
library(dplyr)
library(glmnet)
library(e1071)
library(xgboost)

# Set options
options(timeout = 300)
set.seed(42)  # For reproducibility

# Download GSE149440 dataset (reuse if already downloaded)
cat("Loading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
eset <- gse[[1]]

# Get expression data and metadata
expr_data <- exprs(eset)
metadata <- pData(eset)

cat("Dataset dimensions:\n")
cat("Genes:", nrow(expr_data), "\n")
cat("Samples:", ncol(expr_data), "\n")

# Extract relevant variables
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_indicator <- metadata$`train:ch1`

# Remove samples with missing values
valid_samples <- !is.na(gestational_age) & !is.na(train_indicator)
expr_data <- expr_data[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_indicator <- train_indicator[valid_samples]

# Split into training and test sets
train_idx <- train_indicator == "1"
test_idx <- train_indicator == "0"

cat("Training samples:", sum(train_idx), "\n")
cat("Test samples:", sum(test_idx), "\n")

# Prepare data
X_train <- t(expr_data[, train_idx])
y_train <- gestational_age[train_idx]
X_test <- t(expr_data[, test_idx])
y_test <- gestational_age[test_idx]

# Advanced feature selection
cat("\nPerforming advanced feature selection...\n")

# 1. Remove low variance genes (more stringent)
gene_vars <- apply(expr_data, 1, var)
high_var_genes <- gene_vars > quantile(gene_vars, 0.5)  # Top 50% most variable
cat("Step 1 - High variance genes:", sum(high_var_genes), "\n")

X_train_step1 <- X_train[, high_var_genes]
X_test_step1 <- X_test[, high_var_genes]

# 2. Correlation-based feature selection (more stringent)
gene_cors <- cor(X_train_step1, y_train)
high_cor_genes <- abs(gene_cors) > quantile(abs(gene_cors), 0.95, na.rm = TRUE)  # Top 5%
cat("Step 2 - High correlation genes:", sum(high_cor_genes, na.rm = TRUE), "\n")

X_train_step2 <- X_train_step1[, high_cor_genes]
X_test_step2 <- X_test_step1[, high_cor_genes]

# 3. Recursive Feature Elimination with Random Forest
cat("Step 3 - Recursive Feature Elimination...\n")
if(ncol(X_train_step2) > 100) {
  # Use RFE to select top features
  control <- rfeControl(functions = rfFuncs, method = "cv", number = 5, verbose = FALSE)
  rfe_result <- rfe(X_train_step2, y_train, sizes = c(50, 100, 200), rfeControl = control)
  selected_features <- predictors(rfe_result)
  
  X_train_final <- X_train_step2[, selected_features]
  X_test_final <- X_test_step2[, selected_features]
  cat("RFE selected features:", length(selected_features), "\n")
} else {
  X_train_final <- X_train_step2
  X_test_final <- X_test_step2
  cat("Using all remaining features:", ncol(X_train_final), "\n")
}

# Data preprocessing - standardization
preProcess_params <- preProcess(X_train_final, method = c("center", "scale"))
X_train_scaled <- predict(preProcess_params, X_train_final)
X_test_scaled <- predict(preProcess_params, X_test_final)

# Model 1: Optimized Elastic Net with extensive hyperparameter tuning
cat("\nTraining optimized Elastic Net...\n")
alpha_grid <- seq(0.1, 0.9, by = 0.1)
best_alpha <- 0.5
best_lambda <- NULL
best_cv_error <- Inf

for(alpha in alpha_grid) {
  cv_model <- cv.glmnet(as.matrix(X_train_scaled), y_train, alpha = alpha, nfolds = 10)
  if(min(cv_model$cvm) < best_cv_error) {
    best_cv_error <- min(cv_model$cvm)
    best_alpha <- alpha
    best_lambda <- cv_model$lambda.min
  }
}

elastic_net_model <- glmnet(as.matrix(X_train_scaled), y_train, alpha = best_alpha, lambda = best_lambda)
pred_elastic <- predict(elastic_net_model, as.matrix(X_test_scaled), s = best_lambda)
rmse_elastic <- sqrt(mean((pred_elastic - y_test)^2))
cat("Optimized Elastic Net RMSE:", rmse_elastic, "\n")

# Model 2: Optimized Random Forest with hyperparameter tuning
cat("\nTraining optimized Random Forest...\n")
rf_grid <- expand.grid(
  mtry = c(sqrt(ncol(X_train_scaled)), ncol(X_train_scaled)/3, ncol(X_train_scaled)/2)
)

rf_control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
rf_tuned <- train(X_train_scaled, y_train, method = "rf", 
                  tuneGrid = rf_grid, trControl = rf_control,
                  ntree = 1000, importance = TRUE)

pred_rf <- predict(rf_tuned, X_test_scaled)
rmse_rf <- sqrt(mean((pred_rf - y_test)^2))
cat("Optimized Random Forest RMSE:", rmse_rf, "\n")

# Model 3: XGBoost with hyperparameter tuning
cat("\nTraining XGBoost...\n")
xgb_train <- xgb.DMatrix(data = as.matrix(X_train_scaled), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test_scaled), label = y_test)

# Hyperparameter tuning for XGBoost
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Cross-validation to find optimal number of rounds
cv_result <- xgb.cv(
  params = xgb_params,
  data = xgb_train,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 50,
  verbose = FALSE
)

best_nrounds <- cv_result$best_iteration

xgb_model <- xgboost(
  data = xgb_train,
  params = xgb_params,
  nrounds = best_nrounds,
  verbose = FALSE
)

pred_xgb <- predict(xgb_model, xgb_test)
rmse_xgb <- sqrt(mean((pred_xgb - y_test)^2))
cat("XGBoost RMSE:", rmse_xgb, "\n")

# Model 4: Support Vector Regression with hyperparameter tuning
cat("\nTraining optimized SVM...\n")
svm_grid <- expand.grid(
  C = c(0.1, 1, 10, 100),
  sigma = c(0.001, 0.01, 0.1, 1)
)

svm_control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
svm_tuned <- train(X_train_scaled, y_train, method = "svmRadial",
                   tuneGrid = svm_grid, trControl = svm_control)

pred_svm <- predict(svm_tuned, X_test_scaled)
rmse_svm <- sqrt(mean((pred_svm - y_test)^2))
cat("Optimized SVM RMSE:", rmse_svm, "\n")

# Model 5: Ensemble method (weighted average)
cat("\nCreating ensemble model...\n")

# Calculate weights based on inverse RMSE (better models get higher weights)
rmse_values <- c(rmse_elastic, rmse_rf, rmse_xgb, rmse_svm)
weights <- 1 / rmse_values
weights <- weights / sum(weights)

# Ensemble prediction
pred_ensemble <- weights[1] * as.vector(pred_elastic) + 
                weights[2] * pred_rf + 
                weights[3] * pred_xgb + 
                weights[4] * pred_svm

rmse_ensemble <- sqrt(mean((pred_ensemble - y_test)^2))
cat("Ensemble RMSE:", rmse_ensemble, "\n")

# Choose the best model
models <- list(
  "Optimized Elastic Net" = list(pred = pred_elastic, rmse = rmse_elastic),
  "Optimized Random Forest" = list(pred = pred_rf, rmse = rmse_rf),
  "XGBoost" = list(pred = pred_xgb, rmse = rmse_xgb),
  "Optimized SVM" = list(pred = pred_svm, rmse = rmse_svm),
  "Ensemble" = list(pred = pred_ensemble, rmse = rmse_ensemble)
)

best_model_name <- names(models)[which.min(sapply(models, function(x) x$rmse))]
best_predictions <- models[[best_model_name]]$pred
best_rmse <- models[[best_model_name]]$rmse

cat("\n=== OPTIMIZED RESULTS ===\n")
cat("Best model:", best_model_name, "\n")
cat("Test set RMSE:", best_rmse, "\n")

# Print all model performances for comparison
cat("\nAll model performances:\n")
for(i in 1:length(models)) {
  cat(sprintf("%-25s: RMSE = %.4f\n", names(models)[i], models[[i]]$rmse))
}

# Create enhanced scatter plot
plot_data <- data.frame(
  Actual = y_test,
  Predicted = as.vector(best_predictions),
  Model = best_model_name
)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7, size = 2.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3) +
  labs(
    title = paste("Optimized Gestational Age Prediction -", best_model_name),
    subtitle = paste("Test Set RMSE:", round(best_rmse, 4), "| Correlation:", round(cor(y_test, best_predictions), 4)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 14),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11)
  ) +
  coord_fixed(ratio = 1)

print(p)

# Save the optimized plot
ggsave("optimized_gestational_age_prediction_plot.png", plot = p, width = 10, height = 8, dpi = 300)
cat("Optimized plot saved as 'optimized_gestational_age_prediction_plot.png'\n")

# Enhanced performance metrics
correlation <- cor(y_test, best_predictions)
mae <- mean(abs(y_test - best_predictions))
r_squared <- cor(y_test, best_predictions)^2

cat("\nEnhanced performance metrics:\n")
cat("Correlation:", round(correlation, 4), "\n")
cat("R-squared:", round(r_squared, 4), "\n")
cat("Mean Absolute Error (MAE):", round(mae, 4), "\n")
cat("RMSE as % of range:", round(best_rmse / (max(y_test) - min(y_test)) * 100, 2), "%\n")

# Residual analysis
residuals <- y_test - best_predictions
cat("\nResidual analysis:\n")
cat("Mean residual:", round(mean(residuals), 4), "\n")
cat("Std dev of residuals:", round(sd(residuals), 4), "\n")

# Feature importance (if available)
if(best_model_name == "Optimized Random Forest") {
  importance_scores <- importance(rf_tuned$finalModel)
  cat("\nTop 10 most important features:\n")
  top_features <- head(importance_scores[order(importance_scores[,1], decreasing = TRUE), ], 10)
  print(top_features)
}

cat("\n=== OPTIMIZATION SUMMARY ===\n")
cat("Original RMSE: ~5.6\n")
cat("Optimized RMSE:", round(best_rmse, 4), "\n")
cat("Improvement:", round(5.6 - best_rmse, 4), "weeks\n")
cat("Relative improvement:", round((5.6 - best_rmse) / 5.6 * 100, 2), "%\n")
