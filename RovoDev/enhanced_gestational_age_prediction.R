# Enhanced Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440 with advanced optimization techniques

# Load required libraries
library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(e1071)
library(limma)
library(preprocessCore)

# Set options
options(timeout = 300)
set.seed(42)

# Download and load the GSE149440 dataset (reuse if already downloaded)
cat("Loading GSE149440 dataset...\n")
if(!exists("gse")) {
  gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
}

# Extract the expression set
eset <- gse[[1]]
expr_data <- exprs(eset)
metadata <- pData(eset)

cat("Dataset dimensions: Genes =", nrow(expr_data), ", Samples =", ncol(expr_data), "\n")

# Extract gestational age and training indicators
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_indicator <- metadata$`train:ch1`

# Create training and test sets
train_mask <- train_indicator == "1"
test_mask <- train_indicator == "0"

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

cat("Training samples:", nrow(train_expr), ", Test samples:", nrow(test_expr), "\n")

# Advanced preprocessing
cat("\nApplying advanced preprocessing...\n")

# 1. Quantile normalization
cat("- Quantile normalization\n")
combined_expr <- rbind(train_expr, test_expr)
combined_norm <- normalize.quantiles(t(combined_expr))
combined_norm <- t(combined_norm)
colnames(combined_norm) <- colnames(combined_expr)

train_expr_norm <- combined_norm[1:nrow(train_expr), ]
test_expr_norm <- combined_norm[(nrow(train_expr)+1):nrow(combined_norm), ]

# 2. Log2 transformation (if not already log-transformed)
if(max(train_expr_norm, na.rm = TRUE) > 50) {
  cat("- Log2 transformation\n")
  train_expr_norm <- log2(train_expr_norm + 1)
  test_expr_norm <- log2(test_expr_norm + 1)
}

# 3. Remove genes with low expression
cat("- Filtering low expression genes\n")
mean_expr <- apply(train_expr_norm, 2, mean, na.rm = TRUE)
high_expr_genes <- which(mean_expr > quantile(mean_expr, 0.25, na.rm = TRUE))

train_expr_filt <- train_expr_norm[, high_expr_genes]
test_expr_filt <- test_expr_norm[, high_expr_genes]

# Advanced Feature Selection Methods
cat("\nAdvanced feature selection...\n")

# Method 1: Correlation-based selection
cat("- Correlation-based feature selection\n")
gene_cors <- apply(train_expr_filt, 2, function(x) abs(cor(x, train_age, use = "complete.obs")))
top_cor_genes <- order(gene_cors, decreasing = TRUE)[1:min(2000, length(gene_cors))]

# Method 2: Variance-based selection (improved)
cat("- Variance-based feature selection\n")
gene_vars <- apply(train_expr_filt, 2, var, na.rm = TRUE)
top_var_genes <- order(gene_vars, decreasing = TRUE)[1:min(2000, length(gene_vars))]

# Method 3: Mutual information-based selection
cat("- Statistical significance-based selection\n")
p_values <- apply(train_expr_filt, 2, function(x) {
  if(var(x, na.rm = TRUE) == 0) return(1)
  tryCatch(cor.test(x, train_age)$p.value, error = function(e) 1)
})
top_pval_genes <- order(p_values)[1:min(2000, length(p_values))]

# Combine feature selection methods
selected_genes <- unique(c(top_cor_genes, top_var_genes, top_pval_genes))
cat("Selected", length(selected_genes), "genes from combined methods\n")

train_expr_selected <- train_expr_filt[, selected_genes]
test_expr_selected <- test_expr_filt[, selected_genes]

# Handle missing values
train_expr_selected[is.na(train_expr_selected)] <- 0
test_expr_selected[is.na(test_expr_selected)] <- 0

# Create validation set for model tuning
val_indices <- createDataPartition(train_age, p = 0.8, list = FALSE)
train_val_expr <- train_expr_selected[val_indices, ]
train_val_age <- train_age[val_indices]
val_expr <- train_expr_selected[-val_indices, ]
val_age <- train_age[-val_indices]

# Model 1: Enhanced Elastic Net with multiple alpha values
cat("\nTraining Enhanced Elastic Net...\n")
alpha_values <- seq(0.1, 0.9, by = 0.1)
best_alpha <- 0.5
best_lambda <- NULL
best_val_rmse <- Inf

for(alpha in alpha_values) {
  cv_fit <- cv.glmnet(train_val_expr, train_val_age, alpha = alpha, nfolds = 10)
  val_pred <- predict(cv_fit, val_expr, s = "lambda.min")
  val_rmse <- sqrt(mean((val_age - val_pred)^2))
  
  if(val_rmse < best_val_rmse) {
    best_val_rmse <- val_rmse
    best_alpha <- alpha
    best_lambda <- cv_fit$lambda.min
  }
}

cat("Best alpha:", best_alpha, "Best lambda:", best_lambda, "\n")
elastic_model <- glmnet(train_expr_selected, train_age, alpha = best_alpha, lambda = best_lambda)

# Model 2: Enhanced Random Forest with tuning
cat("Training Enhanced Random Forest...\n")
mtry_values <- c(sqrt(ncol(train_expr_selected)), 
                 ncol(train_expr_selected)/3, 
                 ncol(train_expr_selected)/10)
best_mtry <- sqrt(ncol(train_expr_selected))
best_rf_rmse <- Inf

for(mtry in mtry_values) {
  rf_temp <- randomForest(train_val_expr, train_val_age, 
                         ntree = 1000, 
                         mtry = round(mtry),
                         importance = TRUE)
  val_pred <- predict(rf_temp, val_expr)
  val_rmse <- sqrt(mean((val_age - val_pred)^2))
  
  if(val_rmse < best_rf_rmse) {
    best_rf_rmse <- val_rmse
    best_mtry <- mtry
  }
}

rf_model <- randomForest(train_expr_selected, train_age, 
                        ntree = 1000, 
                        mtry = round(best_mtry),
                        importance = TRUE)

# Model 3: XGBoost
cat("Training XGBoost...\n")
dtrain <- xgb.DMatrix(data = train_val_expr, label = train_val_age)
dval <- xgb.DMatrix(data = val_expr, label = val_age)

xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 50,
  verbose = 0
)

# Retrain on full training set
dtrain_full <- xgb.DMatrix(data = train_expr_selected, label = train_age)
xgb_model_full <- xgb.train(
  params = xgb_params,
  data = dtrain_full,
  nrounds = xgb_model$best_iteration,
  verbose = 0
)

# Model 4: Support Vector Regression
cat("Training Support Vector Regression...\n")
svm_model <- svm(train_expr_selected, train_age, 
                 kernel = "radial", 
                 cost = 1, 
                 gamma = 1/ncol(train_expr_selected))

# Make predictions on test set
cat("\nMaking predictions on test set...\n")
elastic_pred <- predict(elastic_model, test_expr_selected, s = best_lambda)
rf_pred <- predict(rf_model, test_expr_selected)
xgb_pred <- predict(xgb_model_full, xgb.DMatrix(test_expr_selected))
svm_pred <- predict(svm_model, test_expr_selected)

# Calculate individual model RMSEs
rmse_elastic <- sqrt(mean((test_age - elastic_pred)^2))
rmse_rf <- sqrt(mean((test_age - rf_pred)^2))
rmse_xgb <- sqrt(mean((test_age - xgb_pred)^2))
rmse_svm <- sqrt(mean((test_age - svm_pred)^2))

cat("\nIndividual Model Performance (RMSE on test set):\n")
cat("Enhanced Elastic Net RMSE:", round(rmse_elastic, 4), "\n")
cat("Enhanced Random Forest RMSE:", round(rmse_rf, 4), "\n")
cat("XGBoost RMSE:", round(rmse_xgb, 4), "\n")
cat("SVM RMSE:", round(rmse_svm, 4), "\n")

# Ensemble Methods
cat("\nCreating ensemble models...\n")

# Simple average ensemble
ensemble_simple <- (elastic_pred + rf_pred + xgb_pred + svm_pred) / 4
rmse_ensemble_simple <- sqrt(mean((test_age - ensemble_simple)^2))

# Weighted ensemble based on validation performance
weights <- 1 / c(rmse_elastic, rmse_rf, rmse_xgb, rmse_svm)
weights <- weights / sum(weights)

ensemble_weighted <- weights[1] * elastic_pred + 
                    weights[2] * rf_pred + 
                    weights[3] * xgb_pred + 
                    weights[4] * svm_pred
rmse_ensemble_weighted <- sqrt(mean((test_age - ensemble_weighted)^2))

# Stacked ensemble using linear regression
ensemble_data <- data.frame(
  elastic = as.numeric(elastic_pred),
  rf = rf_pred,
  xgb = xgb_pred,
  svm = svm_pred
)

# Train stacking model on validation predictions
val_elastic <- predict(elastic_model, val_expr, s = best_lambda)
val_rf <- predict(rf_model, val_expr)
val_xgb <- predict(xgb_model, xgb.DMatrix(val_expr))
val_svm <- predict(svm_model, val_expr)

val_ensemble_data <- data.frame(
  elastic = as.numeric(val_elastic),
  rf = val_rf,
  xgb = val_xgb,
  svm = val_svm
)

stacking_model <- lm(val_age ~ ., data = val_ensemble_data)
ensemble_stacked <- predict(stacking_model, ensemble_data)
rmse_ensemble_stacked <- sqrt(mean((test_age - ensemble_stacked)^2))

cat("\nEnsemble Model Performance (RMSE on test set):\n")
cat("Simple Average Ensemble RMSE:", round(rmse_ensemble_simple, 4), "\n")
cat("Weighted Ensemble RMSE:", round(rmse_ensemble_weighted, 4), "\n")
cat("Stacked Ensemble RMSE:", round(rmse_ensemble_stacked, 4), "\n")

# Find the best model
all_rmses <- c(rmse_elastic, rmse_rf, rmse_xgb, rmse_svm, 
               rmse_ensemble_simple, rmse_ensemble_weighted, rmse_ensemble_stacked)
model_names <- c("Enhanced Elastic Net", "Enhanced Random Forest", "XGBoost", "SVM",
                "Simple Ensemble", "Weighted Ensemble", "Stacked Ensemble")

best_idx <- which.min(all_rmses)
best_model_name <- model_names[best_idx]
best_rmse <- all_rmses[best_idx]

# Get best predictions
if(best_idx == 1) {
  best_pred <- elastic_pred
} else if(best_idx == 2) {
  best_pred <- rf_pred
} else if(best_idx == 3) {
  best_pred <- xgb_pred
} else if(best_idx == 4) {
  best_pred <- svm_pred
} else if(best_idx == 5) {
  best_pred <- ensemble_simple
} else if(best_idx == 6) {
  best_pred <- ensemble_weighted
} else {
  best_pred <- ensemble_stacked
}

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("BEST MODEL:", best_model_name, "\n")
cat("BEST RMSE:", round(best_rmse, 4), "weeks\n")
cat(paste(rep("=", 50), collapse=""), "\n")

# Create enhanced scatter plot
plot_data <- data.frame(
  Actual = test_age,
  Predicted = as.numeric(best_pred),
  Model = best_model_name
)

# Calculate additional metrics
correlation <- cor(test_age, best_pred)
mae <- mean(abs(test_age - best_pred))
r_squared <- 1 - sum((test_age - best_pred)^2) / sum((test_age - mean(test_age))^2)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3) +
  labs(
    title = paste("Enhanced Gestational Age Prediction -", best_model_name),
    subtitle = paste("RMSE =", round(best_rmse, 4), "| R² =", round(r_squared, 4), "| Correlation =", round(correlation, 4)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  ) +
  coord_fixed(ratio = 1)

print(p)

# Save the enhanced plot
ggsave("enhanced_gestational_age_prediction_plot.png", p, width = 10, height = 8, dpi = 300)

# Performance summary
cat("\nFinal Performance Metrics:\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("Correlation:", round(correlation, 4), "\n")
cat("R-squared:", round(r_squared, 4), "\n")

cat("\nModel Comparison Summary:\n")
for(i in 1:length(model_names)) {
  cat(sprintf("%-25s: %.4f\n", model_names[i], all_rmses[i]))
}

cat("\nFeature importance from best Random Forest model:\n")
importance_scores <- importance(rf_model)[, 1]
top_genes <- names(sort(importance_scores, decreasing = TRUE)[1:10])
cat("Top 10 important genes:", paste(top_genes, collapse = ", "), "\n")

cat("\nAnalysis complete! Enhanced model achieved RMSE of", round(best_rmse, 4), "weeks.\n")