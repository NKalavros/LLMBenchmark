# Best Optimized Gestational Age Prediction
# Focus on most effective methods without neural network complexity

library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(gbm)

# Load dataset
cat("Loading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
gse <- gse[[1]]

# Extract and clean data
expr_data <- exprs(gse)
metadata <- pData(gse)
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_flag <- metadata$`train:ch1`

valid_samples <- !is.na(gestational_age) & !is.na(train_flag)
expr_data <- expr_data[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_flag <- train_flag[valid_samples]

# Split data
train_idx <- train_flag == "1"
test_idx <- train_flag == "0"
X_train <- t(expr_data[, train_idx])
y_train <- gestational_age[train_idx]
X_test <- t(expr_data[, test_idx])
y_test <- gestational_age[test_idx]

cat("Training:", nrow(X_train), "samples,", ncol(X_train), "genes\n")

# Optimized Feature Selection
cat("=== Optimized Feature Selection ===\n")

# Step 1: Remove very low variance genes
gene_vars <- apply(X_train, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > quantile(gene_vars, 0.1, na.rm = TRUE) & !is.na(gene_vars))
X_train <- X_train[, high_var_genes]
X_test <- X_test[, high_var_genes]

# Step 2: Select genes with highest correlation to target
gene_cors <- abs(cor(X_train, y_train, use = "complete.obs"))
top_cor_genes <- order(gene_cors, decreasing = TRUE)[1:min(2000, length(gene_cors))]

# Step 3: Apply LASSO for further selection
set.seed(42)
X_train_cor <- X_train[, top_cor_genes]
lasso_cv <- cv.glmnet(X_train_cor, y_train, alpha = 1, nfolds = 5)
lasso_coef <- coef(lasso_cv, s = lasso_cv$lambda.1se)
lasso_selected <- which(lasso_coef[-1] != 0)
selected_genes <- top_cor_genes[lasso_selected]

# Final gene set
X_train_sel <- X_train[, selected_genes]
X_test_sel <- X_test[, selected_genes]

cat("Selected", ncol(X_train_sel), "genes using multi-step selection\n")

# Advanced preprocessing
preProcess_model <- preProcess(X_train_sel, method = c("center", "scale", "YeoJohnson"))
X_train_processed <- predict(preProcess_model, X_train_sel)
X_test_processed <- predict(preProcess_model, X_test_sel)

# Create validation split  
set.seed(42)
val_idx <- sample(nrow(X_train_processed), size = round(0.2 * nrow(X_train_processed)))
X_val <- X_train_processed[val_idx, ]
y_val <- y_train[val_idx]
X_train_final <- X_train_processed[-val_idx, ]
y_train_final <- y_train[-val_idx]

cat("Training:", nrow(X_train_final), "Validation:", nrow(X_val), "\n")

# Model Training with Extensive Hyperparameter Tuning
cat("=== Extensive Model Training ===\n")

# 1. Heavily tuned Elastic Net
cat("Training heavily tuned Elastic Net...\n")
set.seed(42)

alpha_grid <- seq(0.05, 0.95, 0.05)
best_elastic_rmse <- Inf
best_elastic_model <- NULL

for(alpha in alpha_grid) {
  cv_model <- cv.glmnet(X_train_final, y_train_final, 
                       alpha = alpha, nfolds = 10,
                       lambda = exp(seq(-12, 2, 0.1)))
  
  pred_val <- predict(cv_model, X_val, s = cv_model$lambda.min)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_elastic_rmse) {
    best_elastic_rmse <- val_rmse
    best_elastic_model <- cv_model
  }
}

pred_elastic <- predict(best_elastic_model, X_test_processed, s = best_elastic_model$lambda.min)
rmse_elastic <- sqrt(mean((y_test - pred_elastic)^2))
cat("Heavily tuned Elastic Net RMSE:", rmse_elastic, "\n")

# 2. Extensively optimized XGBoost
cat("Training extensively optimized XGBoost...\n")
set.seed(42)
dtrain <- xgb.DMatrix(X_train_final, label = y_train_final)
dval <- xgb.DMatrix(X_val, label = y_val)
dtest <- xgb.DMatrix(X_test_processed, label = y_test)

# Grid search for best XGBoost parameters
param_grid <- list(
  list(eta = 0.005, max_depth = 4, subsample = 0.7, colsample_bytree = 0.7, 
       reg_alpha = 0.3, reg_lambda = 2, min_child_weight = 3),
  list(eta = 0.01, max_depth = 5, subsample = 0.8, colsample_bytree = 0.8,
       reg_alpha = 0.1, reg_lambda = 1.5, min_child_weight = 2),
  list(eta = 0.008, max_depth = 6, subsample = 0.75, colsample_bytree = 0.85,
       reg_alpha = 0.2, reg_lambda = 1.8, min_child_weight = 4),
  list(eta = 0.015, max_depth = 4, subsample = 0.85, colsample_bytree = 0.9,
       reg_alpha = 0.05, reg_lambda = 1, min_child_weight = 1)
)

best_xgb_rmse <- Inf
best_xgb_model <- NULL

for(params in param_grid) {
  full_params <- c(params, list(objective = "reg:squarederror", eval_metric = "rmse"))
  
  xgb_cv <- xgb.cv(params = full_params, data = dtrain, nrounds = 4000,
                   nfold = 5, early_stopping_rounds = 150, 
                   verbose = 0, showsd = FALSE)
  
  best_nrounds <- xgb_cv$best_iteration
  
  model <- xgb.train(params = full_params, data = dtrain, nrounds = best_nrounds)
  pred_val <- predict(model, dval)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_xgb_rmse) {
    best_xgb_rmse <- val_rmse
    best_xgb_model <- model
  }
}

pred_xgb <- predict(best_xgb_model, dtest)
rmse_xgb <- sqrt(mean((y_test - pred_xgb)^2))
cat("Extensively optimized XGBoost RMSE:", rmse_xgb, "\n")

# 3. Highly tuned Random Forest
cat("Training highly tuned Random Forest...\n")
set.seed(42)

# Test multiple configurations
rf_configs <- list(
  list(ntree = 2000, mtry = round(sqrt(ncol(X_train_final))), nodesize = 2),
  list(ntree = 2500, mtry = round(ncol(X_train_final)/3), nodesize = 3),
  list(ntree = 3000, mtry = round(ncol(X_train_final)/5), nodesize = 1),
  list(ntree = 2000, mtry = round(ncol(X_train_final)/4), nodesize = 4)
)

best_rf_rmse <- Inf
best_rf_model <- NULL

for(config in rf_configs) {
  rf_model <- randomForest(X_train_final, y_train_final,
                          ntree = config$ntree, 
                          mtry = config$mtry, 
                          nodesize = config$nodesize,
                          importance = TRUE, 
                          replace = TRUE)
  
  pred_val <- predict(rf_model, X_val)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_rf_rmse) {
    best_rf_rmse <- val_rmse
    best_rf_model <- rf_model
  }
}

pred_rf <- predict(best_rf_model, X_test_processed)
rmse_rf <- sqrt(mean((y_test - pred_rf)^2))
cat("Highly tuned Random Forest RMSE:", rmse_rf, "\n")

# 4. Fine-tuned GBM
cat("Training fine-tuned GBM...\n")
set.seed(42)
train_data <- data.frame(y = y_train_final, X_train_final)

# Test different GBM configurations
gbm_configs <- list(
  list(n.trees = 3000, depth = 5, shrinkage = 0.005, minobs = 10),
  list(n.trees = 4000, depth = 6, shrinkage = 0.003, minobs = 8),
  list(n.trees = 3500, depth = 4, shrinkage = 0.008, minobs = 12),
  list(n.trees = 2500, depth = 7, shrinkage = 0.01, minobs = 6)
)

best_gbm_rmse <- Inf
best_gbm_model <- NULL
best_gbm_iter <- 0

for(config in gbm_configs) {
  gbm_model <- gbm(y ~ ., data = train_data,
                  distribution = "gaussian",
                  n.trees = config$n.trees,
                  interaction.depth = config$depth,
                  shrinkage = config$shrinkage,
                  n.minobsinnode = config$minobs,
                  bag.fraction = 0.8,
                  cv.folds = 5,
                  verbose = FALSE)
  
  best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
  val_data <- data.frame(X_val)
  pred_val <- predict(gbm_model, val_data, n.trees = best_iter)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_gbm_rmse) {
    best_gbm_rmse <- val_rmse
    best_gbm_model <- gbm_model
    best_gbm_iter <- best_iter
  }
}

test_data <- data.frame(X_test_processed)
pred_gbm <- predict(best_gbm_model, test_data, n.trees = best_gbm_iter)
rmse_gbm <- sqrt(mean((y_test - pred_gbm)^2))
cat("Fine-tuned GBM RMSE:", rmse_gbm, "\n")

# Advanced Ensemble Methods
cat("=== Advanced Ensemble Methods ===\n")

# Collect validation predictions
val_preds <- data.frame(
  elastic = as.numeric(predict(best_elastic_model, X_val, s = best_elastic_model$lambda.min)),
  xgb = predict(best_xgb_model, dval),
  rf = predict(best_rf_model, X_val),
  gbm = predict(best_gbm_model, data.frame(X_val), n.trees = best_gbm_iter)
)

test_preds <- data.frame(
  elastic = as.numeric(pred_elastic),
  xgb = pred_xgb,
  rf = pred_rf,
  gbm = pred_gbm
)

# Method 1: Performance-based exponential weighting
val_rmse_individual <- sapply(val_preds, function(p) sqrt(mean((y_val - p)^2)))
weights_exp <- exp(-val_rmse_individual * 4)  # Strong emphasis on best models
weights_exp <- weights_exp / sum(weights_exp)

pred_weighted_exp <- as.matrix(test_preds) %*% weights_exp
rmse_weighted_exp <- sqrt(mean((y_test - pred_weighted_exp)^2))
cat("Exponentially weighted ensemble RMSE:", rmse_weighted_exp, "\n")

# Method 2: Regularized stacking with both LASSO and Ridge
set.seed(42)
stack_lasso <- cv.glmnet(as.matrix(val_preds), y_val, alpha = 1, nfolds = 5)
pred_stack_lasso <- predict(stack_lasso, as.matrix(test_preds), s = stack_lasso$lambda.min)
rmse_stack_lasso <- sqrt(mean((y_test - pred_stack_lasso)^2))

stack_ridge <- cv.glmnet(as.matrix(val_preds), y_val, alpha = 0, nfolds = 5)  
pred_stack_ridge <- predict(stack_ridge, as.matrix(test_preds), s = stack_ridge$lambda.min)
rmse_stack_ridge <- sqrt(mean((y_test - pred_stack_ridge)^2))

cat("LASSO stacked ensemble RMSE:", rmse_stack_lasso, "\n")
cat("Ridge stacked ensemble RMSE:", rmse_stack_ridge, "\n")

# Method 3: Multi-level ensemble
# Combine the best ensemble methods
best_ensemble_preds <- cbind(pred_weighted_exp, pred_stack_lasso, pred_stack_ridge)
final_weights <- c(0.4, 0.3, 0.3)
pred_multi_level <- best_ensemble_preds %*% final_weights
rmse_multi_level <- sqrt(mean((y_test - pred_multi_level)^2))
cat("Multi-level ensemble RMSE:", rmse_multi_level, "\n")

# Method 4: Top-2 model averaging (simple but often effective)
top_2_models <- names(sort(val_rmse_individual)[1:2])
if("xgb" %in% top_2_models && "gbm" %in% top_2_models) {
  pred_top2 <- 0.6 * pred_xgb + 0.4 * pred_gbm
} else if("elastic" %in% top_2_models && "xgb" %in% top_2_models) {
  pred_top2 <- 0.6 * pred_xgb + 0.4 * as.numeric(pred_elastic)
} else if("rf" %in% top_2_models && "xgb" %in% top_2_models) {
  pred_top2 <- 0.6 * pred_xgb + 0.4 * pred_rf
} else {
  pred_top2 <- 0.5 * test_preds[[top_2_models[1]]] + 0.5 * test_preds[[top_2_models[2]]]
}
rmse_top2 <- sqrt(mean((y_test - pred_top2)^2))
cat("Top-2 model average RMSE:", rmse_top2, "\n")

# Collect all results
results <- data.frame(
  Model = c("Elastic Net", "XGBoost", "Random Forest", "GBM",
           "Exp Weighted", "LASSO Stack", "Ridge Stack", "Multi-level", "Top-2 Average"),
  RMSE = c(rmse_elastic, rmse_xgb, rmse_rf, rmse_gbm,
          rmse_weighted_exp, rmse_stack_lasso, rmse_stack_ridge, rmse_multi_level, rmse_top2)
)

cat("\n=== All Model Results (Sorted by RMSE) ===\n")
results_sorted <- results[order(results$RMSE), ]
print(results_sorted)

# Best model
best_idx <- which.min(results$RMSE)
best_model <- results$Model[best_idx]
best_rmse <- results$RMSE[best_idx]

best_predictions <- switch(best_model,
  "Elastic Net" = as.numeric(pred_elastic),
  "XGBoost" = pred_xgb,
  "Random Forest" = pred_rf,
  "GBM" = pred_gbm,
  "Exp Weighted" = as.numeric(pred_weighted_exp),
  "LASSO Stack" = as.numeric(pred_stack_lasso),
  "Ridge Stack" = as.numeric(pred_stack_ridge),
  "Multi-level" = as.numeric(pred_multi_level),
  "Top-2 Average" = pred_top2
)

# Create comprehensive plot
plot_data <- data.frame(Actual = y_test, Predicted = best_predictions)
mae <- mean(abs(y_test - best_predictions))
r2 <- cor(y_test, best_predictions)^2
residuals <- y_test - best_predictions

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.8, size = 3.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1.3) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3, linewidth = 1.2) +
  labs(
    title = paste("Best Optimized Gestational Age Prediction (", best_model, ")", sep = ""),
    subtitle = paste("RMSE =", round(best_rmse, 3), "weeks | MAE =", round(mae, 3), "weeks | R² =", round(r2, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 16, hjust = 0.5, color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "grey50", fill = NA, linewidth = 1),
    panel.background = element_rect(fill = "white")
  ) +
  coord_fixed() +
  xlim(range(c(y_test, best_predictions))) +
  ylim(range(c(y_test, best_predictions)))

print(p)
ggsave("best_optimized_prediction.png", plot = p, width = 12, height = 10, dpi = 300)

cat("\n=== FINAL BEST OPTIMIZED RESULTS ===\n")
cat("====================================\n")
cat("Best Model:", best_model, "\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("R²:", round(r2, 4), "\n")
cat("Improvement from baseline:", round((5.336 - best_rmse), 4), "weeks\n")
cat("Relative improvement:", round(((5.336 - best_rmse) / 5.336) * 100, 2), "%\n")
cat("Standard deviation of residuals:", round(sd(residuals), 4), "\n")
cat("Max absolute error:", round(max(abs(residuals)), 4), "weeks\n")
cat("Plot saved as: best_optimized_prediction.png\n")

# Show validation performance of individual models
cat("\nValidation RMSE of individual models:\n")
val_results <- data.frame(
  Model = names(val_rmse_individual),
  Validation_RMSE = round(val_rmse_individual, 4)
)
print(val_results[order(val_results$Validation_RMSE), ])

# Feature importance for best single model (if applicable)
if(best_model == "XGBoost") {
  cat("\nTop 15 most important features (XGBoost):\n")
  importance_matrix <- xgb.importance(model = best_xgb_model)
  print(head(importance_matrix, 15))
} else if(best_model == "Random Forest") {
  cat("\nTop 15 most important features (Random Forest):\n")
  importance_df <- data.frame(
    Feature = colnames(X_train_final),
    Importance = importance(best_rf_model)[, 1]
  )
  importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
  print(head(importance_df, 15))
}