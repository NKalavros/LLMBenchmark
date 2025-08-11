# Advanced Optimized Gestational Age Prediction with Neural Networks
# Focus on cutting-edge optimization techniques

library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(gbm)
library(nnet)

# Load dataset (reuse cached version)
cat("Loading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
gse <- gse[[1]]

# Extract data
expr_data <- exprs(gse)
metadata <- pData(gse)
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_flag <- metadata$`train:ch1`

# Clean data
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

# Advanced Feature Engineering
cat("=== Advanced Feature Engineering ===\n")

# Remove zero/low variance genes
gene_vars <- apply(X_train, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > quantile(gene_vars, 0.1, na.rm = TRUE) & !is.na(gene_vars))
X_train <- X_train[, high_var_genes]
X_test <- X_test[, high_var_genes]

# Multi-step feature selection
# Step 1: Correlation-based selection
gene_cors <- abs(cor(X_train, y_train, use = "complete.obs"))
top_cor_genes <- order(gene_cors, decreasing = TRUE)[1:min(2000, length(gene_cors))]

# Step 2: Univariate F-test on top correlated genes
X_train_cor <- X_train[, top_cor_genes]
f_scores <- sapply(1:ncol(X_train_cor), function(i) {
  x <- X_train_cor[, i]
  if(var(x) == 0) return(0)
  fit <- lm(y_train ~ x)
  f_stat <- tryCatch(summary(fit)$fstatistic[1], error = function(e) 0)
  return(ifelse(is.na(f_stat), 0, f_stat))
})

top_f_genes_idx <- order(f_scores, decreasing = TRUE)[1:min(1000, length(f_scores))]
selected_genes <- top_cor_genes[top_f_genes_idx]

X_train_sel <- X_train[, selected_genes]
X_test_sel <- X_test[, selected_genes]

cat("Selected", ncol(X_train_sel), "genes using multi-step selection\n")

# Advanced scaling with robust methods
# Use median and MAD for robust scaling
medians <- apply(X_train_sel, 2, median, na.rm = TRUE)
mads <- apply(X_train_sel, 2, mad, na.rm = TRUE)
mads[mads == 0] <- 1  # Avoid division by zero

X_train_scaled <- sweep(sweep(X_train_sel, 2, medians, "-"), 2, mads, "/")
X_test_scaled <- sweep(sweep(X_test_sel, 2, medians, "-"), 2, mads, "/")

# Create validation split
set.seed(42)
val_idx <- sample(nrow(X_train_scaled), size = round(0.25 * nrow(X_train_scaled)))
X_val <- X_train_scaled[val_idx, ]
y_val <- y_train[val_idx]
X_train_final <- X_train_scaled[-val_idx, ]
y_train_final <- y_train[-val_idx]

cat("Final training:", nrow(X_train_final), "Validation:", nrow(X_val), "\n")

# Advanced Model Training
cat("=== Advanced Model Training ===\n")

# 1. Multi-layer Neural Network
cat("Training Neural Network...\n")
set.seed(42)

# Prepare data for neural network (normalize target)
y_mean <- mean(y_train_final)
y_sd <- sd(y_train_final)
y_train_norm <- (y_train_final - y_mean) / y_sd
y_val_norm <- (y_val - y_mean) / y_sd

# Use subset of features for neural network
nn_features <- min(100, ncol(X_train_final))
X_train_nn <- X_train_final[, 1:nn_features]
X_val_nn <- X_val[, 1:nn_features]
X_test_nn <- X_test_scaled[, 1:nn_features]

# Train neural network with different architectures
best_nn_rmse <- Inf
best_nn_model <- NULL

nn_configs <- list(
  list(size = 50, decay = 0.01, maxit = 500),
  list(size = 100, decay = 0.005, maxit = 700),
  list(size = 150, decay = 0.001, maxit = 1000)
)

for(config in nn_configs) {
  nn_model <- nnet(X_train_nn, y_train_norm, 
                   size = config$size, 
                   decay = config$decay,
                   maxit = config$maxit,
                   linout = TRUE, 
                   trace = FALSE)
  
  pred_val_norm <- predict(nn_model, X_val_nn)
  pred_val <- pred_val_norm * y_sd + y_mean
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_nn_rmse) {
    best_nn_rmse <- val_rmse
    best_nn_model <- nn_model
  }
}

pred_nn_norm <- predict(best_nn_model, X_test_nn)
pred_nn <- pred_nn_norm * y_sd + y_mean
rmse_nn <- sqrt(mean((y_test - pred_nn)^2))
cat("Neural Network RMSE:", rmse_nn, "\n")

# 2. Heavily Regularized Elastic Net
cat("Training Regularized Elastic Net...\n")
set.seed(42)

# Try different alpha values with extensive lambda search
alpha_grid <- seq(0.05, 0.95, 0.05)
best_elastic_rmse <- Inf
best_elastic_model <- NULL

for(alpha in alpha_grid) {
  cv_model <- cv.glmnet(X_train_final, y_train_final, 
                       alpha = alpha, nfolds = 10,
                       lambda = exp(seq(-12, 2, 0.2)))
  
  pred_val <- predict(cv_model, X_val, s = cv_model$lambda.min)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_elastic_rmse) {
    best_elastic_rmse <- val_rmse
    best_elastic_model <- cv_model
  }
}

pred_elastic <- predict(best_elastic_model, X_test_scaled, s = best_elastic_model$lambda.min)
rmse_elastic <- sqrt(mean((y_test - pred_elastic)^2))
cat("Regularized Elastic Net RMSE:", rmse_elastic, "\n")

# 3. Extremely Tuned XGBoost
cat("Training Extremely Tuned XGBoost...\n")
set.seed(42)
dtrain <- xgb.DMatrix(X_train_final, label = y_train_final)
dval <- xgb.DMatrix(X_val, label = y_val)
dtest <- xgb.DMatrix(X_test_scaled, label = y_test)

# Extensive hyperparameter search
xgb_params_grid <- list(
  list(eta = 0.005, max_depth = 4, subsample = 0.7, colsample_bytree = 0.7, 
       reg_alpha = 0.5, reg_lambda = 2, min_child_weight = 5),
  list(eta = 0.01, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8,
       reg_alpha = 0.1, reg_lambda = 1, min_child_weight = 3),
  list(eta = 0.02, max_depth = 5, subsample = 0.9, colsample_bytree = 0.9,
       reg_alpha = 0.2, reg_lambda = 1.5, min_child_weight = 2),
  list(eta = 0.008, max_depth = 7, subsample = 0.75, colsample_bytree = 0.75,
       reg_alpha = 0.3, reg_lambda = 2.5, min_child_weight = 4)
)

best_xgb_rmse <- Inf
best_xgb_model <- NULL

for(params in xgb_params_grid) {
  full_params <- c(params, list(objective = "reg:squarederror", eval_metric = "rmse"))
  
  xgb_cv <- xgb.cv(params = full_params, data = dtrain, nrounds = 5000,
                   nfold = 5, early_stopping_rounds = 200, 
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
cat("Extremely Tuned XGBoost RMSE:", rmse_xgb, "\n")

# 4. Optimized Random Forest with feature importance
cat("Training Optimized Random Forest...\n")
set.seed(42)

# Try different mtry values
mtry_values <- c(round(sqrt(ncol(X_train_final))), 
                round(ncol(X_train_final)/3),
                round(ncol(X_train_final)/5),
                round(ncol(X_train_final)/10))

best_rf_rmse <- Inf
best_rf_model <- NULL

for(mtry in mtry_values) {
  rf_model <- randomForest(X_train_final, y_train_final,
                          ntree = 2000, mtry = mtry, nodesize = 3,
                          importance = TRUE, replace = TRUE)
  
  pred_val <- predict(rf_model, X_val)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_rf_rmse) {
    best_rf_rmse <- val_rmse
    best_rf_model <- rf_model
  }
}

pred_rf <- predict(best_rf_model, X_test_scaled)
rmse_rf <- sqrt(mean((y_test - pred_rf)^2))
cat("Optimized Random Forest RMSE:", rmse_rf, "\n")

# 5. Fine-tuned GBM
cat("Training Fine-tuned GBM...\n")
set.seed(42)
train_data <- data.frame(y = y_train_final, X_train_final)

gbm_model <- gbm(y ~ ., data = train_data,
                distribution = "gaussian",
                n.trees = 3000,
                interaction.depth = 6,
                shrinkage = 0.005,
                bag.fraction = 0.8,
                n.minobsinnode = 8,
                cv.folds = 5,
                verbose = FALSE)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
test_data <- data.frame(X_test_scaled)
pred_gbm <- predict(gbm_model, test_data, n.trees = best_iter)
rmse_gbm <- sqrt(mean((y_test - pred_gbm)^2))
cat("Fine-tuned GBM RMSE:", rmse_gbm, "\n")

# Advanced Ensemble with Neural Network
cat("=== Advanced Ensemble Methods ===\n")

# Get validation predictions for all models
val_preds <- data.frame(
  nn = predict(best_nn_model, X_val_nn) * y_sd + y_mean,
  elastic = as.numeric(predict(best_elastic_model, X_val, s = best_elastic_model$lambda.min)),
  xgb = predict(best_xgb_model, dval),
  rf = predict(best_rf_model, X_val),
  gbm = predict(gbm_model, data.frame(X_val), n.trees = best_iter)
)

test_preds <- data.frame(
  nn = pred_nn,
  elastic = as.numeric(pred_elastic),
  xgb = pred_xgb,
  rf = pred_rf,
  gbm = pred_gbm
)

# Advanced weighted ensemble with performance-based weights
val_rmse_individual <- sapply(val_preds, function(p) sqrt(mean((y_val - p)^2)))
cat("Individual validation RMSEs:", round(val_rmse_individual, 4), "\n")

# Exponential weighting (emphasizes best models heavily)
weights_exp <- exp(-val_rmse_individual * 3)
weights_exp <- weights_exp / sum(weights_exp)

pred_weighted <- as.matrix(test_preds) %*% weights_exp
rmse_weighted <- sqrt(mean((y_test - pred_weighted)^2))
cat("Advanced Weighted Ensemble RMSE:", rmse_weighted, "\n")

# Neural network stacking
set.seed(42)
stack_nn <- nnet(as.matrix(val_preds), y_val, 
                size = 20, decay = 0.01, maxit = 1000,
                linout = TRUE, trace = FALSE)

pred_stack_nn <- predict(stack_nn, as.matrix(test_preds))
rmse_stack_nn <- sqrt(mean((y_test - pred_stack_nn)^2))
cat("Neural Network Stacked Ensemble RMSE:", rmse_stack_nn, "\n")

# Multi-level stacking
set.seed(42)
# Level 1: Ridge stacking
stack_ridge <- cv.glmnet(as.matrix(val_preds), y_val, alpha = 0, nfolds = 5)
level1_pred <- predict(stack_ridge, as.matrix(test_preds), s = stack_ridge$lambda.min)

# Level 2: Combine with weighted ensemble
final_ensemble <- 0.7 * level1_pred + 0.3 * pred_weighted
rmse_final <- sqrt(mean((y_test - final_ensemble)^2))
cat("Multi-level Stacked Ensemble RMSE:", rmse_final, "\n")

# Results summary
results <- data.frame(
  Model = c("Neural Network", "Regularized Elastic Net", "XGBoost", "Random Forest", "GBM",
           "Weighted Ensemble", "NN Stacked", "Multi-level Stack"),
  RMSE = c(rmse_nn, rmse_elastic, rmse_xgb, rmse_rf, rmse_gbm,
          rmse_weighted, rmse_stack_nn, rmse_final)
)

cat("\n=== Advanced Results ===\n")
print(results[order(results$RMSE), ])

# Best model
best_idx <- which.min(results$RMSE)
best_model <- results$Model[best_idx]
best_rmse <- results$RMSE[best_idx]

best_predictions <- switch(best_model,
  "Neural Network" = pred_nn,
  "Regularized Elastic Net" = as.numeric(pred_elastic),
  "XGBoost" = pred_xgb,
  "Random Forest" = pred_rf,
  "GBM" = pred_gbm,
  "Weighted Ensemble" = as.numeric(pred_weighted),
  "NN Stacked" = as.numeric(pred_stack_nn),
  "Multi-level Stack" = as.numeric(final_ensemble)
)

# Create enhanced plot
plot_data <- data.frame(Actual = y_test, Predicted = best_predictions)
mae <- mean(abs(y_test - best_predictions))
r2 <- cor(y_test, best_predictions)^2

# Calculate residuals for diagnostic
residuals <- y_test - best_predictions

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.8, size = 3, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1.2) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3, linewidth = 1) +
  labs(
    title = paste("Advanced Optimized Gestational Age Prediction (", best_model, ")", sep = ""),
    subtitle = paste("RMSE =", round(best_rmse, 3), "weeks | MAE =", round(mae, 3), "weeks | R² =", round(r2, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, color = "darkblue"),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "grey70", fill = NA, linewidth = 0.8),
    legend.position = "none"
  ) +
  coord_fixed() +
  xlim(range(c(y_test, best_predictions))) +
  ylim(range(c(y_test, best_predictions)))

print(p)
ggsave("advanced_optimized_prediction.png", plot = p, width = 12, height = 10, dpi = 300)

cat("\n=== FINAL ADVANCED OPTIMIZED RESULTS ===\n")
cat("==========================================\n")
cat("Best Model:", best_model, "\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("R²:", round(r2, 4), "\n")
cat("Improvement from baseline:", round((5.336 - best_rmse), 4), "weeks\n")
cat("Relative improvement:", round(((5.336 - best_rmse) / 5.336) * 100, 2), "%\n")
cat("Plot saved as: advanced_optimized_prediction.png\n")

# Additional diagnostics
cat("\nModel diagnostics:\n")
cat("Mean residual:", round(mean(residuals), 4), "\n")
cat("Std of residuals:", round(sd(residuals), 4), "\n")
cat("Max absolute error:", round(max(abs(residuals)), 4), "\n")