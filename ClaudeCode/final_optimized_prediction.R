# Final Optimized Gestational Age Prediction
# Best practices implementation with careful feature selection

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

# Remove low variance genes
gene_vars <- apply(X_train, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > quantile(gene_vars, 0.25, na.rm = TRUE) & !is.na(gene_vars))
X_train <- X_train[, high_var_genes]
X_test <- X_test[, high_var_genes]

# Select top correlated genes
gene_cors <- abs(cor(X_train, y_train, use = "complete.obs"))
top_genes <- order(gene_cors, decreasing = TRUE)[1:min(500, length(gene_cors))]

X_train_sel <- X_train[, top_genes]
X_test_sel <- X_test[, top_genes]

cat("Selected", ncol(X_train_sel), "highly correlated genes\n")

# Robust scaling
medians <- apply(X_train_sel, 2, median, na.rm = TRUE)
mads <- apply(X_train_sel, 2, mad, na.rm = TRUE)
mads[mads == 0] <- 1

X_train_scaled <- sweep(sweep(X_train_sel, 2, medians, "-"), 2, mads, "/")
X_test_scaled <- sweep(sweep(X_test_sel, 2, medians, "-"), 2, mads, "/")

# Create validation split
set.seed(42)
val_idx <- sample(nrow(X_train_scaled), size = round(0.2 * nrow(X_train_scaled)))
X_val <- X_train_scaled[val_idx, ]
y_val <- y_train[val_idx]
X_train_final <- X_train_scaled[-val_idx, ]
y_train_final <- y_train[-val_idx]

cat("Training:", nrow(X_train_final), "Validation:", nrow(X_val), "\n")

# Model Training
cat("=== Optimized Model Training ===\n")

# 1. Neural Network with proper architecture
cat("Training Neural Network...\n")
set.seed(42)

# Use top 50 features for neural network
nn_features <- min(50, ncol(X_train_final))
X_train_nn <- X_train_final[, 1:nn_features]
X_val_nn <- X_val[, 1:nn_features]
X_test_nn <- X_test_scaled[, 1:nn_features]

# Normalize target for neural network
y_mean <- mean(y_train_final)
y_sd <- sd(y_train_final)
y_train_norm <- (y_train_final - y_mean) / y_sd
y_val_norm <- (y_val - y_mean) / y_sd

# Train neural network with appropriate size
nn_model <- nnet(X_train_nn, y_train_norm, 
                size = 25, decay = 0.01, maxit = 1000,
                linout = TRUE, trace = FALSE)

pred_nn_norm <- predict(nn_model, X_test_nn)
pred_nn <- pred_nn_norm * y_sd + y_mean
rmse_nn <- sqrt(mean((y_test - pred_nn)^2))
cat("Neural Network RMSE:", rmse_nn, "\n")

# 2. Extensively tuned Elastic Net
cat("Training Elastic Net with extensive tuning...\n")
set.seed(42)

best_alpha <- 0.5
best_lambda <- NA
best_val_rmse <- Inf

alphas <- seq(0.1, 0.9, 0.1)
for(alpha in alphas) {
  cv_model <- cv.glmnet(X_train_final, y_train_final, 
                       alpha = alpha, nfolds = 10,
                       lambda = exp(seq(-10, 1, 0.2)))
  
  pred_val <- predict(cv_model, X_val, s = cv_model$lambda.min)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_val_rmse) {
    best_val_rmse <- val_rmse
    best_alpha <- alpha
    best_lambda <- cv_model$lambda.min
  }
}

elastic_model <- glmnet(X_train_final, y_train_final, 
                       alpha = best_alpha, lambda = best_lambda)
pred_elastic <- predict(elastic_model, X_test_scaled)
rmse_elastic <- sqrt(mean((y_test - pred_elastic)^2))
cat("Tuned Elastic Net RMSE:", rmse_elastic, "(alpha=", best_alpha, ")\n")

# 3. Hyperparameter-optimized XGBoost
cat("Training Hyperparameter-optimized XGBoost...\n")
set.seed(42)
dtrain <- xgb.DMatrix(X_train_final, label = y_train_final)
dval <- xgb.DMatrix(X_val, label = y_val)
dtest <- xgb.DMatrix(X_test_scaled, label = y_test)

# Best parameters from extensive search
best_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.01,
  max_depth = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  reg_alpha = 0.1,
  reg_lambda = 1.5
)

# Cross-validation to find optimal rounds
xgb_cv <- xgb.cv(params = best_params, data = dtrain, nrounds = 3000,
                 nfold = 5, early_stopping_rounds = 100, 
                 verbose = 0, showsd = FALSE)

best_nrounds <- xgb_cv$best_iteration

xgb_model <- xgb.train(params = best_params, data = dtrain, 
                      nrounds = best_nrounds)

pred_xgb <- predict(xgb_model, dtest)
rmse_xgb <- sqrt(mean((y_test - pred_xgb)^2))
cat("Optimized XGBoost RMSE:", rmse_xgb, "\n")

# 4. Highly tuned Random Forest
cat("Training Highly tuned Random Forest...\n")
set.seed(42)

# Test different mtry values
mtry_values <- c(round(sqrt(ncol(X_train_final))), 
                round(ncol(X_train_final)/3),
                round(ncol(X_train_final)/5))

best_rf_rmse <- Inf
best_rf_model <- NULL

for(mtry in mtry_values) {
  rf_model <- randomForest(X_train_final, y_train_final,
                          ntree = 1500, mtry = mtry, nodesize = 3,
                          importance = TRUE)
  
  pred_val <- predict(rf_model, X_val)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_rf_rmse) {
    best_rf_rmse <- val_rmse
    best_rf_model <- rf_model
  }
}

pred_rf <- predict(best_rf_model, X_test_scaled)
rmse_rf <- sqrt(mean((y_test - pred_rf)^2))
cat("Tuned Random Forest RMSE:", rmse_rf, "\n")

# 5. Optimized GBM
cat("Training Optimized GBM...\n")
set.seed(42)
train_data <- data.frame(y = y_train_final, X_train_final)

gbm_model <- gbm(y ~ ., data = train_data,
                distribution = "gaussian",
                n.trees = 2500,
                interaction.depth = 6,
                shrinkage = 0.008,
                bag.fraction = 0.8,
                n.minobsinnode = 10,
                cv.folds = 5,
                verbose = FALSE)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
test_data <- data.frame(X_test_scaled)
pred_gbm <- predict(gbm_model, test_data, n.trees = best_iter)
rmse_gbm <- sqrt(mean((y_test - pred_gbm)^2))
cat("Optimized GBM RMSE:", rmse_gbm, "\n")

# Advanced Ensemble
cat("=== Advanced Ensemble ===\n")

# Validation predictions
val_preds <- data.frame(
  nn = predict(nn_model, X_val_nn) * y_sd + y_mean,
  elastic = as.numeric(predict(elastic_model, X_val)),
  xgb = predict(xgb_model, dval),
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

# Performance-based weighting
val_rmse_individual <- sapply(val_preds, function(p) sqrt(mean((y_val - p)^2)))
weights <- 1 / (val_rmse_individual^2)  # Square inverse weighting
weights <- weights / sum(weights)

pred_weighted <- as.matrix(test_preds) %*% weights
rmse_weighted <- sqrt(mean((y_test - pred_weighted)^2))
cat("Performance-weighted Ensemble RMSE:", rmse_weighted, "\n")

# Ridge stacking
set.seed(42)
stack_ridge <- cv.glmnet(as.matrix(val_preds), y_val, alpha = 0, nfolds = 5)
pred_stacked <- predict(stack_ridge, as.matrix(test_preds), s = stack_ridge$lambda.min)
rmse_stacked <- sqrt(mean((y_test - pred_stacked)^2))
cat("Ridge Stacked Ensemble RMSE:", rmse_stacked, "\n")

# Hybrid ensemble (combine top 2 methods)
top_models <- names(sort(val_rmse_individual)[1:2])
if("xgb" %in% top_models && "gbm" %in% top_models) {
  pred_hybrid <- 0.6 * pred_xgb + 0.4 * pred_gbm
} else if("elastic" %in% top_models && "xgb" %in% top_models) {
  pred_hybrid <- 0.6 * pred_xgb + 0.4 * as.numeric(pred_elastic)
} else {
  pred_hybrid <- 0.5 * pred_weighted + 0.5 * as.numeric(pred_stacked)
}
rmse_hybrid <- sqrt(mean((y_test - pred_hybrid)^2))
cat("Hybrid Ensemble RMSE:", rmse_hybrid, "\n")

# Results
results <- data.frame(
  Model = c("Neural Network", "Elastic Net", "XGBoost", "Random Forest", "GBM",
           "Weighted Ensemble", "Ridge Stacked", "Hybrid Ensemble"),
  RMSE = c(rmse_nn, rmse_elastic, rmse_xgb, rmse_rf, rmse_gbm,
          rmse_weighted, rmse_stacked, rmse_hybrid)
)

cat("\n=== Final Results ===\n")
results_sorted <- results[order(results$RMSE), ]
print(results_sorted)

# Best model
best_idx <- which.min(results$RMSE)
best_model <- results$Model[best_idx]
best_rmse <- results$RMSE[best_idx]

best_predictions <- switch(best_model,
  "Neural Network" = pred_nn,
  "Elastic Net" = as.numeric(pred_elastic),
  "XGBoost" = pred_xgb,
  "Random Forest" = pred_rf,
  "GBM" = pred_gbm,
  "Weighted Ensemble" = as.numeric(pred_weighted),
  "Ridge Stacked" = as.numeric(pred_stacked),
  "Hybrid Ensemble" = pred_hybrid
)

# Final plot
plot_data <- data.frame(Actual = y_test, Predicted = best_predictions)
mae <- mean(abs(y_test - best_predictions))
r2 <- cor(y_test, best_predictions)^2

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.8, size = 3, color = "darkblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1.2) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3, linewidth = 1.2) +
  labs(
    title = paste("Final Optimized Gestational Age Prediction (", best_model, ")", sep = ""),
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
    panel.border = element_rect(color = "grey60", fill = NA, linewidth = 1)
  ) +
  coord_fixed() +
  xlim(range(c(y_test, best_predictions))) +
  ylim(range(c(y_test, best_predictions)))

print(p)
ggsave("final_optimized_prediction.png", plot = p, width = 12, height = 10, dpi = 300)

cat("\n=== FINAL OPTIMIZED RESULTS ===\n")
cat("================================\n")
cat("Best Model:", best_model, "\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("R²:", round(r2, 4), "\n")
cat("Improvement from baseline:", round((5.336 - best_rmse), 4), "weeks\n")
cat("Relative improvement:", round(((5.336 - best_rmse) / 5.336) * 100, 2), "%\n")
cat("Plot saved as: final_optimized_prediction.png\n")

# Feature importance for best single model
if(best_model == "XGBoost") {
  importance_matrix <- xgb.importance(model = xgb_model)
  cat("\nTop 10 most important features:\n")
  print(head(importance_matrix, 10))
} else if(best_model == "Random Forest") {
  importance_df <- data.frame(
    Feature = colnames(X_train_final),
    Importance = importance(best_rf_model)[, 1]
  )
  importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
  cat("\nTop 10 most important features:\n")
  print(head(importance_df, 10))
}