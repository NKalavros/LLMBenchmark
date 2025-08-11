# Efficient Optimized Gestational Age Prediction
# Focus on most promising optimization techniques

library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)

# Download GSE149440 dataset (reuse if already downloaded)
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

# Efficient Feature Selection
cat("=== Efficient Feature Selection ===\n")

# Remove zero variance genes
gene_vars <- apply(X_train, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > 0 & !is.na(gene_vars))
X_train <- X_train[, high_var_genes]  
X_test <- X_test[, high_var_genes]

# Top correlated genes (most efficient)
gene_cors <- abs(cor(X_train, y_train, use = "complete.obs"))
top_genes <- order(gene_cors, decreasing = TRUE)[1:min(1500, length(gene_cors))]

X_train_sel <- X_train[, top_genes]
X_test_sel <- X_test[, top_genes]

cat("Selected", ncol(X_train_sel), "top correlated genes\n")

# Scale data
X_train_scaled <- scale(X_train_sel)
X_test_scaled <- scale(X_test_sel, 
                      center = attr(X_train_scaled, "scaled:center"),
                      scale = attr(X_train_scaled, "scaled:scale"))

# Create validation split
set.seed(42)
val_idx <- sample(nrow(X_train_scaled), size = round(0.2 * nrow(X_train_scaled)))
X_val <- X_train_scaled[val_idx, ]
y_val <- y_train[val_idx]
X_train_final <- X_train_scaled[-val_idx, ]
y_train_final <- y_train[-val_idx]

cat("Training:", nrow(X_train_final), "Validation:", nrow(X_val), "\n")

# Model Training
cat("=== Model Training ===\n")

# 1. Optimized Elastic Net
cat("Training Elastic Net...\n")
set.seed(42)
alphas <- c(0.1, 0.3, 0.5, 0.7, 0.9)
best_alpha <- 0.5
best_val_rmse <- Inf

for(alpha in alphas) {
  cv_model <- cv.glmnet(X_train_final, y_train_final, alpha = alpha, nfolds = 5)
  pred_val <- predict(cv_model, X_val, s = cv_model$lambda.min)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  if(val_rmse < best_val_rmse) {
    best_val_rmse <- val_rmse
    best_alpha <- alpha
  }
}

elastic_cv <- cv.glmnet(X_train_final, y_train_final, alpha = best_alpha, nfolds = 5)
elastic_model <- glmnet(X_train_final, y_train_final, alpha = best_alpha, lambda = elastic_cv$lambda.min)
pred_elastic <- predict(elastic_model, X_test_scaled)
rmse_elastic <- sqrt(mean((y_test - pred_elastic)^2))
cat("Elastic Net RMSE:", rmse_elastic, "\n")

# 2. Tuned Random Forest  
cat("Training Random Forest...\n")
set.seed(42)
rf_model <- randomForest(X_train_final, y_train_final, 
                        ntree = 1000, 
                        mtry = round(sqrt(ncol(X_train_final))), 
                        nodesize = 5,
                        importance = TRUE)
pred_rf <- predict(rf_model, X_test_scaled)
rmse_rf <- sqrt(mean((y_test - pred_rf)^2))
cat("Random Forest RMSE:", rmse_rf, "\n")

# 3. Optimized XGBoost
cat("Training XGBoost...\n")
set.seed(42)
dtrain <- xgb.DMatrix(X_train_final, label = y_train_final)
dval <- xgb.DMatrix(X_val, label = y_val)
dtest <- xgb.DMatrix(X_test_scaled, label = y_test)

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.01,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  reg_alpha = 0.1,
  reg_lambda = 1
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 2000,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 50,
  verbose = 0
)

pred_xgb <- predict(xgb_model, dtest)
rmse_xgb <- sqrt(mean((y_test - pred_xgb)^2))
cat("XGBoost RMSE:", rmse_xgb, "\n")

# 4. Gradient Boosting
cat("Training GBM...\n")
library(gbm)
set.seed(42)
train_data <- data.frame(y = y_train_final, X_train_final)
gbm_model <- gbm(y ~ ., data = train_data,
                distribution = "gaussian",
                n.trees = 1500,
                interaction.depth = 5,
                shrinkage = 0.01,
                bag.fraction = 0.8,
                cv.folds = 5,
                verbose = FALSE)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
test_data <- data.frame(X_test_scaled)
pred_gbm <- predict(gbm_model, test_data, n.trees = best_iter)
rmse_gbm <- sqrt(mean((y_test - pred_gbm)^2))
cat("GBM RMSE:", rmse_gbm, "\n")

# Ensemble Methods  
cat("=== Ensemble Methods ===\n")

# Get validation predictions
val_preds <- data.frame(
  elastic = as.numeric(predict(elastic_model, X_val)),
  rf = predict(rf_model, X_val),
  xgb = predict(xgb_model, dval),
  gbm = predict(gbm_model, data.frame(X_val), n.trees = best_iter)
)

test_preds <- data.frame(
  elastic = as.numeric(pred_elastic),
  rf = pred_rf,
  xgb = pred_xgb,
  gbm = pred_gbm
)

# Weighted ensemble based on validation RMSE
val_rmse <- sapply(val_preds, function(p) sqrt(mean((y_val - p)^2)))
weights <- 1 / val_rmse
weights <- weights / sum(weights)

pred_weighted <- as.matrix(test_preds) %*% weights
rmse_weighted <- sqrt(mean((y_test - pred_weighted)^2))
cat("Weighted Ensemble RMSE:", rmse_weighted, "\n")

# Stacked ensemble with LASSO
set.seed(42)
stack_model <- cv.glmnet(as.matrix(val_preds), y_val, alpha = 1, nfolds = 5)
pred_stacked <- predict(stack_model, as.matrix(test_preds), s = stack_model$lambda.min)
rmse_stacked <- sqrt(mean((y_test - pred_stacked)^2))
cat("Stacked Ensemble RMSE:", rmse_stacked, "\n")

# Results summary
results <- data.frame(
  Model = c("Elastic Net", "Random Forest", "XGBoost", "GBM", "Weighted Ensemble", "Stacked Ensemble"),
  RMSE = c(rmse_elastic, rmse_rf, rmse_xgb, rmse_gbm, rmse_weighted, rmse_stacked)
)

cat("\n=== Final Results ===\n")
print(results[order(results$RMSE), ])

# Best model
best_idx <- which.min(results$RMSE)
best_model <- results$Model[best_idx]
best_rmse <- results$RMSE[best_idx]

best_predictions <- switch(best_model,
  "Elastic Net" = as.numeric(pred_elastic),
  "Random Forest" = pred_rf,
  "XGBoost" = pred_xgb,
  "GBM" = pred_gbm,
  "Weighted Ensemble" = as.numeric(pred_weighted),
  "Stacked Ensemble" = as.numeric(pred_stacked)
)

# Create plot
plot_data <- data.frame(Actual = y_test, Predicted = best_predictions)
mae <- mean(abs(y_test - best_predictions))
r2 <- cor(y_test, best_predictions)^2

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7, size = 2.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3) +
  labs(
    title = paste("Optimized Gestational Age Prediction (", best_model, ")", sep = ""),
    subtitle = paste("RMSE =", round(best_rmse, 3), "| MAE =", round(mae, 3), "| R² =", round(r2, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11)
  ) +
  coord_fixed()

print(p)
ggsave("final_optimized_prediction.png", plot = p, width = 10, height = 8, dpi = 300)

cat("\n=== FINAL OPTIMIZED RESULTS ===\n")
cat("==================================\n")
cat("Best Model:", best_model, "\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("R²:", round(r2, 4), "\n")
cat("Improvement from baseline:", round((5.336 - best_rmse), 4), "weeks\n")
cat("Relative improvement:", round(((5.336 - best_rmse) / 5.336) * 100, 2), "%\n")
cat("Plot saved as: final_optimized_prediction.png\n")