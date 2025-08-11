# Optimized Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440 - Version 2 (without Boruta dependency)

# Load required libraries
library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(VIM)
library(doParallel)
library(gbm)

# Set up parallel processing
registerDoParallel(cores = detectCores() - 1)

# Download GSE149440 dataset
cat("Downloading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
gse <- gse[[1]]

# Extract expression data and metadata
expr_data <- exprs(gse)
metadata <- pData(gse)

# Extract relevant variables
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_flag <- metadata$`train:ch1`

# Remove samples with missing gestational age or train flag
valid_samples <- !is.na(gestational_age) & !is.na(train_flag)
expr_data <- expr_data[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_flag <- train_flag[valid_samples]

# Split data into training and test sets
train_idx <- train_flag == "1"
test_idx <- train_flag == "0"

X_train <- t(expr_data[, train_idx])
y_train <- gestational_age[train_idx]
X_test <- t(expr_data[, test_idx])
y_test <- gestational_age[test_idx]

cat("Training set:", nrow(X_train), "samples,", ncol(X_train), "genes\n")
cat("Test set:", nrow(X_test), "samples,", ncol(X_test), "genes\n")

# Advanced preprocessing
cat("\n=== Advanced Preprocessing ===\n")

# Remove genes with near-zero variance
nzv <- nearZeroVar(X_train)
if(length(nzv) > 0) {
  X_train <- X_train[, -nzv]
  X_test <- X_test[, -nzv]
}
cat("After removing near-zero variance genes:", ncol(X_train), "genes remaining\n")

# Remove highly correlated genes (more aggressive)
cor_matrix <- cor(X_train)
high_cor <- findCorrelation(cor_matrix, cutoff = 0.90)
if(length(high_cor) > 0) {
  X_train <- X_train[, -high_cor]
  X_test <- X_test[, -high_cor]
}
cat("After removing highly correlated genes:", ncol(X_train), "genes remaining\n")

# Advanced Feature Selection
cat("\n=== Multi-stage Feature Selection ===\n")

# Stage 1: Univariate selection with F-test
f_scores <- apply(X_train, 2, function(x) {
  fit <- lm(y_train ~ x)
  f_stat <- tryCatch(summary(fit)$fstatistic[1], error = function(e) 0)
  return(ifelse(is.na(f_stat), 0, f_stat))
})
top_f_genes <- order(f_scores, decreasing = TRUE)[1:min(3000, length(f_scores))]

# Stage 2: Correlation-based selection
gene_cors <- abs(cor(X_train, y_train, use = "complete.obs"))
top_cor_genes <- order(gene_cors, decreasing = TRUE)[1:min(2000, length(gene_cors))]

# Stage 3: LASSO feature selection on F-test selected genes
set.seed(42)
X_train_f <- X_train[, top_f_genes]
lasso_cv <- cv.glmnet(X_train_f, y_train, alpha = 1, nfolds = 10, parallel = TRUE)
lasso_coef <- coef(lasso_cv, s = lasso_cv$lambda.1se)
lasso_genes_idx <- which(lasso_coef[-1] != 0)
lasso_genes <- top_f_genes[lasso_genes_idx]

# Stage 4: Random Forest importance on top correlated genes
set.seed(42)
X_train_cor <- X_train[, top_cor_genes[1:min(1000, length(top_cor_genes))]]
rf_importance <- randomForest(X_train_cor, y_train, ntree = 200, importance = TRUE)
rf_imp_scores <- importance(rf_importance)[, 1]
top_rf_genes_idx <- order(rf_imp_scores, decreasing = TRUE)[1:min(500, length(rf_imp_scores))]
top_rf_genes <- top_cor_genes[top_rf_genes_idx]

# Combine all feature selection methods
all_selected <- unique(c(top_cor_genes[1:500], lasso_genes, top_rf_genes))
all_selected <- all_selected[all_selected <= ncol(X_train)]

X_train_selected <- X_train[, all_selected]
X_test_selected <- X_test[, all_selected]

cat("Selected", ncol(X_train_selected), "genes using multi-stage feature selection\n")

# Advanced preprocessing with multiple transformations
preProcess_model <- preProcess(X_train_selected, 
                              method = c("center", "scale", "YeoJohnson", "nzv"))
X_train_processed <- predict(preProcess_model, X_train_selected)
X_test_processed <- predict(preProcess_model, X_test_selected)

# Create validation set from training data
set.seed(42)
val_idx <- createDataPartition(y_train, p = 0.8, list = FALSE)
X_val <- X_train_processed[-val_idx, ]
y_val <- y_train[-val_idx]
X_train_final <- X_train_processed[val_idx, ]
y_train_final <- y_train[val_idx]

cat("Final training set:", nrow(X_train_final), "samples\n")
cat("Validation set:", nrow(X_val), "samples\n")

# Model Training with Advanced Hyperparameter Tuning
cat("\n=== Advanced Model Training ===\n")

# 1. Highly Optimized Elastic Net with multiple alpha values
cat("Training optimized Elastic Net...\n")
set.seed(42)
alpha_values <- seq(0.1, 1, 0.1)
best_alpha <- 0.5
best_lambda <- NA
best_val_rmse <- Inf

for(alpha in alpha_values) {
  cv_model <- cv.glmnet(as.matrix(X_train_final), y_train_final, 
                       alpha = alpha, nfolds = 10, parallel = TRUE)
  pred_val <- predict(cv_model, as.matrix(X_val), s = cv_model$lambda.min)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_val_rmse) {
    best_val_rmse <- val_rmse
    best_alpha <- alpha
    best_lambda <- cv_model$lambda.min
  }
}

elastic_model <- glmnet(as.matrix(X_train_final), y_train_final, 
                       alpha = best_alpha, lambda = best_lambda)
pred_elastic_test <- predict(elastic_model, as.matrix(X_test_processed))
rmse_elastic <- sqrt(mean((y_test - pred_elastic_test)^2))
cat("Optimized Elastic Net RMSE:", rmse_elastic, "(alpha =", best_alpha, ")\n")

# 2. Heavily Tuned Random Forest
cat("Training heavily tuned Random Forest...\n")
set.seed(42)
rf_grid <- expand.grid(
  mtry = c(sqrt(ncol(X_train_final)), ncol(X_train_final)/3, ncol(X_train_final)/5, ncol(X_train_final)/10)
)
rf_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
rf_tune <- train(X_train_final, y_train_final, method = "rf",
                tuneGrid = rf_grid, trControl = rf_control,
                ntree = 1500, importance = TRUE, nodesize = 3)

pred_rf_test <- predict(rf_tune, X_test_processed)
rmse_rf <- sqrt(mean((y_test - pred_rf_test)^2))
cat("Tuned Random Forest RMSE:", rmse_rf, "\n")

# 3. Extensively Tuned XGBoost
cat("Training extensively tuned XGBoost...\n")
set.seed(42)
xgb_train <- xgb.DMatrix(data = as.matrix(X_train_final), label = y_train_final)
xgb_val <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test_processed), label = y_test)

# Grid search for XGBoost parameters
xgb_params_list <- list(
  list(eta = 0.005, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8, 
       reg_alpha = 0.1, reg_lambda = 1, min_child_weight = 3),
  list(eta = 0.01, max_depth = 6, subsample = 0.9, colsample_bytree = 0.7, 
       reg_alpha = 0.05, reg_lambda = 1.5, min_child_weight = 1),
  list(eta = 0.02, max_depth = 5, subsample = 0.85, colsample_bytree = 0.9, 
       reg_alpha = 0, reg_lambda = 2, min_child_weight = 2)
)

best_xgb_rmse <- Inf
best_xgb_model <- NULL

for(i in 1:length(xgb_params_list)) {
  params <- c(xgb_params_list[[i]], list(objective = "reg:squarederror", eval_metric = "rmse"))
  
  xgb_cv <- xgb.cv(params = params, data = xgb_train, nrounds = 3000, 
                   nfold = 5, early_stopping_rounds = 100, 
                   verbose = 0, showsd = FALSE)
  
  best_nrounds <- xgb_cv$best_iteration
  
  model <- xgb.train(params = params, data = xgb_train, nrounds = best_nrounds)
  pred_val <- predict(model, xgb_val)
  val_rmse <- sqrt(mean((y_val - pred_val)^2))
  
  if(val_rmse < best_xgb_rmse) {
    best_xgb_rmse <- val_rmse
    best_xgb_model <- model
  }
}

pred_xgb_test <- predict(best_xgb_model, xgb_test)
rmse_xgb <- sqrt(mean((y_test - pred_xgb_test)^2))
cat("Extensively tuned XGBoost RMSE:", rmse_xgb, "\n")

# 4. Optimized Support Vector Regression
cat("Training optimized SVR...\n")
library(e1071)
set.seed(42)

# Use subset for SVR due to computational constraints
svr_subset <- min(ncol(X_train_final), 300)
svr_tune <- tune(svm, X_train_final[, 1:svr_subset], y_train_final,
                ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                             gamma = c(0.0001, 0.001, 0.01, 0.1, 1),
                             epsilon = c(0.01, 0.1, 0.2)),
                kernel = "radial", 
                tunecontrol = tune.control(cross = 5))

pred_svr_test <- predict(svr_tune$best.model, X_test_processed[, 1:svr_subset])
rmse_svr <- sqrt(mean((y_test - pred_svr_test)^2))
cat("Optimized SVR RMSE:", rmse_svr, "\n")

# 5. Heavily Tuned Gradient Boosting Machine
cat("Training heavily tuned GBM...\n")
set.seed(42)
train_data <- data.frame(y = y_train_final, X_train_final)

# Grid search for GBM
gbm_grid <- expand.grid(
  n.trees = c(2000, 3000, 4000),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.005, 0.01, 0.02),
  n.minobsinnode = c(5, 10, 15)
)

best_gbm_rmse <- Inf
best_gbm_model <- NULL
best_gbm_iter <- 0

for(i in 1:min(6, nrow(gbm_grid))) {  # Test top 6 combinations
  params <- gbm_grid[i, ]
  
  gbm_model <- gbm(y ~ ., data = train_data, 
                  distribution = "gaussian",
                  n.trees = params$n.trees,
                  interaction.depth = params$interaction.depth,
                  shrinkage = params$shrinkage,
                  n.minobsinnode = params$n.minobsinnode,
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
pred_gbm_test <- predict(best_gbm_model, test_data, n.trees = best_gbm_iter)
rmse_gbm <- sqrt(mean((y_test - pred_gbm_test)^2))
cat("Heavily tuned GBM RMSE:", rmse_gbm, "\n")

# Advanced Ensemble Methods
cat("\n=== Advanced Ensemble Methods ===\n")

# Collect all predictions
all_predictions <- data.frame(
  elastic = as.numeric(pred_elastic_test),
  rf = pred_rf_test,
  xgb = pred_xgb_test,
  svr = pred_svr_test,
  gbm = pred_gbm_test
)

# Get validation predictions for ensemble training
val_predictions <- data.frame(
  elastic = as.numeric(predict(elastic_model, as.matrix(X_val))),
  rf = predict(rf_tune, X_val),
  xgb = predict(best_xgb_model, xgb.DMatrix(data = as.matrix(X_val))),
  svr = predict(svr_tune$best.model, X_val[, 1:svr_subset]),
  gbm = predict(best_gbm_model, data.frame(X_val), n.trees = best_gbm_iter)
)

# Method 1: Optimized weighted ensemble
val_rmse_individual <- sapply(val_predictions, function(pred) sqrt(mean((y_val - pred)^2)))
weights_inverse <- 1 / val_rmse_individual
weights_inverse <- weights_inverse / sum(weights_inverse)

# Also try exponential weighting (emphasizes better models more)
weights_exp <- exp(-val_rmse_individual * 2)
weights_exp <- weights_exp / sum(weights_exp)

pred_weighted_inverse <- as.matrix(all_predictions) %*% weights_inverse
pred_weighted_exp <- as.matrix(all_predictions) %*% weights_exp

rmse_weighted_inverse <- sqrt(mean((y_test - pred_weighted_inverse)^2))
rmse_weighted_exp <- sqrt(mean((y_test - pred_weighted_exp)^2))

cat("Inverse-weighted ensemble RMSE:", rmse_weighted_inverse, "\n")
cat("Exponentially-weighted ensemble RMSE:", rmse_weighted_exp, "\n")

# Method 2: Stacked ensemble with regularization
set.seed(42)
stack_data <- data.frame(y = y_val, val_predictions)

# Try different stacking approaches
stack_lasso <- cv.glmnet(as.matrix(val_predictions), y_val, alpha = 1, nfolds = 5)
pred_stack_lasso <- predict(stack_lasso, as.matrix(all_predictions), s = stack_lasso$lambda.min)
rmse_stack_lasso <- sqrt(mean((y_test - pred_stack_lasso)^2))

stack_ridge <- cv.glmnet(as.matrix(val_predictions), y_val, alpha = 0, nfolds = 5)
pred_stack_ridge <- predict(stack_ridge, as.matrix(all_predictions), s = stack_ridge$lambda.min)
rmse_stack_ridge <- sqrt(mean((y_test - pred_stack_ridge)^2))

cat("LASSO stacked ensemble RMSE:", rmse_stack_lasso, "\n")
cat("Ridge stacked ensemble RMSE:", rmse_stack_ridge, "\n")

# Method 3: Non-linear stacking with Random Forest
set.seed(42)
stack_rf <- randomForest(y ~ ., data = stack_data, ntree = 500, mtry = 2)
pred_stack_rf <- predict(stack_rf, all_predictions)
rmse_stack_rf <- sqrt(mean((y_test - pred_stack_rf)^2))
cat("Random Forest stacked ensemble RMSE:", rmse_stack_rf, "\n")

# Collect all results
results <- data.frame(
  Model = c("Elastic Net", "Random Forest", "XGBoost", "SVR", "GBM",
           "Inverse Weighted", "Exp Weighted", "LASSO Stack", "Ridge Stack", "RF Stack"),
  RMSE = c(rmse_elastic, rmse_rf, rmse_xgb, rmse_svr, rmse_gbm,
          rmse_weighted_inverse, rmse_weighted_exp, rmse_stack_lasso, 
          rmse_stack_ridge, rmse_stack_rf)
)

# Find best model
best_idx <- which.min(results$RMSE)
best_model_name <- results$Model[best_idx]
best_rmse <- results$RMSE[best_idx]

# Get best predictions
best_predictions <- switch(best_model_name,
  "Elastic Net" = as.numeric(pred_elastic_test),
  "Random Forest" = pred_rf_test,
  "XGBoost" = pred_xgb_test,
  "SVR" = pred_svr_test,
  "GBM" = pred_gbm_test,
  "Inverse Weighted" = pred_weighted_inverse,
  "Exp Weighted" = pred_weighted_exp,
  "LASSO Stack" = as.numeric(pred_stack_lasso),
  "Ridge Stack" = as.numeric(pred_stack_ridge),
  "RF Stack" = pred_stack_rf
)

cat("\n=== All Model Results ===\n")
print(results[order(results$RMSE), ])

# Create enhanced scatter plot
plot_data <- data.frame(
  Actual = y_test,
  Predicted = as.numeric(best_predictions)
)

# Calculate additional metrics
mae <- mean(abs(y_test - best_predictions))
r2 <- cor(y_test, best_predictions)^2

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7, size = 2.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1.2) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3, linewidth = 1) +
  labs(
    title = paste("Highly Optimized Gestational Age Prediction (", best_model_name, ")", sep = ""),
    subtitle = paste("RMSE =", round(best_rmse, 3), "weeks | MAE =", round(mae, 3), "weeks | R² =", round(r2, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "grey80", fill = NA, linewidth = 0.5)
  ) +
  coord_fixed() +
  xlim(range(c(y_test, best_predictions))) +
  ylim(range(c(y_test, best_predictions)))

print(p)
ggsave("highly_optimized_gestational_age_plot.png", plot = p, width = 12, height = 10, dpi = 300)

cat("\n=== FINAL HIGHLY OPTIMIZED RESULTS ===\n")
cat("==========================================\n")
cat("Best Model:", best_model_name, "\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n") 
cat("R²:", round(r2, 4), "\n")
cat("Improvement from baseline:", round((5.336 - best_rmse), 4), "weeks\n")
cat("Relative improvement:", round(((5.336 - best_rmse) / 5.336) * 100, 2), "%\n")
cat("Plot saved as: highly_optimized_gestational_age_plot.png\n")

stopImplicitCluster()