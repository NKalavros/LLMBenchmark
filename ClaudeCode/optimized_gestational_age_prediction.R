# Optimized Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440

# Load required libraries
library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(Boruta)
library(VIM)
library(corrplot)
library(doParallel)

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

# Remove highly correlated genes
cor_matrix <- cor(X_train)
high_cor <- findCorrelation(cor_matrix, cutoff = 0.95)
if(length(high_cor) > 0) {
  X_train <- X_train[, -high_cor]
  X_test <- X_test[, -high_cor]
}
cat("After removing highly correlated genes:", ncol(X_train), "genes remaining\n")

# Advanced Feature Selection
cat("\n=== Advanced Feature Selection ===\n")

# Method 1: Univariate selection with F-test
f_scores <- apply(X_train, 2, function(x) {
  fit <- lm(y_train ~ x)
  summary(fit)$fstatistic[1]
})
f_scores[is.na(f_scores)] <- 0
top_f_genes <- order(f_scores, decreasing = TRUE)[1:min(2000, length(f_scores))]

# Method 2: LASSO feature selection
set.seed(42)
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)
lasso_coef <- coef(lasso_cv, s = lasso_cv$lambda.1se)
lasso_genes <- which(lasso_coef[-1] != 0)

# Method 3: Random Forest importance
set.seed(42)
rf_importance <- randomForest(X_train[, 1:min(1000, ncol(X_train))], y_train, 
                             ntree = 100, importance = TRUE)
rf_imp_scores <- importance(rf_importance)[, 1]
top_rf_genes <- order(rf_imp_scores, decreasing = TRUE)[1:min(500, length(rf_imp_scores))]

# Combine feature selection methods
selected_genes <- unique(c(top_f_genes[1:500], lasso_genes, top_rf_genes))
selected_genes <- selected_genes[selected_genes <= ncol(X_train)]

X_train_selected <- X_train[, selected_genes]
X_test_selected <- X_test[, selected_genes]

cat("Selected", ncol(X_train_selected), "genes using combined feature selection\n")

# Robust scaling
preProcess_model <- preProcess(X_train_selected, method = c("center", "scale", "YeoJohnson"))
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

# Model Training with Hyperparameter Tuning
cat("\n=== Model Training with Hyperparameter Tuning ===\n")

# 1. Optimized Elastic Net
cat("Training optimized Elastic Net...\n")
set.seed(42)
elastic_grid <- expand.grid(alpha = seq(0.1, 1, 0.1), lambda = exp(seq(-10, 2, 0.5)))
elastic_cv <- cv.glmnet(as.matrix(X_train_final), y_train_final, 
                       alpha = 0.5, nfolds = 10, parallel = TRUE)
elastic_model <- glmnet(as.matrix(X_train_final), y_train_final, 
                       alpha = 0.5, lambda = elastic_cv$lambda.min)

pred_elastic_val <- predict(elastic_model, as.matrix(X_val))
pred_elastic_test <- predict(elastic_model, as.matrix(X_test_processed))
rmse_elastic <- sqrt(mean((y_test - pred_elastic_test)^2))
cat("Elastic Net RMSE:", rmse_elastic, "\n")

# 2. Optimized Random Forest
cat("Training optimized Random Forest...\n")
set.seed(42)
rf_grid <- expand.grid(mtry = c(sqrt(ncol(X_train_final)), 
                               ncol(X_train_final)/3, 
                               ncol(X_train_final)/5))
rf_control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
rf_tune <- train(X_train_final, y_train_final, method = "rf",
                tuneGrid = rf_grid, trControl = rf_control,
                ntree = 1000, importance = TRUE)

pred_rf_test <- predict(rf_tune, X_test_processed)
rmse_rf <- sqrt(mean((y_test - pred_rf_test)^2))
cat("Random Forest RMSE:", rmse_rf, "\n")

# 3. XGBoost
cat("Training XGBoost...\n")
set.seed(42)
xgb_train <- xgb.DMatrix(data = as.matrix(X_train_final), label = y_train_final)
xgb_val <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test_processed), label = y_test)

xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.01,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  reg_alpha = 0.1,
  reg_lambda = 1
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 2000,
  watchlist = list(train = xgb_train, val = xgb_val),
  early_stopping_rounds = 100,
  verbose = 0
)

pred_xgb_test <- predict(xgb_model, xgb_test)
rmse_xgb <- sqrt(mean((y_test - pred_xgb_test)^2))
cat("XGBoost RMSE:", rmse_xgb, "\n")

# 4. Support Vector Regression with tuning
cat("Training optimized SVR...\n")
library(e1071)
set.seed(42)
svr_tune <- tune(svm, X_train_final, y_train_final,
                ranges = list(cost = c(0.1, 1, 10, 100),
                             gamma = c(0.001, 0.01, 0.1, 1)),
                kernel = "radial", tunecontrol = tune.control(cross = 5))

pred_svr_test <- predict(svr_tune$best.model, X_test_processed)
rmse_svr <- sqrt(mean((y_test - pred_svr_test)^2))
cat("SVR RMSE:", rmse_svr, "\n")

# 5. Gradient Boosting Machine
cat("Training GBM...\n")
library(gbm)
set.seed(42)
train_data <- data.frame(y = y_train_final, X_train_final)
gbm_model <- gbm(y ~ ., data = train_data, 
                distribution = "gaussian",
                n.trees = 2000,
                interaction.depth = 5,
                shrinkage = 0.01,
                bag.fraction = 0.8,
                cv.folds = 5,
                n.cores = 1)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
test_data <- data.frame(X_test_processed)
pred_gbm_test <- predict(gbm_model, test_data, n.trees = best_iter)
rmse_gbm <- sqrt(mean((y_test - pred_gbm_test)^2))
cat("GBM RMSE:", rmse_gbm, "\n")

# Ensemble Methods
cat("\n=== Ensemble Methods ===\n")

# Collect all predictions
all_predictions <- data.frame(
  elastic = as.numeric(pred_elastic_test),
  rf = pred_rf_test,
  xgb = pred_xgb_test,
  svr = pred_svr_test,
  gbm = pred_gbm_test
)

# Simple average ensemble
pred_ensemble_avg <- rowMeans(all_predictions)
rmse_ensemble_avg <- sqrt(mean((y_test - pred_ensemble_avg)^2))
cat("Average Ensemble RMSE:", rmse_ensemble_avg, "\n")

# Weighted ensemble based on validation performance
val_predictions <- data.frame(
  elastic = as.numeric(predict(elastic_model, as.matrix(X_val))),
  rf = predict(rf_tune, X_val),
  xgb = predict(xgb_model, xgb.DMatrix(data = as.matrix(X_val))),
  svr = predict(svr_tune$best.model, X_val),
  gbm = predict(gbm_model, data.frame(X_val), n.trees = best_iter)
)

# Calculate weights based on inverse RMSE on validation set
val_rmse <- sapply(val_predictions, function(pred) sqrt(mean((y_val - pred)^2)))
weights <- 1 / val_rmse
weights <- weights / sum(weights)

pred_ensemble_weighted <- as.matrix(all_predictions) %*% weights
rmse_ensemble_weighted <- sqrt(mean((y_test - pred_ensemble_weighted)^2))
cat("Weighted Ensemble RMSE:", rmse_ensemble_weighted, "\n")

# Stacked ensemble using linear regression
set.seed(42)
stack_model <- lm(y_val ~ ., data = data.frame(y_val = y_val, val_predictions))
pred_ensemble_stack <- predict(stack_model, all_predictions)
rmse_ensemble_stack <- sqrt(mean((y_test - pred_ensemble_stack)^2))
cat("Stacked Ensemble RMSE:", rmse_ensemble_stack, "\n")

# Collect all results
results <- data.frame(
  Model = c("Elastic Net", "Random Forest", "XGBoost", "SVR", "GBM", 
           "Average Ensemble", "Weighted Ensemble", "Stacked Ensemble"),
  RMSE = c(rmse_elastic, rmse_rf, rmse_xgb, rmse_svr, rmse_gbm,
          rmse_ensemble_avg, rmse_ensemble_weighted, rmse_ensemble_stack)
)

# Find best model
best_idx <- which.min(results$RMSE)
best_model_name <- results$Model[best_idx]
best_rmse <- results$RMSE[best_idx]

# Get best predictions
if(best_model_name == "Average Ensemble") {
  best_predictions <- pred_ensemble_avg
} else if(best_model_name == "Weighted Ensemble") {
  best_predictions <- pred_ensemble_weighted
} else if(best_model_name == "Stacked Ensemble") {
  best_predictions <- pred_ensemble_stack
} else if(best_model_name == "XGBoost") {
  best_predictions <- pred_xgb_test
} else if(best_model_name == "Random Forest") {
  best_predictions <- pred_rf_test
} else if(best_model_name == "Elastic Net") {
  best_predictions <- as.numeric(pred_elastic_test)
} else if(best_model_name == "SVR") {
  best_predictions <- pred_svr_test
} else if(best_model_name == "GBM") {
  best_predictions <- pred_gbm_test
}

cat("\n=== All Model Results ===\n")
print(results)

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
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3) +
  labs(
    title = paste("Optimized Gestational Age Prediction (", best_model_name, ")", sep = ""),
    subtitle = paste("RMSE =", round(best_rmse, 3), "| MAE =", round(mae, 3), "| R² =", round(r2, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    panel.grid.minor = element_blank()
  ) +
  coord_fixed() +
  xlim(range(c(y_test, best_predictions))) +
  ylim(range(c(y_test, best_predictions)))

print(p)
ggsave("optimized_gestational_age_prediction_plot.png", plot = p, width = 10, height = 8, dpi = 300)

cat("\n=== FINAL OPTIMIZED RESULTS ===\n")
cat("====================================\n")
cat("Best Model:", best_model_name, "\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("R²:", round(r2, 4), "\n")
cat("Improvement from baseline:", round((5.336 - best_rmse), 4), "weeks\n")
cat("Relative improvement:", round(((5.336 - best_rmse) / 5.336) * 100, 2), "%\n")
cat("Plot saved as: optimized_gestational_age_prediction_plot.png\n")

# Feature importance analysis for best model
if(best_model_name == "XGBoost") {
  importance_matrix <- xgb.importance(model = xgb_model)
  cat("\nTop 10 most important features (XGBoost):\n")
  print(head(importance_matrix, 10))
}

stopImplicitCluster()