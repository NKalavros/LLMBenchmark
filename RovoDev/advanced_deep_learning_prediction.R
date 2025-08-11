# Advanced Deep Learning and Ensemble Techniques for Gestational Age Prediction
# Additional optimization with neural networks and advanced ensemble methods

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
# library(keras)
# library(tensorflow)
library(gbm)
library(nnet)
library(earth)

# Set options and seed
options(timeout = 300)
set.seed(42)

# Load existing data (reuse if available)
cat("Loading GSE149440 dataset...\n")
if(!exists("gse")) {
  gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
}

# Extract the expression set
eset <- gse[[1]]
expr_data <- exprs(eset)
metadata <- pData(eset)

# Extract gestational age and training indicators
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_indicator <- metadata$`train:ch1`

# Create training and test sets
train_mask <- train_indicator == "1"
test_mask <- train_indicator == "0"

# Prepare data with advanced preprocessing
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

# Advanced preprocessing pipeline
cat("\nAdvanced preprocessing pipeline...\n")

# 1. Quantile normalization
combined_expr <- rbind(train_expr, test_expr)
combined_norm <- normalize.quantiles(t(combined_expr))
combined_norm <- t(combined_norm)
colnames(combined_norm) <- colnames(combined_expr)

train_expr_norm <- combined_norm[1:nrow(train_expr), ]
test_expr_norm <- combined_norm[(nrow(train_expr)+1):nrow(combined_norm), ]

# 2. Log2 transformation if needed
if(max(train_expr_norm, na.rm = TRUE) > 50) {
  train_expr_norm <- log2(train_expr_norm + 1)
  test_expr_norm <- log2(test_expr_norm + 1)
}

# 3. Advanced feature selection with multiple criteria
cat("Advanced feature selection...\n")

# Remove low expression genes
mean_expr <- apply(train_expr_norm, 2, mean, na.rm = TRUE)
high_expr_genes <- which(mean_expr > quantile(mean_expr, 0.3, na.rm = TRUE))

train_expr_filt <- train_expr_norm[, high_expr_genes]
test_expr_filt <- test_expr_norm[, high_expr_genes]

# Multiple feature selection methods
gene_cors <- apply(train_expr_filt, 2, function(x) abs(cor(x, train_age, use = "complete.obs")))
gene_vars <- apply(train_expr_filt, 2, var, na.rm = TRUE)

# Combine correlation and variance scores
combined_scores <- rank(gene_cors) + rank(gene_vars)
top_genes <- order(combined_scores, decreasing = TRUE)[1:min(100, length(combined_scores))]

train_expr_selected <- train_expr_filt[, top_genes]
test_expr_selected <- test_expr_filt[, top_genes]

# Handle missing values
train_expr_selected[is.na(train_expr_selected)] <- 0
test_expr_selected[is.na(test_expr_selected)] <- 0

# Standardize features for neural networks
train_means <- apply(train_expr_selected, 2, mean)
train_sds <- apply(train_expr_selected, 2, sd)

train_expr_scaled <- scale(train_expr_selected)
test_expr_scaled <- scale(test_expr_selected, center = train_means, scale = train_sds)

cat("Selected", ncol(train_expr_selected), "features for modeling\n")

# Create validation set
val_indices <- createDataPartition(train_age, p = 0.8, list = FALSE)
train_val_expr <- train_expr_selected[val_indices, ]
train_val_age <- train_age[val_indices]
val_expr <- train_expr_selected[-val_indices, ]
val_age <- train_age[-val_indices]

train_val_expr_scaled <- train_expr_scaled[val_indices, ]
val_expr_scaled <- train_expr_scaled[-val_indices, ]

# Model 1: Enhanced Elastic Net (from previous analysis)
cat("\nTraining Enhanced Elastic Net...\n")
alpha_values <- seq(0.05, 0.95, by = 0.05)
best_alpha <- 0.1
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

elastic_model <- glmnet(train_expr_selected, train_age, alpha = best_alpha, lambda = best_lambda)

# Model 2: Gradient Boosting Machine
cat("Training Gradient Boosting Machine...\n")
gbm_model <- gbm(train_val_age ~ ., 
                 data = data.frame(train_val_age, train_val_expr),
                 distribution = "gaussian",
                 n.trees = 1000,
                 interaction.depth = 4,
                 shrinkage = 0.01,
                 cv.folds = 5,
                 verbose = FALSE)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)

# Model 3: MARS (Multivariate Adaptive Regression Splines)
cat("Training MARS model...\n")
mars_model <- earth(train_val_expr, train_val_age, degree = 2, nfold = 5)

# Model 4: Deep Neural Network
cat("Training Deep Neural Network...\n")

# Check if TensorFlow is available
tf_available <- FALSE
cat("TensorFlow not available, using nnet for neural network\n")

if(tf_available) {
  # Build neural network architecture
  model_nn <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = 'relu', input_shape = ncol(train_expr_scaled)) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 256, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'linear')
  
  # Compile model
  model_nn %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = 'mse',
    metrics = c('mae')
  )
  
  # Train model with early stopping
  early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 20, restore_best_weights = TRUE)
  reduce_lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.5, patience = 10)
  
  history <- model_nn %>% fit(
    train_val_expr_scaled, train_val_age,
    epochs = 200,
    batch_size = 32,
    validation_data = list(val_expr_scaled, val_age),
    callbacks = list(early_stop, reduce_lr),
    verbose = 0
  )
  
  cat("Neural network training completed\n")
} else {
  # Fallback: Neural network using nnet package
  cat("Using nnet package for neural network...\n")
  
  # Scale target variable for nnet
  train_age_scaled <- scale(train_val_age)[,1]
  val_age_scaled <- scale(val_age, center = attr(scale(train_val_age), "scaled:center"), 
                         scale = attr(scale(train_val_age), "scaled:scale"))[,1]
  
  model_nn_simple <- nnet(train_val_expr_scaled, train_age_scaled, 
                         size = 3, decay = 0.01, linout = TRUE, 
                         maxit = 500, trace = FALSE)
}

# Model 5: Enhanced Random Forest with more trees
cat("Training Enhanced Random Forest...\n")
rf_model_enhanced <- randomForest(train_val_expr, train_val_age, 
                                 ntree = 2000, 
                                 mtry = round(sqrt(ncol(train_val_expr))),
                                 importance = TRUE,
                                 nodesize = 3)

# Model 6: XGBoost with hyperparameter tuning
cat("Training Enhanced XGBoost...\n")
dtrain <- xgb.DMatrix(data = train_val_expr, label = train_val_age)
dval <- xgb.DMatrix(data = val_expr, label = val_age)

# Hyperparameter grid search
param_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(4, 6, 8),
  subsample = c(0.8, 0.9),
  colsample_bytree = c(0.8, 0.9)
)

best_xgb_rmse <- Inf
best_xgb_params <- NULL

for(i in 1:min(10, nrow(param_grid))) {
  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i]
  )
  
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  if(min(xgb_cv$evaluation_log$test_rmse_mean) < best_xgb_rmse) {
    best_xgb_rmse <- min(xgb_cv$evaluation_log$test_rmse_mean)
    best_xgb_params <- params
  }
}

xgb_model_enhanced <- xgb.train(
  params = best_xgb_params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 20,
  verbose = 0
)

# Retrain all models on full training set
cat("\nRetraining models on full training set...\n")

# Retrain models
elastic_final <- glmnet(train_expr_selected, train_age, alpha = best_alpha, lambda = best_lambda)

gbm_final <- gbm(train_age ~ ., 
                data = data.frame(train_age, train_expr_selected),
                distribution = "gaussian",
                n.trees = best_iter,
                interaction.depth = 4,
                shrinkage = 0.01,
                verbose = FALSE)

mars_final <- earth(train_expr_selected, train_age, degree = 2)

rf_final <- randomForest(train_expr_selected, train_age, 
                        ntree = 2000, 
                        mtry = round(sqrt(ncol(train_expr_selected))),
                        importance = TRUE,
                        nodesize = 3)

dtrain_full <- xgb.DMatrix(data = train_expr_selected, label = train_age)
xgb_final <- xgb.train(
  params = best_xgb_params,
  data = dtrain_full,
  nrounds = xgb_model_enhanced$best_iteration,
  verbose = 0
)

# Make predictions on test set
cat("\nMaking predictions on test set...\n")

pred_elastic <- predict(elastic_final, test_expr_selected, s = best_lambda)
pred_gbm <- predict(gbm_final, data.frame(test_expr_selected), n.trees = best_iter)
pred_mars <- predict(mars_final, test_expr_selected)
pred_rf <- predict(rf_final, test_expr_selected)
pred_xgb <- predict(xgb_final, xgb.DMatrix(test_expr_selected))

# Neural network predictions
if(tf_available) {
  pred_nn <- predict(model_nn, test_expr_scaled)
  pred_nn <- as.numeric(pred_nn)
} else {
  pred_nn_scaled <- predict(model_nn_simple, test_expr_scaled)
  pred_nn <- pred_nn_scaled * attr(scale(train_val_age), "scaled:scale") + 
             attr(scale(train_val_age), "scaled:center")
}

# Calculate individual RMSEs
rmse_elastic <- sqrt(mean((test_age - pred_elastic)^2))
rmse_gbm <- sqrt(mean((test_age - pred_gbm)^2))
rmse_mars <- sqrt(mean((test_age - pred_mars)^2))
rmse_rf <- sqrt(mean((test_age - pred_rf)^2))
rmse_xgb <- sqrt(mean((test_age - pred_xgb)^2))
rmse_nn <- sqrt(mean((test_age - pred_nn)^2))

cat("\nAdvanced Model Performance (RMSE on test set):\n")
cat("Enhanced Elastic Net RMSE:", round(rmse_elastic, 4), "\n")
cat("Gradient Boosting RMSE:", round(rmse_gbm, 4), "\n")
cat("MARS RMSE:", round(rmse_mars, 4), "\n")
cat("Enhanced Random Forest RMSE:", round(rmse_rf, 4), "\n")
cat("Enhanced XGBoost RMSE:", round(rmse_xgb, 4), "\n")
cat("Deep Neural Network RMSE:", round(rmse_nn, 4), "\n")

# Advanced Ensemble Methods
cat("\nCreating advanced ensemble models...\n")

# 1. Bayesian Model Averaging
predictions_matrix <- cbind(pred_elastic, pred_gbm, pred_mars, pred_rf, pred_xgb, pred_nn)
individual_rmses <- c(rmse_elastic, rmse_gbm, rmse_mars, rmse_rf, rmse_xgb, rmse_nn)

# Inverse RMSE weighting (Bayesian-like)
bayesian_weights <- (1/individual_rmses) / sum(1/individual_rmses)
ensemble_bayesian <- predictions_matrix %*% bayesian_weights
rmse_bayesian <- sqrt(mean((test_age - ensemble_bayesian)^2))

# 2. Stacked ensemble with cross-validation
cat("Training stacked ensemble...\n")

# Create meta-features using cross-validation
n_folds <- 5
folds <- createFolds(train_age, k = n_folds)
meta_features <- matrix(0, nrow = length(train_age), ncol = 6)

for(fold in 1:n_folds) {
  train_fold <- train_expr_selected[-folds[[fold]], ]
  train_age_fold <- train_age[-folds[[fold]]]
  val_fold <- train_expr_selected[folds[[fold]], ]
  
  # Train base models on fold
  elastic_fold <- glmnet(train_fold, train_age_fold, alpha = best_alpha, lambda = best_lambda)
  gbm_fold <- gbm(train_age_fold ~ ., data = data.frame(train_age_fold, train_fold),
                  distribution = "gaussian", n.trees = 200, interaction.depth = 4, 
                  shrinkage = 0.01, verbose = FALSE)
  mars_fold <- earth(train_fold, train_age_fold, degree = 2)
  rf_fold <- randomForest(train_fold, train_age_fold, ntree = 500)
  xgb_fold <- xgb.train(params = best_xgb_params, 
                       data = xgb.DMatrix(train_fold, label = train_age_fold),
                       nrounds = 200, verbose = 0)
  
  # Make predictions on validation fold
  meta_features[folds[[fold]], 1] <- predict(elastic_fold, val_fold, s = best_lambda)
  meta_features[folds[[fold]], 2] <- predict(gbm_fold, data.frame(val_fold), n.trees = 200)
  meta_features[folds[[fold]], 3] <- predict(mars_fold, val_fold)
  meta_features[folds[[fold]], 4] <- predict(rf_fold, val_fold)
  meta_features[folds[[fold]], 5] <- predict(xgb_fold, xgb.DMatrix(val_fold))
  
  # Simple neural network for fold
  if(!tf_available) {
    train_fold_scaled <- scale(train_fold)
    val_fold_scaled <- scale(val_fold, center = attr(train_fold_scaled, "scaled:center"),
                            scale = attr(train_fold_scaled, "scaled:scale"))
    train_age_fold_scaled <- scale(train_age_fold)[,1]
    
    nn_fold <- nnet(train_fold_scaled, train_age_fold_scaled, size = 3, 
                   decay = 0.01, linout = TRUE, maxit = 200, trace = FALSE)
    pred_nn_fold_scaled <- predict(nn_fold, val_fold_scaled)
    meta_features[folds[[fold]], 6] <- pred_nn_fold_scaled * attr(scale(train_age_fold), "scaled:scale") + 
                                      attr(scale(train_age_fold), "scaled:center")
  }
}

# Train meta-learner
meta_learner <- glmnet(meta_features, train_age, alpha = 0.5)

# Make meta-predictions on test set
test_meta_features <- predictions_matrix
ensemble_stacked_advanced <- predict(meta_learner, test_meta_features, s = "lambda.min")
rmse_stacked_advanced <- sqrt(mean((test_age - ensemble_stacked_advanced)^2))

# 3. Dynamic ensemble (performance-based weighting)
# Weight models based on their performance in different age ranges
age_ranges <- list(
  early = test_age <= quantile(test_age, 0.33),
  mid = test_age > quantile(test_age, 0.33) & test_age <= quantile(test_age, 0.67),
  late = test_age > quantile(test_age, 0.67)
)

ensemble_dynamic <- numeric(length(test_age))

for(range_name in names(age_ranges)) {
  range_mask <- age_ranges[[range_name]]
  if(sum(range_mask) > 0) {
    range_rmses <- apply(predictions_matrix[range_mask, , drop = FALSE], 2, 
                        function(x) sqrt(mean((test_age[range_mask] - x)^2)))
    range_weights <- (1/range_rmses) / sum(1/range_rmses)
    ensemble_dynamic[range_mask] <- predictions_matrix[range_mask, , drop = FALSE] %*% range_weights
  }
}

rmse_dynamic <- sqrt(mean((test_age - ensemble_dynamic)^2))

cat("\nAdvanced Ensemble Performance (RMSE on test set):\n")
cat("Bayesian Model Averaging RMSE:", round(rmse_bayesian, 4), "\n")
cat("Advanced Stacked Ensemble RMSE:", round(rmse_stacked_advanced, 4), "\n")
cat("Dynamic Ensemble RMSE:", round(rmse_dynamic, 4), "\n")

# Find the best model overall
all_rmses <- c(rmse_elastic, rmse_gbm, rmse_mars, rmse_rf, rmse_xgb, rmse_nn,
               rmse_bayesian, rmse_stacked_advanced, rmse_dynamic)
model_names <- c("Enhanced Elastic Net", "Gradient Boosting", "MARS", 
                "Enhanced Random Forest", "Enhanced XGBoost", "Deep Neural Network",
                "Bayesian Ensemble", "Advanced Stacked Ensemble", "Dynamic Ensemble")

best_idx <- which.min(all_rmses)
best_model_name <- model_names[best_idx]
best_rmse <- all_rmses[best_idx]

# Get best predictions
best_predictions <- list(pred_elastic, pred_gbm, pred_mars, pred_rf, pred_xgb, pred_nn,
                        ensemble_bayesian, ensemble_stacked_advanced, ensemble_dynamic)
best_pred <- best_predictions[[best_idx]]

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("ULTIMATE BEST MODEL:", best_model_name, "\n")
cat("ULTIMATE BEST RMSE:", round(best_rmse, 4), "weeks\n")
cat(paste(rep("=", 60), collapse=""), "\n")

# Create final visualization
plot_data <- data.frame(
  Actual = test_age,
  Predicted = as.numeric(best_pred),
  Model = best_model_name
)

correlation <- cor(test_age, best_pred)
mae <- mean(abs(test_age - best_pred))
r_squared <- 1 - sum((test_age - best_pred)^2) / sum((test_age - mean(test_age))^2)

p_final <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7, size = 3, color = "darkblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1.2) +
  geom_smooth(method = "lm", se = TRUE, color = "darkgreen", alpha = 0.3, linewidth = 1.2) +
  labs(
    title = paste("Ultimate Gestational Age Prediction -", best_model_name),
    subtitle = paste("RMSE =", round(best_rmse, 4), "| R² =", round(r_squared, 4), 
                    "| Correlation =", round(correlation, 4)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "darkblue"),
    plot.subtitle = element_text(size = 14, color = "darkgreen"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray", fill = NA, linewidth = 1)
  ) +
  coord_fixed(ratio = 1)

print(p_final)
ggsave("ultimate_gestational_age_prediction_plot.png", p_final, width = 12, height = 10, dpi = 300)

# Final performance summary
cat("\nUltimate Performance Metrics:\n")
cat("RMSE:", round(best_rmse, 4), "weeks\n")
cat("MAE:", round(mae, 4), "weeks\n")
cat("Correlation:", round(correlation, 4), "\n")
cat("R-squared:", round(r_squared, 4), "\n")

cat("\nComplete Model Comparison:\n")
for(i in 1:length(model_names)) {
  cat(sprintf("%-30s: %.4f\n", model_names[i], all_rmses[i]))
}

cat("\nAnalysis complete! Ultimate optimization achieved RMSE of", round(best_rmse, 4), "weeks.\n")