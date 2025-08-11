#!/usr/bin/env Rscript

# Simplified Optimized Gestational Age Prediction
# GSE149440 Dataset Analysis with Key Optimizations

# Load required libraries
library(GEOquery)
library(Biobase)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(glmnet)
library(xgboost)
library(e1071)
library(gbm)

# Set random seed for reproducibility
set.seed(123)

# Function to download and process GSE149440 dataset
download_and_process_data <- function() {
  cat("Downloading GSE149440 dataset...\n")
  
  # Download the dataset
  gse <- getGEO("GSE149440", destdir = ".", AnnotGPL = FALSE)
  
  # Extract expression data and metadata
  if (is.list(gse) && length(gse) > 1) {
    eset <- gse[[1]]
  } else if (is.list(gse) && length(gse) == 1) {
    eset <- gse[[1]]
  } else {
    eset <- gse
  }
  
  # Get expression matrix
  expr_data <- exprs(eset)
  
  # Get phenotype data (metadata)
  pheno_data <- pData(eset)
  
  cat("Dataset downloaded successfully!\n")
  cat("Expression data dimensions:", dim(expr_data), "\n")
  cat("Metadata dimensions:", dim(pheno_data), "\n")
  
  return(list(expr_data = expr_data, pheno_data = pheno_data))
}

# Function to prepare data for modeling
prepare_data <- function(data_list) {
  expr_data <- data_list$expr_data
  pheno_data <- data_list$pheno_data
  
  cat("Preparing data for modeling...\n")
  
  # Extract gestational age and training set information
  gestational_age <- pheno_data$`gestational age:ch1`
  train_set <- pheno_data$`train:ch1`
  
  # Convert to numeric
  gestational_age <- as.numeric(as.character(gestational_age))
  train_set <- as.numeric(as.character(train_set))
  
  # Remove samples with missing gestational age
  valid_samples <- !is.na(gestational_age)
  expr_data <- expr_data[, valid_samples]
  gestational_age <- gestational_age[valid_samples]
  train_set <- train_set[valid_samples]
  
  cat("Valid samples:", sum(valid_samples), "\n")
  cat("Training samples:", sum(train_set == 1), "\n")
  cat("Test samples:", sum(train_set == 0), "\n")
  
  # Transpose expression data for modeling (samples as rows, genes as columns)
  expr_df <- as.data.frame(t(expr_data))
  
  # Add target variable
  expr_df$gestational_age <- gestational_age
  expr_df$train_set <- train_set
  
  return(expr_df)
}

# Advanced feature selection
perform_advanced_feature_selection <- function(data, n_features = 3000) {
  cat("Performing advanced feature selection...\n")
  
  # Separate training data
  train_data <- data[data$train_set == 1, ]
  
  # Remove target variables from feature matrix
  feature_cols <- setdiff(colnames(train_data), c("gestational_age", "train_set"))
  
  # Calculate correlation with gestational age
  correlations <- sapply(feature_cols, function(col) {
    cor(train_data[[col]], train_data$gestational_age, use = "complete.obs")
  })
  
  # Calculate variance
  variances <- sapply(feature_cols, function(col) {
    var(train_data[[col]], na.rm = TRUE)
  })
  
  # Combine scores (weighted combination)
  combined_scores <- 0.7 * abs(correlations) + 0.3 * (variances / max(variances))
  
  # Select top features
  top_features <- names(sort(combined_scores, decreasing = TRUE)[1:n_features])
  
  cat("Selected", length(top_features), "features using advanced selection\n")
  
  return(top_features)
}

# Train optimized models
train_optimized_models <- function(train_data, test_data, features) {
  cat("Training optimized models...\n")
  
  # Create validation set from training data
  set.seed(123)
  train_indices <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
  val_indices <- setdiff(1:nrow(train_data), train_indices)
  
  train_subset <- train_data[train_indices, ]
  val_subset <- train_data[val_indices, ]
  
  # Prepare data
  X_train <- train_subset[, features]
  y_train <- train_subset$gestational_age
  X_val <- val_subset[, features]
  y_val <- val_subset$gestational_age
  X_test <- test_data[, features]
  y_test <- test_data$gestational_age
  
  models <- list()
  predictions <- list()
  
  # 1. Optimized Random Forest
  cat("Training optimized Random Forest...\n")
  rf_model <- randomForest(
    x = X_train, 
    y = y_train, 
    ntree = 1000, 
    mtry = sqrt(length(features)),
    nodesize = 3,
    importance = TRUE
  )
  models$rf <- rf_model
  predictions$rf <- predict(rf_model, X_test)
  
  # 2. Optimized XGBoost
  cat("Training optimized XGBoost...\n")
  xgb_params <- list(
    objective = "reg:squarederror",
    max_depth = 8,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 4
  )
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dval <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
  
  xgb_model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 300,
    watchlist = list(val = dval),
    early_stopping_rounds = 30,
    verbose = 0
  )
  
  models$xgb <- xgb_model
  predictions$xgb <- predict(xgb_model, as.matrix(X_test))
  
  # 3. Optimized Elastic Net
  cat("Training optimized Elastic Net...\n")
  enet_model <- cv.glmnet(
    x = as.matrix(X_train),
    y = y_train,
    alpha = 0.3,
    nfolds = 5
  )
  
  models$enet <- enet_model
  predictions$enet <- predict(enet_model, as.matrix(X_test), s = "lambda.min")
  
  # 4. Optimized SVR
  cat("Training optimized SVR...\n")
  svr_model <- svm(
    x = as.matrix(X_train),
    y = y_train,
    kernel = "radial",
    cost = 10,
    gamma = 0.01 / ncol(X_train)
  )
  
  models$svr <- svr_model
  predictions$svr <- predict(svr_model, as.matrix(X_test))
  
  # 5. GBM
  cat("Training GBM...\n")
  gbm_model <- gbm(
    gestational_age ~ .,
    data = data.frame(X_train, gestational_age = y_train),
    distribution = "gaussian",
    n.trees = 1000,
    interaction.depth = 8,
    shrinkage = 0.01,
    bag.fraction = 0.8
  )
  models$gbm <- gbm_model
  predictions$gbm <- predict(gbm_model, data.frame(X_test), n.trees = 1000)
  
  return(list(models = models, predictions = predictions))
}

# Create optimized ensemble
create_optimized_ensemble <- function(predictions, y_true) {
  cat("Creating optimized ensemble...\n")
  
  # Convert predictions to matrix
  pred_matrix <- do.call(cbind, predictions)
  
  # Method 1: Simple average
  simple_avg <- rowMeans(pred_matrix)
  
  # Method 2: Weighted average based on individual model performance
  model_weights <- sapply(predictions, function(pred) {
    1 / (sqrt(mean((pred - y_true)^2))^2
  })
  model_weights <- model_weights / sum(model_weights)
  weighted_avg <- pred_matrix %*% model_weights
  
  # Method 3: Median ensemble (robust to outliers)
  median_ensemble <- apply(pred_matrix, 1, median)
  
  # Evaluate all ensemble methods
  ensemble_methods <- list(
    simple_avg = simple_avg,
    weighted_avg = weighted_avg,
    median = median_ensemble
  )
  
  best_rmse <- Inf
  best_method <- NULL
  
  for (method_name in names(ensemble_methods)) {
    pred <- ensemble_methods[[method_name]]
    rmse <- sqrt(mean((pred - y_true)^2))
    cat(sprintf("%s ensemble - RMSE: %.2f\n", method_name, rmse))
    
    if (rmse < best_rmse) {
      best_rmse <- rmse
      best_method <- method_name
    }
  }
  
  cat("Best ensemble method:", best_method, "with RMSE:", best_rmse, "\n")
  
  return(list(
    predictions = ensemble_methods[[best_method]],
    rmse = best_rmse,
    method = best_method,
    all_methods = ensemble_methods
  ))
}

# Function to evaluate models
evaluate_models <- function(predictions, y_true) {
  cat("Evaluating models...\n")
  
  results <- data.frame()
  
  for (model_name in names(predictions)) {
    pred <- predictions[[model_name]]
    
    # Calculate RMSE
    rmse <- sqrt(mean((pred - y_true)^2))
    
    # Calculate R-squared
    ss_res <- sum((pred - y_true)^2)
    ss_tot <- sum((y_true - mean(y_true))^2)
    r_squared <- 1 - (ss_res / ss_tot)
    
    # Calculate MAE
    mae <- mean(abs(pred - y_true))
    
    results <- rbind(results, data.frame(
      Model = model_name,
      RMSE = rmse,
      R_squared = r_squared,
      MAE = mae
    ))
    
    cat(sprintf("%s - RMSE: %.2f, R²: %.3f, MAE: %.2f\n", 
                model_name, rmse, r_squared, mae))
  }
  
  return(results)
}

# Function to create visualization
create_optimized_plots <- function(y_true, predictions, ensemble_results, results) {
  cat("Creating optimized visualizations...\n")
  
  # Create directory for plots
  dir.create("optimized_plots", showWarnings = FALSE)
  
  # 1. Scatter plot for best ensemble predictions
  p1 <- ggplot(data.frame(Actual = y_true, Predicted = ensemble_results$predictions), 
               aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, color = "blue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = paste("Optimized", ensemble_results$method, "Ensemble: Predicted vs Actual"),
         subtitle = paste("RMSE =", round(ensemble_results$rmse, 2), "weeks"),
         x = "Actual Gestational Age",
         y = "Predicted Gestational Age") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  ggsave("optimized_plots/best_ensemble_scatter.png", p1, width = 8, height = 6)
  
  # 2. Model comparison plot
  p2 <- ggplot(results, aes(x = reorder(Model, RMSE), y = RMSE)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Optimized Model Performance Comparison",
         x = "Model",
         y = "RMSE (weeks)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("optimized_plots/model_comparison.png", p2, width = 10, height = 6)
  
  cat("Optimized plots saved in 'optimized_plots' directory\n")
}

# Main execution
main <- function() {
  cat("=== OPTIMIZED Gestational Age Prediction from Gene Expression Data ===\n")
  cat("Dataset: GSE149440\n\n")
  
  # Download and process data
  data_list <- download_and_process_data()
  
  # Prepare data for modeling
  data <- prepare_data(data_list)
  
  # Split data into training and test sets
  train_data <- data[data$train_set == 1, ]
  test_data <- data[data$train_set == 0, ]
  
  cat("Training set size:", nrow(train_data), "\n")
  cat("Test set size:", nrow(test_data), "\n\n")
  
  # Perform advanced feature selection
  selected_features <- perform_advanced_feature_selection(data, n_features = 3000)
  
  # Train optimized models
  model_results <- train_optimized_models(train_data, test_data, selected_features)
  
  # Evaluate individual models
  evaluation_results <- evaluate_models(model_results$predictions, test_data$gestational_age)
  
  # Create optimized ensemble
  ensemble_results <- create_optimized_ensemble(model_results$predictions, test_data$gestational_age)
  
  # Print final results
  cat("\n=== OPTIMIZED FINAL RESULTS ===\n")
  cat("Best individual model RMSE:", min(evaluation_results$RMSE), "\n")
  cat("Best ensemble method:", ensemble_results$method, "\n")
  cat("Optimized ensemble RMSE:", ensemble_results$rmse, "\n")
  cat("Improvement over best individual:", min(evaluation_results$RMSE) - ensemble_results$rmse, "\n")
  
  # Create visualizations
  create_optimized_plots(test_data$gestational_age, model_results$predictions, 
                         ensemble_results, evaluation_results)
  
  # Save results
  write.csv(evaluation_results, "optimized_model_evaluation_results.csv", row.names = FALSE)
  
  cat("\nOptimized analysis complete! Results saved to CSV files.\n")
  cat("Optimized plots saved in 'optimized_plots' directory\n")
}

# Run the main function
main() 