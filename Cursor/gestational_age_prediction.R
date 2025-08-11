#!/usr/bin/env Rscript

# Gestational Age Prediction from Gene Expression Data
# GSE149440 Dataset Analysis

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
# library(parallel)
# library(doParallel)

# Set random seed for reproducibility
set.seed(123)

# Function to download and process GSE149440 dataset
download_and_process_data <- function() {
  cat("Downloading GSE149440 dataset...\n")
  
  # Download the dataset
  gse <- getGEO("GSE149440", destdir = ".", AnnotGPL = FALSE)
  
  # Extract expression data and metadata
  if (is.list(gse) && length(gse) > 1) {
    # If multiple platforms, use the first one
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

# Function to perform feature selection
perform_feature_selection <- function(data, n_features = 1000) {
  cat("Performing feature selection...\n")
  
  # Separate training data
  train_data <- data[data$train_set == 1, ]
  
  # Remove target variables from feature matrix
  feature_cols <- setdiff(colnames(train_data), c("gestational_age", "train_set"))
  
  # Calculate correlation with gestational age
  correlations <- sapply(feature_cols, function(col) {
    cor(train_data[[col]], train_data$gestational_age, use = "complete.obs")
  })
  
  # Select top features by absolute correlation
  top_features <- names(sort(abs(correlations), decreasing = TRUE)[1:n_features])
  
  cat("Selected", length(top_features), "features\n")
  
  return(top_features)
}

# Function to train multiple models
train_models <- function(train_data, test_data, features) {
  cat("Training multiple models...\n")
  
  # Prepare training data
  X_train <- train_data[, features]
  y_train <- train_data$gestational_age
  
  # Prepare test data
  X_test <- test_data[, features]
  y_test <- test_data$gestational_age
  
  models <- list()
  predictions <- list()
  
  # 1. Random Forest
  cat("Training Random Forest...\n")
  rf_model <- randomForest(x = X_train, y = y_train, ntree = 500, mtry = sqrt(length(features)))
  rf_pred <- predict(rf_model, X_test)
  
  models$rf <- rf_model
  predictions$rf <- rf_pred
  
  # 2. XGBoost
  cat("Training XGBoost...\n")
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
  
  xgb_params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 4
  )
  
  xgb_model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 100,
    watchlist = list(test = dtest),
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  xgb_pred <- predict(xgb_model, as.matrix(X_test))
  
  models$xgb <- xgb_model
  predictions$xgb <- xgb_pred
  
  # 3. Elastic Net
  cat("Training Elastic Net...\n")
  enet_model <- cv.glmnet(
    x = as.matrix(X_train),
    y = y_train,
    alpha = 0.5,
    nfolds = 5
  )
  
  enet_pred <- predict(enet_model, as.matrix(X_test), s = "lambda.min")
  
  models$enet <- enet_model
  predictions$enet <- enet_pred
  
  # 4. Support Vector Regression
  cat("Training SVR...\n")
  svr_model <- svm(
    x = as.matrix(X_train),
    y = y_train,
    kernel = "radial",
    cost = 1,
    gamma = 1/ncol(X_train)
  )
  
  svr_pred <- predict(svr_model, as.matrix(X_test))
  
  models$svr <- svr_model
  predictions$svr <- svr_pred
  
  return(list(models = models, predictions = predictions))
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

# Function to create ensemble model
create_ensemble <- function(predictions, y_true) {
  cat("Creating ensemble model...\n")
  
  # Simple average ensemble
  pred_matrix <- do.call(cbind, predictions)
  ensemble_pred <- rowMeans(pred_matrix)
  
  # Calculate ensemble metrics
  rmse <- sqrt(mean((ensemble_pred - y_true)^2))
  ss_res <- sum((ensemble_pred - y_true)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  mae <- mean(abs(ensemble_pred - y_true))
  
  cat(sprintf("Ensemble - RMSE: %.2f, R²: %.3f, MAE: %.2f\n", 
              rmse, r_squared, mae))
  
  return(list(predictions = ensemble_pred, rmse = rmse, r_squared = r_squared, mae = mae))
}

# Function to create visualization
create_plots <- function(y_true, predictions, ensemble_pred, results) {
  cat("Creating visualizations...\n")
  
  # Create directory for plots
  dir.create("plots", showWarnings = FALSE)
  
  # 1. Scatter plot for ensemble predictions
  plot_data <- data.frame(Actual = y_true, Predicted = ensemble_pred)
  p1 <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, color = "blue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = "Ensemble Model: Predicted vs Actual Gestational Age",
         x = "Actual Gestational Age",
         y = "Predicted Gestational Age") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  ggsave("plots/ensemble_prediction_scatter.png", p1, width = 8, height = 6)
  
  # 2. Model comparison plot
  p2 <- ggplot(results, aes(x = reorder(Model, RMSE), y = RMSE)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Model Performance Comparison (RMSE)",
         x = "Model",
         y = "RMSE") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("plots/model_comparison.png", p2, width = 8, height = 6)
  
  # 3. Individual model scatter plots
  for (model_name in names(predictions)) {
    plot_data <- data.frame(Actual = y_true, Predicted = predictions[[model_name]])
    p3 <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6, color = "darkgreen") +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(title = paste(model_name, "Model: Predicted vs Actual Gestational Age"),
           x = "Actual Gestational Age",
           y = "Predicted Gestational Age") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggsave(paste0("plots/", model_name, "_prediction_scatter.png"), p3, width = 8, height = 6)
  }
  
  cat("Plots saved in 'plots' directory\n")
}

# Main execution
main <- function() {
  cat("=== Gestational Age Prediction from Gene Expression Data ===\n")
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
  
  # Perform feature selection
  selected_features <- perform_feature_selection(data, n_features = 2000)
  
  # Train models
  model_results <- train_models(train_data, test_data, selected_features)
  
  # Evaluate individual models
  evaluation_results <- evaluate_models(model_results$predictions, test_data$gestational_age)
  
  # Create ensemble
  ensemble_results <- create_ensemble(model_results$predictions, test_data$gestational_age)
  
  # Print final results
  cat("\n=== FINAL RESULTS ===\n")
  cat("Best individual model RMSE:", min(evaluation_results$RMSE), "\n")
  cat("Ensemble model RMSE:", ensemble_results$rmse, "\n")
  
  # Create visualizations
  create_plots(test_data$gestational_age, model_results$predictions, 
               ensemble_results$predictions, evaluation_results)
  
  # Save results
  write.csv(evaluation_results, "model_evaluation_results.csv", row.names = FALSE)
  
  cat("\nAnalysis complete! Results saved to 'model_evaluation_results.csv'\n")
  cat("Plots saved in 'plots' directory\n")
}

# Run the main function
main() 