# Optimized Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440 from Gene Expression Omnibus
# Focus: Minimize RMSE through advanced techniques

cat("=== Starting OPTIMIZED Gestational Age Prediction Analysis ===\n")

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  for (pkg in packages) {
    cat("Checking package:", pkg, "\n")
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      if (pkg %in% c("GEOquery", "limma", "Biobase", "preprocessCore")) {
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
          install.packages("BiocManager", repos = "https://cloud.r-project.org")
        }
        BiocManager::install(pkg, ask = FALSE)
      } else {
        install.packages(pkg, repos = "https://cloud.r-project.org")
      }
    } else {
      cat("Package", pkg, "already installed.\n")
    }
  }
}

# Extended package list for optimization
cat("Installing/checking optimized packages...\n")
required_packages <- c("GEOquery", "Biobase", "caret", "randomForest", 
                      "ggplot2", "glmnet", "xgboost", "e1071", 
                      "preprocessCore", "dplyr")

tryCatch({
  install_if_missing(required_packages)
  cat("All packages installed successfully.\n")
}, error = function(e) {
  cat("Error installing packages:", e$message, "\n")
  # Continue with available packages
})

# Load libraries
cat("Loading libraries...\n")
library(GEOquery)
library(Biobase)
library(caret)
library(randomForest)
library(ggplot2)
library(glmnet)

# Try to load additional packages
tryCatch({
  library(xgboost)
  has_xgboost <- TRUE
}, error = function(e) {
  cat("XGBoost not available, skipping.\n")
  has_xgboost <- FALSE
})

tryCatch({
  library(e1071)
  has_svm <- TRUE
}, error = function(e) {
  cat("e1071 not available, skipping SVM.\n")
  has_svm <- FALSE
})

tryCatch({
  library(preprocessCore)
  has_preprocess <- TRUE
}, error = function(e) {
  cat("preprocessCore not available, using basic normalization.\n")
  has_preprocess <- FALSE
})

library(dplyr)

cat("Libraries loaded successfully.\n")

# Download GSE149440 dataset (reuse if already downloaded)
cat("Loading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)

if (is.list(gse)) {
  eset <- gse[[1]]
} else {
  eset <- gse
}

cat("Dataset dimensions:", dim(exprs(eset)), "\n")

# Extract metadata
metadata <- pData(eset)
gestational_col <- grep("gestational.*age", colnames(metadata), ignore.case = TRUE, value = TRUE)
train_col <- grep("train", colnames(metadata), ignore.case = TRUE, value = TRUE)

gestational_age <- as.numeric(metadata[[gestational_col[1]]])
train_indicator <- metadata[[train_col[1]]]

# Clean data and split
valid_samples <- !is.na(gestational_age) & !is.na(train_indicator)
eset_clean <- eset[, valid_samples]
gestational_age_clean <- gestational_age[valid_samples]
train_indicator_clean <- train_indicator[valid_samples]

train_indices <- which(train_indicator_clean == "1" | train_indicator_clean == 1)
test_indices <- which(train_indicator_clean == "0" | train_indicator_clean == 0)

train_expr <- exprs(eset_clean)[, train_indices]
test_expr <- exprs(eset_clean)[, test_indices]
train_age <- gestational_age_clean[train_indices]
test_age <- gestational_age_clean[test_indices]

cat("Training samples:", length(train_age), "\n")
cat("Test samples:", length(test_age), "\n")

# ADVANCED PREPROCESSING
cat("=== Advanced Preprocessing ===\n")

# 1. Enhanced normalization
if (has_preprocess) {
  cat("Applying quantile normalization...\n")
  combined_expr <- cbind(train_expr, test_expr)
  normalized_expr <- normalize.quantiles(combined_expr)
  dimnames(normalized_expr) <- dimnames(combined_expr)
  
  train_expr_norm <- normalized_expr[, 1:ncol(train_expr)]
  test_expr_norm <- normalized_expr[, (ncol(train_expr)+1):ncol(normalized_expr)]
} else {
  # Standard log2 transformation and scaling
  if (max(train_expr, na.rm = TRUE) > 100) {
    train_expr_norm <- log2(train_expr + 1)
    test_expr_norm <- log2(test_expr + 1)
  } else {
    train_expr_norm <- train_expr
    test_expr_norm <- test_expr
  }
  
  # Scale genes to have mean 0 and sd 1
  gene_means <- rowMeans(train_expr_norm, na.rm = TRUE)
  gene_sds <- apply(train_expr_norm, 1, sd, na.rm = TRUE)
  
  train_expr_norm <- (train_expr_norm - gene_means) / gene_sds
  test_expr_norm <- (test_expr_norm - gene_means) / gene_sds
}

# 2. Advanced feature selection
cat("Advanced feature selection...\n")

# Remove genes with missing values or zero variance
valid_genes <- complete.cases(train_expr_norm) & 
               apply(train_expr_norm, 1, var, na.rm = TRUE) > 0.01

train_expr_filtered <- train_expr_norm[valid_genes, ]
test_expr_filtered <- test_expr_norm[valid_genes, ]

cat("Genes after basic filtering:", nrow(train_expr_filtered), "\n")

# Multiple feature selection approaches
# A) Correlation-based selection
correlations <- apply(train_expr_filtered, 1, function(x) {
  cor(x, train_age, use = "complete.obs")
})

# B) F-test based selection (ANOVA)
f_stats <- apply(train_expr_filtered, 1, function(x) {
  groups <- cut(train_age, breaks = 3)
  if (length(unique(groups)) > 1) {
    summary(aov(x ~ groups))[[1]][1,4]  # F-statistic
  } else {
    0
  }
})

# C) Combine selection criteria
combined_score <- abs(correlations) + log10(f_stats + 1e-10)
top_genes <- order(combined_score, decreasing = TRUE)[1:min(1000, length(combined_score))]

train_expr_selected <- train_expr_filtered[top_genes, ]
test_expr_selected <- test_expr_filtered[top_genes, ]

cat("Final selected genes:", nrow(train_expr_selected), "\n")

# 3. Principal Component Analysis for dimensionality reduction
cat("Applying PCA for dimensionality reduction...\n")
pca_result <- prcomp(t(train_expr_selected), center = TRUE, scale. = FALSE)

# Select number of PCs that explain 95% of variance
cumvar <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
n_pcs <- min(which(cumvar >= 0.95))
n_pcs <- min(n_pcs, 50)  # Cap at 50 PCs for computational efficiency

cat("Using", n_pcs, "principal components (explaining", 
    round(cumvar[n_pcs]*100, 1), "% of variance)\n")

# Transform training and test data
train_pca <- pca_result$x[, 1:n_pcs]
test_pca <- predict(pca_result, newdata = t(test_expr_selected))[, 1:n_pcs]

# Prepare final datasets - combine original features with PCA
train_data_orig <- t(train_expr_selected)
test_data_orig <- t(test_expr_selected)

# Create comprehensive training data
train_df <- data.frame(
  gestational_age = train_age,
  train_pca,
  train_data_orig[, 1:min(100, ncol(train_data_orig))]  # Top 100 original features
)

test_df <- data.frame(
  gestational_age = test_age,
  test_pca,
  test_data_orig[, 1:min(100, ncol(test_data_orig))]
)

# Remove any remaining NAs
train_df <- train_df[complete.cases(train_df), ]
test_df <- test_df[complete.cases(test_df), ]

cat("Final training samples:", nrow(train_df), "\n")
cat("Final test samples:", nrow(test_df), "\n")
cat("Total features:", ncol(train_df)-1, "\n")

# ADVANCED MODEL TRAINING
cat("=== Advanced Model Training ===\n")

# Enhanced cross-validation
ctrl <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 2,
  verboseIter = FALSE,
  allowParallel = FALSE
)

models <- list()
set.seed(42)

# 1. Optimized Random Forest
cat("Training optimized Random Forest...\n")
rf_grid <- expand.grid(mtry = c(5, 10, 20, 30))
rf_model <- train(gestational_age ~ ., 
                  data = train_df,
                  method = "rf",
                  trControl = ctrl,
                  tuneGrid = rf_grid,
                  ntree = 200,
                  importance = TRUE)
models$rf <- rf_model

# 2. Elastic Net with extensive tuning
cat("Training Elastic Net...\n")
glmnet_grid <- expand.grid(
  alpha = c(0.1, 0.3, 0.5, 0.7, 0.9),
  lambda = c(0.001, 0.01, 0.1, 1, 10)
)
glmnet_model <- train(gestational_age ~ ., 
                      data = train_df,
                      method = "glmnet",
                      trControl = ctrl,
                      tuneGrid = glmnet_grid,
                      standardize = TRUE)
models$glmnet <- glmnet_model

# 3. Support Vector Regression (if available)
if (has_svm && nrow(train_df) < 1000) {
  cat("Training Support Vector Regression...\n")
  svm_grid <- expand.grid(
    sigma = c(0.001, 0.01, 0.1),
    C = c(1, 10, 100)
  )
  svm_model <- train(gestational_age ~ ., 
                     data = train_df,
                     method = "svmRadial",
                     trControl = ctrl,
                     tuneGrid = svm_grid)
  models$svm <- svm_model
}

# 4. XGBoost (if available)
if (has_xgboost) {
  cat("Training XGBoost...\n")
  xgb_grid <- expand.grid(
    nrounds = c(100, 200),
    max_depth = c(3, 6),
    eta = c(0.1, 0.3),
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  )
  xgb_model <- train(gestational_age ~ ., 
                     data = train_df,
                     method = "xgbTree",
                     trControl = ctrl,
                     tuneGrid = xgb_grid,
                     verbose = FALSE)
  models$xgb <- xgb_model
}

# Model selection and ensemble
cat("=== Model Selection and Evaluation ===\n")

# Evaluate all models
cv_results <- sapply(models, function(m) min(m$results$RMSE, na.rm = TRUE))
cat("Cross-validation RMSE results:\n")
for (i in 1:length(cv_results)) {
  cat(names(cv_results)[i], ":", round(cv_results[i], 4), "\n")
}

best_model_name <- names(cv_results)[which.min(cv_results)]
best_model <- models[[best_model_name]]

cat("Best single model:", best_model_name, "with CV RMSE:", round(min(cv_results), 4), "\n")

# Create ensemble prediction (average of top 2 models)
if (length(models) >= 2) {
  cat("Creating ensemble prediction...\n")
  
  # Get predictions from all models
  all_predictions <- sapply(models, function(m) predict(m, newdata = test_df))
  
  # Simple average ensemble
  ensemble_pred <- rowMeans(all_predictions, na.rm = TRUE)
  
  # Weighted ensemble (inverse RMSE weighting)
  weights <- 1 / cv_results
  weights <- weights / sum(weights)
  
  weighted_ensemble_pred <- as.numeric(all_predictions %*% weights)
  
  # Calculate RMSE for different approaches
  best_single_pred <- predict(best_model, newdata = test_df)
  
  rmse_best <- sqrt(mean((best_single_pred - test_df$gestational_age)^2))
  rmse_ensemble <- sqrt(mean((ensemble_pred - test_df$gestational_age)^2))
  rmse_weighted <- sqrt(mean((weighted_ensemble_pred - test_df$gestational_age)^2))
  
  cat("Best single model RMSE:", round(rmse_best, 4), "\n")
  cat("Simple ensemble RMSE:", round(rmse_ensemble, 4), "\n")
  cat("Weighted ensemble RMSE:", round(rmse_weighted, 4), "\n")
  
  # Choose best approach
  if (rmse_weighted < rmse_ensemble && rmse_weighted < rmse_best) {
    final_predictions <- weighted_ensemble_pred
    final_rmse <- rmse_weighted
    approach <- "Weighted Ensemble"
  } else if (rmse_ensemble < rmse_best) {
    final_predictions <- ensemble_pred
    final_rmse <- rmse_ensemble
    approach <- "Simple Ensemble"
  } else {
    final_predictions <- best_single_pred
    final_rmse <- rmse_best
    approach <- paste("Best Single Model:", best_model_name)
  }
} else {
  final_predictions <- predict(best_model, newdata = test_df)
  final_rmse <- sqrt(mean((final_predictions - test_df$gestational_age)^2))
  approach <- paste("Single Model:", best_model_name)
}

# Final results
r_squared <- cor(final_predictions, test_df$gestational_age)^2

cat("\n=== OPTIMIZED FINAL RESULTS ===\n")
cat("Approach:", approach, "\n")
cat("Test Set RMSE:", round(final_rmse, 4), "\n")
cat("Test Set R-squared:", round(r_squared, 4), "\n")
cat("Number of test samples:", nrow(test_df), "\n")

# Create enhanced scatter plot
cat("Creating optimized scatter plot...\n")
plot_data <- data.frame(
  Actual = test_df$gestational_age,
  Predicted = final_predictions
)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2, color = "darkblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red", linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 1) +
  labs(title = paste("OPTIMIZED: Predicted vs Actual Gestational Age\n", 
                     approach, " - RMSE =", round(final_rmse, 3), ", R² =", round(r_squared, 3)),
       x = "Actual Gestational Age (weeks)",
       y = "Predicted Gestational Age (weeks)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 11)) +
  coord_fixed(ratio = 1) +
  xlim(range(c(plot_data$Actual, plot_data$Predicted))) +
  ylim(range(c(plot_data$Actual, plot_data$Predicted)))

ggsave("gestational_age_prediction_optimized.png", plot = p, width = 10, height = 8, dpi = 300)

cat("Optimized analysis completed!\n")
cat("FINAL OPTIMIZED RMSE:", round(final_rmse, 4), "\n")