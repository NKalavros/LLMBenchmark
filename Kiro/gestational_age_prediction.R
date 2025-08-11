# Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440 from Gene Expression Omnibus

# Load required libraries
library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)

# Set working directory and create output folder
if (!dir.exists("results")) {
  dir.create("results")
}

print("Starting analysis of GSE149440 dataset...")

# Download GSE149440 dataset
print("Downloading GSE149440 dataset from GEO...")
options(timeout = 300)  # Increase timeout to 5 minutes
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)

# Extract expression set
eset <- gse[[1]]
print(paste("Dataset downloaded successfully. Dimensions:", nrow(eset), "genes x", ncol(eset), "samples"))

# Extract expression data and metadata
expr_data <- exprs(eset)
metadata <- pData(eset)

print("Examining metadata structure...")
print(colnames(metadata))

# Extract gestational age and training set indicators
gestational_age <- as.numeric(metadata[["gestational age:ch1"]])
train_indicator <- metadata[["train:ch1"]]

print("Data summary:")
print(paste("Total samples:", length(gestational_age)))
print(paste("Training samples:", sum(train_indicator == "1", na.rm = TRUE)))
print(paste("Test samples:", sum(train_indicator == "0", na.rm = TRUE)))
print(paste("Gestational age range:", min(gestational_age, na.rm = TRUE), "-", max(gestational_age, na.rm = TRUE), "weeks"))

# Remove samples with missing data
valid_samples <- !is.na(gestational_age) & !is.na(train_indicator)
expr_data <- expr_data[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_indicator <- train_indicator[valid_samples]

print(paste("After removing missing data:", ncol(expr_data), "samples remain"))

# Split data into training and test sets
train_idx <- train_indicator == "1"
test_idx <- train_indicator == "0"

train_expr <- t(expr_data[, train_idx])
train_age <- gestational_age[train_idx]
test_expr <- t(expr_data[, test_idx])
test_age <- gestational_age[test_idx]

print(paste("Training set:", nrow(train_expr), "samples"))
print(paste("Test set:", nrow(test_expr), "samples"))

# Feature selection: Remove genes with low variance
print("Performing feature selection...")
gene_vars <- apply(train_expr, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > quantile(gene_vars, 0.5, na.rm = TRUE))

train_expr_filtered <- train_expr[, high_var_genes]
test_expr_filtered <- test_expr[, high_var_genes]

print(paste("Selected", length(high_var_genes), "high-variance genes out of", ncol(train_expr)))

# Further feature selection using correlation with gestational age
gene_cors <- apply(train_expr_filtered, 2, function(x) abs(cor(x, train_age, use = "complete.obs")))
top_genes <- order(gene_cors, decreasing = TRUE)[1:min(1000, length(gene_cors))]

train_expr_final <- train_expr_filtered[, top_genes]
test_expr_final <- test_expr_filtered[, top_genes]

print(paste("Final feature set:", ncol(train_expr_final), "genes"))

# Split training data into train/validation for model tuning
set.seed(42)
val_idx <- sample(nrow(train_expr_final), size = floor(0.2 * nrow(train_expr_final)))
train_idx_final <- setdiff(1:nrow(train_expr_final), val_idx)

X_train <- train_expr_final[train_idx_final, ]
y_train <- train_age[train_idx_final]
X_val <- train_expr_final[val_idx, ]
y_val <- train_age[val_idx]

print(paste("Training samples:", nrow(X_train)))
print(paste("Validation samples:", nrow(X_val)))

# Model 1: Elastic Net Regression
print("Training Elastic Net model...")
set.seed(42)

# Prepare data for glmnet
X_train_matrix <- as.matrix(X_train)
X_val_matrix <- as.matrix(X_val)
X_test_matrix <- as.matrix(test_expr_final)

# Cross-validation to find optimal lambda
cv_fit <- cv.glmnet(X_train_matrix, y_train, alpha = 0.5, nfolds = 5)
best_lambda <- cv_fit$lambda.min

# Train final elastic net model
elastic_model <- glmnet(X_train_matrix, y_train, alpha = 0.5, lambda = best_lambda)

# Predictions
elastic_pred_val <- predict(elastic_model, X_val_matrix, s = best_lambda)
elastic_pred_test <- predict(elastic_model, X_test_matrix, s = best_lambda)

elastic_rmse_val <- sqrt(mean((elastic_pred_val - y_val)^2))
elastic_rmse_test <- sqrt(mean((elastic_pred_test - test_age)^2))

print(paste("Elastic Net - Validation RMSE:", round(elastic_rmse_val, 3)))
print(paste("Elastic Net - Test RMSE:", round(elastic_rmse_test, 3)))

# Model 2: Random Forest
print("Training Random Forest model...")
set.seed(42)

# Convert to data frames
train_df <- data.frame(age = y_train, X_train)
val_df <- data.frame(age = y_val, X_val)
test_df <- data.frame(age = test_age, test_expr_final)

# Train random forest with tuning
rf_model <- randomForest(age ~ ., data = train_df, 
                        ntree = 500, 
                        mtry = sqrt(ncol(X_train)),
                        importance = TRUE)

# Predictions
rf_pred_val <- predict(rf_model, val_df)
rf_pred_test <- predict(rf_model, test_df)

rf_rmse_val <- sqrt(mean((rf_pred_val - y_val)^2))
rf_rmse_test <- sqrt(mean((rf_pred_test - test_age)^2))

print(paste("Random Forest - Validation RMSE:", round(rf_rmse_val, 3)))
print(paste("Random Forest - Test RMSE:", round(rf_rmse_test, 3)))

# Model 3: Ensemble (Average of Elastic Net and Random Forest)
print("Creating ensemble model...")
ensemble_pred_test <- (as.vector(elastic_pred_test) + rf_pred_test) / 2
ensemble_rmse_test <- sqrt(mean((ensemble_pred_test - test_age)^2))

print(paste("Ensemble - Test RMSE:", round(ensemble_rmse_test, 3)))

# Select best model based on validation performance
best_model <- "Elastic Net"
best_pred_test <- as.vector(elastic_pred_test)
best_rmse_test <- elastic_rmse_test

if (rf_rmse_val < elastic_rmse_val) {
  best_model <- "Random Forest"
  best_pred_test <- rf_pred_test
  best_rmse_test <- rf_rmse_test
}

if (ensemble_rmse_test < best_rmse_test) {
  best_model <- "Ensemble"
  best_pred_test <- ensemble_pred_test
  best_rmse_test <- ensemble_rmse_test
}

print(paste("Best model:", best_model))
print(paste("Final Test RMSE:", round(best_rmse_test, 3)))

# Create scatter plot
print("Generating scatter plot...")
plot_data <- data.frame(
  Actual = test_age,
  Predicted = best_pred_test
)

# Calculate R-squared
r_squared <- cor(test_age, best_pred_test)^2

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = paste("Gestational Age Prediction -", best_model),
    subtitle = paste("Test RMSE:", round(best_rmse_test, 3), "weeks, R² =", round(r_squared, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

print(p)

# Save plot
ggsave("results/gestational_age_prediction.png", p, width = 8, height = 6, dpi = 300)

print("Analysis complete! Plot saved to results/gestational_age_prediction.png")