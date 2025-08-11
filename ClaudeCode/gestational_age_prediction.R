# Gestational Age Prediction from Gene Expression Data
# Dataset: GSE149440

# Load required libraries
library(GEOquery)
library(Biobase)
library(randomForest)
library(glmnet)
library(caret)
library(ggplot2)
library(dplyr)

# Download GSE149440 dataset
cat("Downloading GSE149440 dataset...\n")
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
gse <- gse[[1]]

# Extract expression data and metadata
expr_data <- exprs(gse)
metadata <- pData(gse)

cat("Dataset dimensions:\n")
cat("Expression data:", dim(expr_data), "\n")
cat("Metadata:", dim(metadata), "\n")

# Examine metadata structure
cat("\nMetadata columns:\n")
print(colnames(metadata))

# Extract relevant variables
gestational_age <- as.numeric(metadata$`gestational age:ch1`)
train_flag <- metadata$`train:ch1`

cat("\nGestational age summary:\n")
print(summary(gestational_age))

cat("\nTrain/test split:\n")
print(table(train_flag))

# Remove samples with missing gestational age or train flag
valid_samples <- !is.na(gestational_age) & !is.na(train_flag)
expr_data <- expr_data[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_flag <- train_flag[valid_samples]

cat("\nAfter removing missing values:\n")
cat("Expression data:", dim(expr_data), "\n")
cat("Valid samples:", length(gestational_age), "\n")

# Split data into training and test sets
train_idx <- train_flag == "1"
test_idx <- train_flag == "0"

X_train <- t(expr_data[, train_idx])
y_train <- gestational_age[train_idx]
X_test <- t(expr_data[, test_idx])
y_test <- gestational_age[test_idx]

cat("\nTraining set:", nrow(X_train), "samples,", ncol(X_train), "genes\n")
cat("Test set:", nrow(X_test), "samples,", ncol(X_test), "genes\n")

# Remove genes with zero variance
gene_vars <- apply(X_train, 2, var, na.rm = TRUE)
high_var_genes <- which(gene_vars > 0 & !is.na(gene_vars))

X_train <- X_train[, high_var_genes]
X_test <- X_test[, high_var_genes]

cat("After removing zero variance genes:", ncol(X_train), "genes remaining\n")

# Feature selection using correlation with gestational age
cat("\nPerforming feature selection...\n")
gene_cors <- abs(cor(X_train, y_train, use = "complete.obs"))
top_genes_idx <- order(gene_cors, decreasing = TRUE)[1:min(1000, length(gene_cors))]

X_train_selected <- X_train[, top_genes_idx]
X_test_selected <- X_test[, top_genes_idx]

cat("Selected top", ncol(X_train_selected), "genes based on correlation with gestational age\n")

# Scale the data
X_train_scaled <- scale(X_train_selected)
X_test_scaled <- scale(X_test_selected, 
                       center = attr(X_train_scaled, "scaled:center"),
                       scale = attr(X_train_scaled, "scaled:scale"))

# Model 1: Elastic Net Regression
cat("\nTraining Elastic Net model...\n")
set.seed(42)

# Use cross-validation to find optimal lambda
cv_model <- cv.glmnet(X_train_scaled, y_train, alpha = 0.5, nfolds = 5)
best_lambda <- cv_model$lambda.min

# Train final model
elastic_model <- glmnet(X_train_scaled, y_train, alpha = 0.5, lambda = best_lambda)

# Predict on test set
pred_elastic <- predict(elastic_model, X_test_scaled)
rmse_elastic <- sqrt(mean((y_test - pred_elastic)^2))

cat("Elastic Net RMSE:", rmse_elastic, "\n")

# Model 2: Random Forest
cat("\nTraining Random Forest model...\n")
set.seed(42)

# Use a subset of genes for Random Forest due to computational constraints
rf_genes <- min(500, ncol(X_train_scaled))
rf_train_data <- data.frame(
  gestational_age = y_train,
  X_train_scaled[, 1:rf_genes]
)

rf_model <- randomForest(gestational_age ~ ., data = rf_train_data, 
                        ntree = 500, mtry = sqrt(rf_genes), importance = TRUE)

# Predict on test set
rf_test_data <- data.frame(X_test_scaled[, 1:rf_genes])
pred_rf <- predict(rf_model, rf_test_data)
rmse_rf <- sqrt(mean((y_test - pred_rf)^2))

cat("Random Forest RMSE:", rmse_rf, "\n")

# Model 3: Support Vector Regression with RBF kernel
cat("\nTraining SVR model...\n")
library(e1071)
set.seed(42)

# Use subset of genes for SVR
svr_genes <- min(200, ncol(X_train_scaled))
svr_model <- svm(X_train_scaled[, 1:svr_genes], y_train, 
                kernel = "radial", cost = 1, gamma = 1/svr_genes)

pred_svr <- predict(svr_model, X_test_scaled[, 1:svr_genes])
rmse_svr <- sqrt(mean((y_test - pred_svr)^2))

cat("SVR RMSE:", rmse_svr, "\n")

# Choose best model
models <- list(
  "Elastic Net" = list(pred = pred_elastic, rmse = rmse_elastic),
  "Random Forest" = list(pred = pred_rf, rmse = rmse_rf),
  "SVR" = list(pred = pred_svr, rmse = rmse_svr)
)

best_model_name <- names(models)[which.min(sapply(models, function(x) x$rmse))]
best_predictions <- models[[best_model_name]]$pred
best_rmse <- models[[best_model_name]]$rmse

cat("\nBest model:", best_model_name, "\n")
cat("Best RMSE:", best_rmse, "\n")

# Create scatter plot
plot_data <- data.frame(
  Actual = y_test,
  Predicted = as.numeric(best_predictions)
)

p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(
    title = paste("Predicted vs Actual Gestational Age (", best_model_name, ")", sep = ""),
    subtitle = paste("RMSE =", round(best_rmse, 3)),
    x = "Actual Gestational Age (weeks)",
    y = "Predicted Gestational Age (weeks)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  coord_fixed()

print(p)

# Save plot
ggsave("gestational_age_prediction_plot.png", plot = p, width = 8, height = 6, dpi = 300)

# Calculate additional metrics
mae <- mean(abs(y_test - best_predictions))
r2 <- cor(y_test, best_predictions)^2

cat("\nFinal Results:\n")
cat("==============\n")
cat("Best Model:", best_model_name, "\n")
cat("RMSE:", round(best_rmse, 3), "\n")
cat("MAE:", round(mae, 3), "\n")
cat("R²:", round(r2, 3), "\n")
cat("Plot saved as: gestational_age_prediction_plot.png\n")