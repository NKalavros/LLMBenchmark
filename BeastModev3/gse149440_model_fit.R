# Step 5: Fit Elastic Net regression model with cross-validation

# Load preprocessed data
load("gse149440_preprocessed.RData")



# Filter to top 10000 most variable genes
gene_vars <- apply(expr_z, 1, var)
top_n <- 10000
top_genes <- order(gene_vars, decreasing = TRUE)[1:min(top_n, length(gene_vars))]
expr_z_top <- expr_z[top_genes, ]

# Split into train and test
X_train <- t(expr_z_top[, train_set == 1])
y_train <- gest_age[train_set == 1]
X_test <- t(expr_z_top[, train_set == 0])
y_test <- gest_age[train_set == 0]


# Load glmnet, set CRAN mirror if needed
if (!requireNamespace("glmnet", quietly = TRUE)) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  install.packages("glmnet")
}
library(glmnet)

# Fit Elastic Net with cross-validation (alpha search)
best_rmse <- Inf
best_alpha <- NA
best_model <- NULL

# Finer alpha grid search
alphas <- seq(0, 1, by = 0.1)
for (a in alphas) {
  cvfit <- cv.glmnet(X_train, y_train, alpha = a, nfolds = 5, standardize = FALSE)
  if (min(cvfit$cvm) < best_rmse) {
    best_rmse <- min(cvfit$cvm)
    best_alpha <- a
    best_model <- cvfit
  }
}

cat(sprintf("Best alpha: %.2f, CV RMSE: %.3f\n", best_alpha, best_rmse))


# Predict on test set (Elastic Net)
pred_test_enet <- predict(best_model, newx = X_test, s = "lambda.min")

# Try Random Forest regression
if (!requireNamespace("randomForest", quietly = TRUE)) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  install.packages("randomForest")
}
library(randomForest)
set.seed(42)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
pred_test_rf <- predict(rf_model, newdata = X_test)

# Ensure y_test is in the global environment
y_test <- as.numeric(y_test)

# Save results (include y_test)
save(best_model, pred_test_enet, rf_model, pred_test_rf, y_test, X_test, file = "gse149440_model_results.RData")
cat("Elastic Net and Random Forest models fit and predictions saved as gse149440_model_results.RData\n")
