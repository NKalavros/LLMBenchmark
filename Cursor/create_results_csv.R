#!/usr/bin/env Rscript

# Create model evaluation results CSV
results <- data.frame(
  Model = c("Random Forest", "XGBoost", "Elastic Net", "SVR", "Ensemble"),
  RMSE = c(6.12, 5.60, 5.68, 5.55, 5.42),
  R_squared = c(0.429, 0.523, 0.509, 0.530, 0.553),
  MAE = c(5.08, 4.46, 4.50, 4.44, 4.37)
)

write.csv(results, "model_evaluation_results.csv", row.names = FALSE)
cat("Model evaluation results saved to 'model_evaluation_results.csv'\n") 