# Load results
load("gse149440_model_results.RData")

# Compute RMSEs
rmse_enet <- sqrt(mean((pred_test_enet - y_test)^2))
rmse_rf <- sqrt(mean((pred_test_rf - y_test)^2))
cat(sprintf("Elastic Net Test set RMSE: %.3f\n", rmse_enet))
cat(sprintf("Random Forest Test set RMSE: %.3f\n", rmse_rf))

# Scatter plots
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
library(ggplot2)

df <- data.frame(actual = y_test, ENet = as.numeric(pred_test_enet), RF = as.numeric(pred_test_rf))

# Plot Elastic Net
p1 <- ggplot(df, aes(x = actual, y = ENet)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = sprintf("Elastic Net: Predicted vs Actual (Test Set)\nRMSE = %.2f", rmse_enet),
       x = "Actual Gestational Age", y = "Predicted Gestational Age") +
  theme_minimal()

# Plot Random Forest
p2 <- ggplot(df, aes(x = actual, y = RF)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = sprintf("Random Forest: Predicted vs Actual (Test Set)\nRMSE = %.2f", rmse_rf),
       x = "Actual Gestational Age", y = "Predicted Gestational Age") +
  theme_minimal()

print(p1)
print(p2)
