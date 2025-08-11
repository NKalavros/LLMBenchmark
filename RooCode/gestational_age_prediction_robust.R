# Gestational Age Prediction from Gene Expression Data - Robust Version
# Dataset: GSE149440 from Gene Expression Omnibus

cat("=== Starting Gestational Age Prediction Analysis ===\n")

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  for (pkg in packages) {
    cat("Checking package:", pkg, "\n")
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      if (pkg %in% c("GEOquery", "limma", "Biobase")) {
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

# Required packages
cat("Installing/checking required packages...\n")
required_packages <- c("GEOquery", "Biobase", "caret", "randomForest", "ggplot2")

tryCatch({
  install_if_missing(required_packages)
  cat("All packages installed successfully.\n")
}, error = function(e) {
  cat("Error installing packages:", e$message, "\n")
  stop("Package installation failed")
})

# Load libraries
cat("Loading libraries...\n")
tryCatch({
  library(GEOquery)
  library(Biobase)
  library(caret)
  library(randomForest)
  library(ggplot2)
  cat("Libraries loaded successfully.\n")
}, error = function(e) {
  cat("Error loading libraries:", e$message, "\n")
  stop("Library loading failed")
})

# Download GSE149440 dataset
cat("Downloading GSE149440 dataset from GEO...\n")
tryCatch({
  gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
  
  # Extract expression set - handle list structure properly
  if (is.list(gse)) {
    cat("GSE object is a list with", length(gse), "elements\n")
    eset <- gse[[1]]
  } else {
    eset <- gse
  }
  
  # Verify we have an ExpressionSet object
  if (!is(eset, "ExpressionSet")) {
    cat("Object class:", class(eset), "\n")
    stop("Downloaded object is not an ExpressionSet")
  }
  
  cat("Dataset downloaded successfully.\n")
  cat("Dataset dimensions:", dim(exprs(eset)), "\n")
  
}, error = function(e) {
  cat("Error downloading dataset:", e$message, "\n")
  stop("Dataset download failed")
})

# Extract metadata
cat("Extracting metadata...\n")
tryCatch({
  metadata <- pData(eset)
  cat("Metadata columns available:\n")
  print(colnames(metadata))
  
  # Check for required columns (with flexible column name matching)
  gestational_col <- grep("gestational.*age", colnames(metadata), ignore.case = TRUE, value = TRUE)
  train_col <- grep("train", colnames(metadata), ignore.case = TRUE, value = TRUE)
  
  if (length(gestational_col) == 0) {
    cat("Available columns:\n")
    print(colnames(metadata))
    stop("Gestational age column not found in metadata")
  }
  
  if (length(train_col) == 0) {
    cat("Available columns:\n")
    print(colnames(metadata))
    stop("Train/test split column not found in metadata")
  }
  
  cat("Using gestational age column:", gestational_col[1], "\n")
  cat("Using train/test column:", train_col[1], "\n")
  
  # Extract gestational age and training indicators
  gestational_age <- as.numeric(metadata[[gestational_col[1]]])
  train_indicator <- metadata[[train_col[1]]]
  
  cat("Gestational age range:", range(gestational_age, na.rm = TRUE), "\n")
  cat("Training samples:", sum(train_indicator == "1" | train_indicator == 1, na.rm = TRUE), "\n")
  cat("Test samples:", sum(train_indicator == "0" | train_indicator == 0, na.rm = TRUE), "\n")
  
}, error = function(e) {
  cat("Error extracting metadata:", e$message, "\n")
  stop("Metadata extraction failed")
})

# Clean data and split into training/test sets
cat("Cleaning data and splitting into train/test sets...\n")
tryCatch({
  # Remove samples with missing data
  valid_samples <- !is.na(gestational_age) & !is.na(train_indicator)
  eset_clean <- eset[, valid_samples]
  gestational_age_clean <- gestational_age[valid_samples]
  train_indicator_clean <- train_indicator[valid_samples]
  
  cat("Clean dataset dimensions:", dim(exprs(eset_clean)), "\n")
  
  # Split into training and test sets
  train_indices <- which(train_indicator_clean == "1" | train_indicator_clean == 1)
  test_indices <- which(train_indicator_clean == "0" | train_indicator_clean == 0)
  
  if (length(train_indices) == 0 || length(test_indices) == 0) {
    stop("No training or test samples found")
  }
  
  train_expr <- exprs(eset_clean)[, train_indices]
  test_expr <- exprs(eset_clean)[, test_indices]
  
  train_age <- gestational_age_clean[train_indices]
  test_age <- gestational_age_clean[test_indices]
  
  cat("Training set:", dim(train_expr), "\n")
  cat("Test set:", dim(test_expr), "\n")
  
}, error = function(e) {
  cat("Error in data cleaning:", e$message, "\n")
  stop("Data cleaning failed")
})

# Preprocess expression data
cat("Preprocessing expression data...\n")
tryCatch({
  # Check if data needs log transformation
  expr_range <- range(train_expr, na.rm = TRUE)
  cat("Expression data range:", expr_range, "\n")
  
  if (expr_range[2] > 100) {
    cat("Applying log2 transformation...\n")
    train_expr <- log2(train_expr + 1)
    test_expr <- log2(test_expr + 1)
  }
  
  # Remove features with low variance or missing values
  gene_vars <- apply(train_expr, 1, var, na.rm = TRUE)
  valid_genes <- !is.na(gene_vars) & gene_vars > 0.01
  
  train_expr_filtered <- train_expr[valid_genes, ]
  test_expr_filtered <- test_expr[valid_genes, ]
  
  cat("Genes after filtering:", nrow(train_expr_filtered), "\n")
  
  # Feature selection using correlation with gestational age
  correlations <- apply(train_expr_filtered, 1, function(x) {
    cor(x, train_age, use = "complete.obs")
  })
  
  # Select top correlated genes
  top_n_genes <- min(500, sum(!is.na(correlations)))
  top_gene_indices <- order(abs(correlations), decreasing = TRUE)[1:top_n_genes]
  
  train_expr_selected <- train_expr_filtered[top_gene_indices, ]
  test_expr_selected <- test_expr_filtered[top_gene_indices, ]
  
  cat("Selected genes:", nrow(train_expr_selected), "\n")
  
}, error = function(e) {
  cat("Error in preprocessing:", e$message, "\n")
  stop("Preprocessing failed")
})

# Prepare data for modeling
cat("Preparing data for modeling...\n")
tryCatch({
  # Transpose data for modeling (samples as rows, genes as columns)
  train_data <- t(train_expr_selected)
  test_data <- t(test_expr_selected)
  
  # Create data frames
  train_df <- data.frame(gestational_age = train_age, train_data)
  test_df <- data.frame(gestational_age = test_age, test_data)
  
  # Remove any remaining NA values
  train_df <- train_df[complete.cases(train_df), ]
  test_df <- test_df[complete.cases(test_df), ]
  
  cat("Final training samples:", nrow(train_df), "\n")
  cat("Final test samples:", nrow(test_df), "\n")
  
}, error = function(e) {
  cat("Error preparing data:", e$message, "\n")
  stop("Data preparation failed")
})

# Train prediction model
cat("Training prediction model...\n")
tryCatch({
  # Set up cross-validation
  ctrl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
  
  # Train Random Forest model
  cat("Training Random Forest model...\n")
  set.seed(42)
  rf_model <- train(gestational_age ~ ., 
                    data = train_df,
                    method = "rf",
                    trControl = ctrl,
                    tuneGrid = expand.grid(mtry = c(10, 20)),
                    ntree = 50)
  
  cat("Model training completed.\n")
  cat("Best mtry:", rf_model$bestTune$mtry, "\n")
  cat("Cross-validation RMSE:", min(rf_model$results$RMSE), "\n")
  
}, error = function(e) {
  cat("Error in model training:", e$message, "\n")
  stop("Model training failed")
})

# Make predictions and evaluate
cat("Making predictions and evaluating...\n")
tryCatch({
  # Make predictions on test set
  test_predictions <- predict(rf_model, newdata = test_df)
  
  # Calculate RMSE on test set
  test_rmse <- sqrt(mean((test_predictions - test_df$gestational_age)^2))
  
  # Calculate R-squared
  r_squared <- cor(test_predictions, test_df$gestational_age)^2
  
  cat("\n=== FINAL RESULTS ===\n")
  cat("Test Set RMSE:", round(test_rmse, 4), "\n")
  cat("Test Set R-squared:", round(r_squared, 4), "\n")
  cat("Number of test samples:", nrow(test_df), "\n")
  
}, error = function(e) {
  cat("Error in prediction/evaluation:", e$message, "\n")
  stop("Prediction/evaluation failed")
})

# Create scatter plot
cat("Creating scatter plot...\n")
tryCatch({
  plot_data <- data.frame(
    Actual = test_df$gestational_age,
    Predicted = test_predictions
  )
  
  p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, size = 2, color = "blue") +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
    labs(title = paste("Predicted vs Actual Gestational Age (Test Set)\nRMSE =", 
                       round(test_rmse, 3), ", R² =", round(r_squared, 3)),
         x = "Actual Gestational Age (weeks)",
         y = "Predicted Gestational Age (weeks)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12))
  
  # Save the plot
  ggsave("gestational_age_prediction_plot.png", plot = p, width = 8, height = 6, dpi = 300)
  
  cat("Scatter plot created and saved as 'gestational_age_prediction_plot.png'\n")
  
}, error = function(e) {
  cat("Error creating plot:", e$message, "\n")
  cat("Continuing without plot...\n")
})

cat("\n=== Analysis completed successfully! ===\n")
cat("Final Test RMSE:", round(test_rmse, 4), "\n")