# Step 4: Preprocess data (normalization, filtering)

# Load data
load("gse149440_model_data.RData")

# Filter to samples with non-missing gestational age and train/test assignment
valid <- !is.na(gest_age) & !is.na(train_set)
expr <- expr[, valid]
gest_age <- gest_age[valid]
train_set <- train_set[valid]
meta <- meta[valid, ]

# Remove genes with near-zero variance (unexpressed or uninformative)
var_genes <- apply(expr, 1, var)
keep_genes <- var_genes > 0.01  # threshold can be tuned
expr <- expr[keep_genes, ]

# Z-score normalization (genes x samples)
expr_z <- t(scale(t(expr)))

# Save for modeling
save(expr_z, gest_age, train_set, meta, file = "gse149440_preprocessed.RData")
cat("Preprocessing complete. Data saved as gse149440_preprocessed.RData\n")
