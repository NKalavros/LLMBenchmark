# Step 3: Identify gestational age and train/test split

# Load expression and metadata
load("gse149440_expr_meta.RData")

# Check available columns in metadata
cat("Metadata columns:\n")
print(colnames(meta))

# Extract gestational age and train/test assignment
# (Assuming columns are named 'gestational age:ch1' and 'train:ch1')
gest_age <- as.numeric(meta$`gestational age:ch1`)
train_set <- as.numeric(meta$`train:ch1`)

# Check for NAs or issues
cat("Gestational age summary:\n")
print(summary(gest_age))
cat("Train/test split counts:\n")
print(table(train_set))

# Save for next steps
save(expr, meta, gest_age, train_set, file = "gse149440_model_data.RData")
cat("Gestational age and train/test split extracted and saved as gse149440_model_data.RData\n")
