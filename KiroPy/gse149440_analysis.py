#!/usr/bin/env python3
"""
Gene Expression Analysis for Gestational Age Prediction
Dataset: GSE149440 from Gene Expression Omnibus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import GEOparse
import warnings
warnings.filterwarnings('ignore')

print("Starting GSE149440 analysis...")

# Download and parse the GEO dataset
print("Downloading GSE149440 dataset...")
try:
    gse = GEOparse.get_GEO(geo="GSE149440", destdir="./data")
    print("Dataset downloaded successfully!")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit(1)

# Extract expression data and metadata
print("Extracting expression data and metadata...")

# Get the first (and likely only) platform
platform_name = list(gse.gpls.keys())[0]
gpl = gse.gpls[platform_name]

# Get expression data
expression_data = gse.pivot_samples('VALUE')
print(f"Expression data shape: {expression_data.shape}")

# Extract metadata
metadata = []
for gsm_name, gsm in gse.gsms.items():
    sample_info = {
        'sample_id': gsm_name,
        'title': gsm.metadata.get('title', [''])[0],
        'source_name': gsm.metadata.get('source_name_ch1', [''])[0],
    }
    
    # Extract characteristics
    characteristics = gsm.metadata.get('characteristics_ch1', [])
    for char in characteristics:
        if ':' in char:
            key, value = char.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            sample_info[key] = value
    
    metadata.append(sample_info)

metadata_df = pd.DataFrame(metadata)
print(f"Metadata shape: {metadata_df.shape}")
print("Metadata columns:", metadata_df.columns.tolist())

# Display first few rows to understand the data structure
print("\nFirst few metadata rows:")
print(metadata_df.head())

# Check for gestational age and train columns
gestational_age_col = None
train_col = None

for col in metadata_df.columns:
    if 'gestational' in col.lower() and 'age' in col.lower():
        gestational_age_col = col
    if 'train' in col.lower():
        train_col = col

print(f"Gestational age column: {gestational_age_col}")
print(f"Train column: {train_col}")

if gestational_age_col is None or train_col is None:
    print("Required columns not found. Available columns:")
    for col in metadata_df.columns:
        print(f"  {col}: {metadata_df[col].unique()[:5]}")
    
    # Try to find the correct columns by examining unique values
    for col in metadata_df.columns:
        unique_vals = metadata_df[col].unique()
        if len(unique_vals) == 2 and set(map(str, unique_vals)) == {'0', '1'}:
            print(f"Potential train column: {col}")
            train_col = col
        
        # Check if column contains numeric values that could be gestational age
        try:
            numeric_vals = pd.to_numeric(metadata_df[col], errors='coerce')
            if not numeric_vals.isna().all() and numeric_vals.min() > 20 and numeric_vals.max() < 50:
                print(f"Potential gestational age column: {col}")
                gestational_age_col = col
        except:
            pass

# Clean and prepare the data
print(f"\nUsing gestational age column: {gestational_age_col}")
print(f"Using train column: {train_col}")

# Convert gestational age to numeric
metadata_df[gestational_age_col] = pd.to_numeric(metadata_df[gestational_age_col], errors='coerce')

# Convert train column to numeric
metadata_df[train_col] = pd.to_numeric(metadata_df[train_col], errors='coerce')

# Remove samples with missing values
valid_samples = metadata_df.dropna(subset=[gestational_age_col, train_col])
print(f"Valid samples after removing NaN: {len(valid_samples)}")

# Filter expression data to match valid samples
expression_data_filtered = expression_data.loc[:, valid_samples['sample_id']]
print(f"Filtered expression data shape: {expression_data_filtered.shape}")

# Prepare training and test sets
train_mask = valid_samples[train_col] == 1
test_mask = valid_samples[train_col] == 0

train_samples = valid_samples[train_mask]
test_samples = valid_samples[test_mask]

print(f"Training samples: {len(train_samples)}")
print(f"Test samples: {len(test_samples)}")

# Get expression data for train and test
X_train_full = expression_data_filtered.loc[:, train_samples['sample_id']].T
X_test = expression_data_filtered.loc[:, test_samples['sample_id']].T

y_train_full = train_samples[gestational_age_col].values
y_test = test_samples[gestational_age_col].values

print(f"X_train_full shape: {X_train_full.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train_full shape: {y_train_full.shape}")
print(f"y_test shape: {y_test.shape}")

# Remove genes with low variance or missing values
print("Preprocessing expression data...")
# Remove genes with too many missing values
missing_threshold = 0.1
valid_genes = (X_train_full.isna().sum() / len(X_train_full)) < missing_threshold
X_train_full = X_train_full.loc[:, valid_genes]
X_test = X_test.loc[:, valid_genes]

# Fill remaining missing values with median
X_train_full = X_train_full.fillna(X_train_full.median())
X_test = X_test.fillna(X_train_full.median())  # Use training median for test

print(f"After preprocessing - X_train shape: {X_train_full.shape}")

# Remove genes with zero variance
gene_std = X_train_full.std()
non_zero_var_genes = gene_std > 0
X_train_full = X_train_full.loc[:, non_zero_var_genes]
X_test = X_test.loc[:, non_zero_var_genes]

print(f"After removing zero variance genes - X_train shape: {X_train_full.shape}")

# Create validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"Final training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Feature selection and scaling
print("Performing feature selection...")

# Select top k features based on univariate statistical tests
k_features = min(5000, X_train.shape[1])  # Select top 5000 features or all if less
selector = SelectKBest(score_func=f_regression, k=k_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

print(f"Selected {k_features} features")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)
X_test_scaled = scaler.transform(X_test_selected)

print("Feature scaling completed")

# Model training and optimization
print("Training and optimizing models...")

models = {
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42, max_iter=2000),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1)
}

param_grids = {
    'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

best_models = {}
val_scores = {}

for name, model in models.items():
    print(f"Optimizing {name}...")
    
    grid_search = GridSearchCV(
        model, param_grids[name], 
        cv=5, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_models[name] = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_pred = grid_search.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_scores[name] = val_rmse
    
    print(f"{name} - Best params: {grid_search.best_params_}")
    print(f"{name} - Validation RMSE: {val_rmse:.4f}")

# Select best model based on validation performance
best_model_name = min(val_scores, key=val_scores.get)
best_model = best_models[best_model_name]

print(f"\nBest model: {best_model_name} (Validation RMSE: {val_scores[best_model_name]:.4f})")

# Retrain best model on full training set
print(f"Retraining {best_model_name} on full training set...")

# Use full training set for final model
X_train_full_selected = selector.transform(X_train_full)
X_train_full_scaled = scaler.fit_transform(X_train_full_selected)
X_test_final_scaled = scaler.transform(X_test_selected)

# Retrain the best model
final_model = best_models[best_model_name]
final_model.fit(X_train_full_scaled, y_train_full)

# Make predictions on test set
y_test_pred = final_model.predict(X_test_final_scaled)

# Calculate test RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n{'='*50}")
print(f"FINAL RESULTS")
print(f"{'='*50}")
print(f"Model: {best_model_name}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"{'='*50}")

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Gestational Age', fontsize=12)
plt.ylabel('Predicted Gestational Age', fontsize=12)
plt.title(f'Predicted vs Actual Gestational Age\n{best_model_name} Model (Test RMSE: {test_rmse:.4f})', fontsize=14)
plt.grid(True, alpha=0.3)

# Add correlation coefficient
correlation = np.corrcoef(y_test, y_test_pred)[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)

plt.tight_layout()
plt.savefig('gestational_age_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional statistics
print(f"\nAdditional Statistics:")
print(f"Correlation coefficient: {correlation:.4f}")
print(f"Mean absolute error: {np.mean(np.abs(y_test - y_test_pred)):.4f}")
print(f"R² score: {1 - (np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)):.4f}")

print(f"\nActual gestational age range: {y_test.min():.1f} - {y_test.max():.1f}")
print(f"Predicted gestational age range: {y_test_pred.min():.1f} - {y_test_pred.max():.1f}")

print("\nAnalysis completed successfully!")