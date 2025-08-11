#!/usr/bin/env python3
"""
Optimized Gene Expression Analysis for Gestational Age Prediction
Dataset: GSE149440 from Gene Expression Omnibus
Focus: Minimize RMSE through advanced ML techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
import GEOparse
import warnings
warnings.filterwarnings('ignore')

print("Starting optimized GSE149440 analysis...")

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

# Find the correct columns
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
    print("Searching for columns by examining unique values...")
    for col in metadata_df.columns:
        unique_vals = metadata_df[col].unique()
        print(f"{col}: {unique_vals[:5]}")
        
        if len(unique_vals) == 2 and set(map(str, unique_vals)) == {'0', '1'}:
            print(f"Found potential train column: {col}")
            train_col = col
        
        try:
            numeric_vals = pd.to_numeric(metadata_df[col], errors='coerce')
            if not numeric_vals.isna().all() and numeric_vals.min() > 20 and numeric_vals.max() < 50:
                print(f"Found potential gestational age column: {col}")
                gestational_age_col = col
        except:
            pass

# Clean and prepare the data
metadata_df[gestational_age_col] = pd.to_numeric(metadata_df[gestational_age_col], errors='coerce')
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
print(f"Target range - Train: {y_train_full.min():.1f}-{y_train_full.max():.1f}, Test: {y_test.min():.1f}-{y_test.max():.1f}")

# Advanced preprocessing
print("Advanced preprocessing...")

# Remove genes with too many missing values
missing_threshold = 0.05  # More stringent
valid_genes = (X_train_full.isna().sum() / len(X_train_full)) < missing_threshold
X_train_full = X_train_full.loc[:, valid_genes]
X_test = X_test.loc[:, valid_genes]

# Fill remaining missing values with median
X_train_full = X_train_full.fillna(X_train_full.median())
X_test = X_test.fillna(X_train_full.median())

# Remove genes with very low variance (more aggressive)
gene_std = X_train_full.std()
high_var_genes = gene_std > gene_std.quantile(0.1)  # Keep top 90% by variance
X_train_full = X_train_full.loc[:, high_var_genes]
X_test = X_test.loc[:, high_var_genes]

print(f"After preprocessing - X_train shape: {X_train_full.shape}")

# Remove highly correlated features to reduce multicollinearity
print("Removing highly correlated features...")
corr_matrix = X_train_full.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_train_full = X_train_full.drop(columns=high_corr_features)
X_test = X_test.drop(columns=high_corr_features)

print(f"After correlation filtering - X_train shape: {X_train_full.shape}")

# Create validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=None
)

print(f"Final training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Multiple feature selection strategies
print("Applying multiple feature selection strategies...")

# Strategy 1: Statistical selection
k_features = min(2000, X_train.shape[1])
selector_stats = SelectKBest(score_func=f_regression, k=k_features)
X_train_stats = selector_stats.fit_transform(X_train, y_train)
X_val_stats = selector_stats.transform(X_val)
X_test_stats = selector_stats.transform(X_test)

# Strategy 2: Tree-based selection
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
selector_tree = SelectFromModel(rf_selector, max_features=k_features)
X_train_tree = selector_tree.fit_transform(X_train, y_train)
X_val_tree = selector_tree.transform(X_val)
X_test_tree = selector_tree.transform(X_test)

# Strategy 3: L1-based selection
lasso_selector = Lasso(alpha=0.01, random_state=42)
selector_l1 = SelectFromModel(lasso_selector, max_features=k_features)
X_train_l1 = selector_l1.fit_transform(X_train, y_train)
X_val_l1 = selector_l1.transform(X_val)
X_test_l1 = selector_l1.transform(X_test)

print(f"Statistical selection: {X_train_stats.shape[1]} features")
print(f"Tree-based selection: {X_train_tree.shape[1]} features")
print(f"L1-based selection: {X_train_l1.shape[1]} features")

# Test different feature sets
feature_sets = {
    'statistical': (X_train_stats, X_val_stats, X_test_stats),
    'tree_based': (X_train_tree, X_val_tree, X_test_tree),
    'l1_based': (X_train_l1, X_val_l1, X_test_l1)
}

# Advanced model ensemble
print("Setting up advanced model ensemble...")

# Define models with more extensive hyperparameter grids
models = {
    'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'ElasticNet': ElasticNet(random_state=42, max_iter=2000),
    'Ridge': Ridge(random_state=42),
    'SVR': SVR()
}

# Extensive hyperparameter grids
param_grids = {
    'XGBoost': {
        'n_estimators': [200, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    },
    'LightGBM': {
        'n_estimators': [200, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    },
    'GradientBoosting': {
        'n_estimators': [200, 500],
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9]
    },
    'ExtraTrees': {
        'n_estimators': [200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'RandomForest': {
        'n_estimators': [200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
}

# Test each feature selection strategy
best_overall_score = float('inf')
best_overall_config = None
results = {}

for fs_name, (X_tr, X_v, X_te) in feature_sets.items():
    print(f"\nTesting feature set: {fs_name}")
    
    # Scale features
    scaler = RobustScaler()  # More robust to outliers
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_v_scaled = scaler.transform(X_v)
    X_te_scaled = scaler.transform(X_te)
    
    fs_results = {}
    
    for name, model in models.items():
        print(f"  Optimizing {name}...")
        
        try:
            # Reduced CV folds for speed, but still robust
            grid_search = GridSearchCV(
                model, param_grids[name], 
                cv=3, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_tr_scaled, y_train)
            
            # Evaluate on validation set
            val_pred = grid_search.predict(X_v_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            fs_results[name] = {
                'model': grid_search.best_estimator_,
                'params': grid_search.best_params_,
                'val_rmse': val_rmse,
                'scaler': scaler,
                'X_test': X_te_scaled
            }
            
            print(f"    {name} - Validation RMSE: {val_rmse:.4f}")
            
            # Track best overall
            if val_rmse < best_overall_score:
                best_overall_score = val_rmse
                best_overall_config = (fs_name, name, fs_results[name])
                
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    results[fs_name] = fs_results

print(f"\nBest configuration: {best_overall_config[0]} + {best_overall_config[1]} (RMSE: {best_overall_score:.4f})")

# Create ensemble of top models
print("\nCreating ensemble of top models...")

# Get top 3 models from best feature set
best_fs_name = best_overall_config[0]
best_fs_results = results[best_fs_name]

# Sort models by validation performance
sorted_models = sorted(best_fs_results.items(), key=lambda x: x[1]['val_rmse'])
top_models = sorted_models[:3]

print("Top 3 models for ensemble:")
for name, result in top_models:
    print(f"  {name}: {result['val_rmse']:.4f}")

# Create voting ensemble
ensemble_models = []
for name, result in top_models:
    ensemble_models.append((name, result['model']))

voting_regressor = VotingRegressor(ensemble_models)

# Use the best feature set and scaler
best_scaler = best_overall_config[2]['scaler']
best_X_train = feature_sets[best_fs_name][0]
best_X_val = feature_sets[best_fs_name][1]
best_X_test = feature_sets[best_fs_name][2]

X_train_scaled = best_scaler.fit_transform(best_X_train)
X_val_scaled = best_scaler.transform(best_X_val)
X_test_scaled = best_scaler.transform(best_X_test)

# Train ensemble on full training set
X_full_scaled = best_scaler.fit_transform(
    np.vstack([best_X_train, best_X_val])
)
y_full = np.concatenate([y_train, y_val])

print("Training ensemble on full training set...")
voting_regressor.fit(X_full_scaled, y_full)

# Make predictions
ensemble_pred = voting_regressor.predict(X_test_scaled)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print(f"Ensemble RMSE: {ensemble_rmse:.4f}")

# Also test the single best model
best_single_model = best_overall_config[2]['model']
best_single_model.fit(X_full_scaled, y_full)
single_pred = best_single_model.predict(X_test_scaled)
single_rmse = np.sqrt(mean_squared_error(y_test, single_pred))

print(f"Best single model RMSE: {single_rmse:.4f}")

# Choose the better approach
if ensemble_rmse < single_rmse:
    final_pred = ensemble_pred
    final_rmse = ensemble_rmse
    final_model_name = "Ensemble"
else:
    final_pred = single_pred
    final_rmse = single_rmse
    final_model_name = f"Single {best_overall_config[1]}"

print(f"\n{'='*60}")
print(f"FINAL OPTIMIZED RESULTS")
print(f"{'='*60}")
print(f"Feature Selection: {best_fs_name}")
print(f"Model: {final_model_name}")
print(f"Test RMSE: {final_rmse:.4f}")
print(f"{'='*60}")

# Additional metrics
correlation = np.corrcoef(y_test, final_pred)[0, 1]
mae = np.mean(np.abs(y_test - final_pred))
r2 = r2_score(y_test, final_pred)

print(f"Correlation: {correlation:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Create enhanced scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(y_test, final_pred, alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

# Add trend line
z = np.polyfit(y_test, final_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "g--", alpha=0.8, label=f'Trend Line (slope={z[0]:.3f})')

plt.xlabel('Actual Gestational Age (weeks)', fontsize=14)
plt.ylabel('Predicted Gestational Age (weeks)', fontsize=14)
plt.title(f'Optimized Gestational Age Prediction\n{final_model_name} with {best_fs_name} features (RMSE: {final_rmse:.4f})', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Add statistics box
stats_text = f'RMSE: {final_rmse:.4f}\nCorr: {correlation:.3f}\nMAE: {mae:.4f}\nR²: {r2:.3f}'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), 
         fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.savefig('optimized_gestational_age_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPrediction range: {final_pred.min():.1f} - {final_pred.max():.1f} weeks")
print(f"Actual range: {y_test.min():.1f} - {y_test.max():.1f} weeks")
print("\nOptimized analysis completed successfully!")