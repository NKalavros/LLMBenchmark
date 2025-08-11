#!/usr/bin/env python3
"""
GSE149440 Gestational Age Prediction Analysis

This script downloads the GSE149440 dataset from Gene Expression Omnibus (GEO)
and builds a prediction model for gestational age using gene expression data.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check and install required packages
def install_package(package):
    """Install package if not already installed"""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        os.system(f"{sys.executable} -m pip install {package}")

# Install required packages
required_packages = [
    'GEOparse',
    'pandas',
    'numpy', 
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'scipy'
]

print("Checking and installing required packages...")
for package in required_packages:
    install_package(package)

import GEOparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("All packages imported successfully!")

def download_and_parse_dataset():
    """Download and parse GSE149440 dataset from GEO"""
    print("Downloading GSE149440 dataset from GEO...")
    try:
        # Download the dataset
        gse = GEOparse.get_GEO(geo="GSE149440", destdir="./data")
        print("Dataset downloaded successfully!")
        
        # Get the expression data and metadata
        if hasattr(gse, 'gpls') and gse.gpls:
            gpl = list(gse.gpls.keys())[0]  # Get platform info
            print(f"Platform: {gpl}")
        
        if hasattr(gse, 'gsms') and gse.gsms:
            gsm_list = list(gse.gsms.keys())  # Get sample IDs
            print(f"Found {len(gsm_list)} samples")
        else:
            print("No GSM samples found in GSE object")
        
        return gse
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def extract_expression_and_metadata(gse):
    """Extract expression data and metadata from GSE object"""
    print("Extracting expression data and metadata...")
    
    # Get sample metadata
    metadata_list = []
    expression_data = []
    sample_ids = []
    
    for gsm_name, gsm in gse.gsms.items():
        # Extract metadata
        metadata = {}
        metadata['sample_id'] = gsm_name
        
        # Parse characteristics
        for char in gsm.metadata.get('characteristics_ch1', []):
            if ':' in char:
                key, value = char.split(':', 1)
                metadata[key.strip()] = value.strip()
        
        metadata_list.append(metadata)
        
        # Extract expression data
        expr_table = gsm.table
        expression_data.append(expr_table['VALUE'].values)
        sample_ids.append(gsm_name)
    
    # Convert to DataFrames
    metadata_df = pd.DataFrame(metadata_list)
    
    # Get gene IDs from first sample
    first_sample = list(gse.gsms.values())[0]
    gene_ids = first_sample.table['ID_REF'].values
    
    # Create expression matrix
    expression_matrix = np.array(expression_data).T  # Genes x Samples
    expression_df = pd.DataFrame(expression_matrix, index=gene_ids, columns=sample_ids)
    
    print(f"Expression matrix shape: {expression_df.shape}")
    print(f"Metadata shape: {metadata_df.shape}")
    
    return expression_df, metadata_df

def prepare_data_for_modeling(expression_df, metadata_df):
    """Prepare data for machine learning modeling"""
    print("Preparing data for modeling...")
    
    # Print available columns in metadata
    print("Available metadata columns:")
    print(metadata_df.columns.tolist())
    print("\nFirst few rows of metadata:")
    print(metadata_df.head())
    
    # Find gestational age and train columns
    gestational_age_col = None
    train_col = None
    
    for col in metadata_df.columns:
        if 'gestational age' in col.lower() or 'gestational_age' in col.lower():
            gestational_age_col = col
        if 'train' in col.lower() and ('ch1' in col.lower() or col.lower() == 'train'):
            train_col = col
    
    if gestational_age_col is None:
        print("Available columns that might contain gestational age:")
        for col in metadata_df.columns:
            if any(keyword in col.lower() for keyword in ['age', 'gestational', 'ga']):
                print(f"- {col}: {metadata_df[col].iloc[:5].tolist()}")
        
        # Try to find it manually
        possible_cols = [col for col in metadata_df.columns if 'age' in col.lower()]
        if possible_cols:
            gestational_age_col = possible_cols[0]
            print(f"Using {gestational_age_col} as gestational age column")
    
    if train_col is None:
        print("Available columns that might contain train info:")
        for col in metadata_df.columns:
            if 'train' in col.lower():
                print(f"- {col}: {metadata_df[col].iloc[:5].tolist()}")
        
        possible_cols = [col for col in metadata_df.columns if 'train' in col.lower()]
        if possible_cols:
            train_col = possible_cols[0]
            print(f"Using {train_col} as train column")
    
    if gestational_age_col is None or train_col is None:
        raise ValueError("Could not find gestational age or train columns in metadata")
    
    # Clean and convert data
    metadata_df[gestational_age_col] = pd.to_numeric(metadata_df[gestational_age_col], errors='coerce')
    metadata_df[train_col] = metadata_df[train_col].astype(str)
    
    # Remove samples with missing gestational age
    valid_samples = metadata_df[gestational_age_col].notna()
    metadata_df = metadata_df[valid_samples].reset_index(drop=True)
    expression_df = expression_df[metadata_df['sample_id']]
    
    # Split train/test based on metadata
    train_mask = metadata_df[train_col] == '1'
    test_mask = metadata_df[train_col] == '0'
    
    print(f"Training samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    
    # Prepare features (transpose so samples are rows)
    X = expression_df.T  # Samples x Genes
    y = metadata_df[gestational_age_col].values
    
    # Split data using integer indexing to avoid alignment issues
    train_indices = metadata_df[train_mask].index
    test_indices = metadata_df[test_mask].index
    
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} genes")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} genes")
    
    return X_train, y_train, X_test, y_test

def preprocess_features(X_train, X_test, y_train, n_features=5000):
    """Preprocess features: handle missing values, scale, and select top features"""
    print("Preprocessing features...")
    
    # Handle missing values (fill with median)
    X_train_clean = X_train.fillna(X_train.median())
    X_test_clean = X_test.fillna(X_train.median())  # Use training median for test
    
    # Remove features with too many missing values or zero variance
    feature_variance = X_train_clean.var()
    valid_features = feature_variance > 0
    
    X_train_clean = X_train_clean.loc[:, valid_features]
    X_test_clean = X_test_clean.loc[:, valid_features]
    
    print(f"After removing zero-variance features: {X_train_clean.shape[1]} features")
    
    # Feature selection - select top k features based on correlation with target
    if X_train_clean.shape[1] > n_features:
        print(f"Selecting top {n_features} features...")
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_train_selected = selector.fit_transform(X_train_clean, y_train)
        X_test_selected = selector.transform(X_test_clean)
        
        # Don't convert back to DataFrame - keep as numpy arrays after feature selection
        print(f"Selected {X_train_selected.shape[1]} features")
        X_train_clean = X_train_selected
        X_test_clean = X_test_selected
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    print(f"Final feature matrix: {X_train_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_and_optimize_models(X_train, y_train):
    """Train and optimize different regression models"""
    print("Training and optimizing models...")
    
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    param_grids = {
        'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'RandomForest': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
        'GradientBoosting': {'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5]}
    }
    
    best_model = None
    best_score = float('inf')
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"Optimizing {name}...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=1  # Keep single-threaded to avoid timeout
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best score (convert back to positive RMSE)
        best_cv_score = np.sqrt(-grid_search.best_score_)
        results[name] = {
            'model': grid_search.best_estimator_,
            'cv_rmse': best_cv_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"{name} - CV RMSE: {best_cv_score:.4f}, Best params: {grid_search.best_params_}")
        
        if best_cv_score < best_score:
            best_score = best_cv_score
            best_model = grid_search.best_estimator_
            best_name = name
    
    print(f"\nBest model: {best_name} with CV RMSE: {best_score:.4f}")
    
    return best_model, best_name, results

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate correlation
    correlation, p_value = pearsonr(y_test, y_pred)
    
    print(f"Test Set Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Correlation: {correlation:.4f} (p-value: {p_value:.2e})")
    
    return y_pred, rmse, correlation

def create_scatter_plot(y_test, y_pred, rmse, correlation):
    """Create scatter plot of predicted vs actual values"""
    print("Creating scatter plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=50)
    
    # Add diagonal line (perfect prediction)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('Actual Gestational Age')
    plt.ylabel('Predicted Gestational Age')
    plt.title(f'Predicted vs Actual Gestational Age\nRMSE: {rmse:.4f}, Correlation: {correlation:.4f}')
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('gestational_age_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Scatter plot saved as 'gestational_age_predictions.png'")

def main():
    """Main analysis pipeline"""
    print("=== GSE149440 Gestational Age Prediction Analysis ===\n")
    
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Step 1: Download and parse dataset
    gse = download_and_parse_dataset()
    if gse is None:
        print("Failed to download dataset. Exiting.")
        return
    
    # Step 2: Extract expression data and metadata
    expression_df, metadata_df = extract_expression_and_metadata(gse)
    
    # Step 3: Prepare data for modeling
    X_train, y_train, X_test, y_test = prepare_data_for_modeling(expression_df, metadata_df)
    
    # Step 4: Preprocess features
    X_train_processed, X_test_processed, scaler = preprocess_features(X_train, X_test, y_train)
    
    # Step 5: Train and optimize models
    best_model, best_name, results = train_and_optimize_models(X_train_processed, y_train)
    
    # Step 6: Evaluate on test set
    y_pred, rmse, correlation = evaluate_model(best_model, X_test_processed, y_test)
    
    # Step 7: Create scatter plot
    create_scatter_plot(y_test, y_pred, rmse, correlation)
    
    print(f"\n=== Final Results ===")
    print(f"Best Model: {best_name}")
    print(f"Test Set RMSE: {rmse:.4f}")
    print(f"Test Set Correlation: {correlation:.4f}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()