#!/usr/bin/env python3
"""
Gene Expression Omnibus GSE149440 Dataset Analysis
Predicting Gestational Age from Gene Expression Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import GEOparse
import warnings
warnings.filterwarnings('ignore')

def download_and_process_data():
    """Download GSE149440 dataset and process it"""
    print("Downloading GSE149440 dataset...")
    
    # Download the dataset
    gse = GEOparse.get_GEO(geo="GSE149440", destdir="./")
    
    print(f"Dataset downloaded. Found {len(gse.gsms)} samples.")
    
    # Extract sample metadata
    sample_metadata = []
    for gsm_name, gsm in gse.gsms.items():
        metadata = {
            'sample_id': gsm_name,
            'title': gsm.metadata.get('title', [''])[0],
        }
        
        # Extract characteristics
        characteristics = gsm.metadata.get('characteristics_ch1', [])
        for char in characteristics:
            if ':' in char:
                key, value = char.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                metadata[key] = value
        
        sample_metadata.append(metadata)
    
    metadata_df = pd.DataFrame(sample_metadata)
    print(f"Metadata columns: {metadata_df.columns.tolist()}")
    print(f"Sample metadata shape: {metadata_df.shape}")
    
    # Get expression data
    print("Processing expression data...")
    
    # Extract expression matrix
    expression_data = {}
    for gsm_name, gsm in gse.gsms.items():
        expression_data[gsm_name] = gsm.table['VALUE'].values
    
    # Get gene identifiers
    gene_ids = list(gse.gsms.values())[0].table['ID_REF'].values
    
    # Create expression dataframe
    expression_df = pd.DataFrame(expression_data, index=gene_ids).T
    print(f"Expression data shape: {expression_df.shape}")
    
    return metadata_df, expression_df

def prepare_data(metadata_df, expression_df):
    """Prepare data for modeling"""
    print("Preparing data for modeling...")
    
    # Use the correct column names
    gestational_age_col = 'gestational age'
    train_col = 'train'
    
    print(f"Gestational age column: {gestational_age_col}")
    print(f"Train/test column: {train_col}")
    
    # Align metadata and expression data
    metadata_samples = set(metadata_df['sample_id'])
    expression_samples = set(expression_df.index)
    common_samples = metadata_samples.intersection(expression_samples)
    print(f"Common samples between metadata and expression: {len(common_samples)}")
    
    # Filter data to common samples
    common_samples_list = list(common_samples)
    metadata_filtered = metadata_df[metadata_df['sample_id'].isin(common_samples_list)].copy()
    expression_filtered = expression_df.loc[common_samples_list].copy()
    
    # Convert gestational age to numeric
    metadata_filtered['gestational_age_numeric'] = pd.to_numeric(metadata_filtered[gestational_age_col], errors='coerce')
    
    # Remove samples with missing gestational age
    valid_samples = metadata_filtered.dropna(subset=['gestational_age_numeric'])
    print(f"Samples with valid gestational age: {len(valid_samples)}")
    
    # Filter expression data to valid samples
    expression_final = expression_filtered.loc[valid_samples['sample_id']].copy()
    
    # Convert expression data to numeric and handle missing values
    print("Converting expression data to numeric...")
    expression_final = expression_final.apply(pd.to_numeric, errors='coerce')
    
    # Remove genes with too many missing values (>50% missing)
    missing_threshold = 0.5
    genes_to_keep = expression_final.isnull().mean() < missing_threshold
    expression_final = expression_final.loc[:, genes_to_keep]
    print(f"Genes after filtering missing values: {expression_final.shape[1]}")
    
    # Fill remaining missing values with median
    expression_final = expression_final.fillna(expression_final.median())
    
    # Remove genes with zero variance
    gene_variance = expression_final.var()
    genes_with_variance = gene_variance > 0
    expression_final = expression_final.loc[:, genes_with_variance]
    print(f"Genes after removing zero variance: {expression_final.shape[1]}")
    
    return valid_samples, expression_final

def train_and_evaluate_models(metadata_df, expression_df):
    """Train and evaluate prediction models"""
    print("\n" + "="*50)
    print("TRAINING AND EVALUATING MODELS")
    print("="*50)
    
    # Split data based on train column
    train_mask = metadata_df['train'] == '1'
    test_mask = metadata_df['train'] == '0'
    
    train_metadata = metadata_df[train_mask].copy()
    test_metadata = metadata_df[test_mask].copy()
    
    train_expression = expression_df.loc[train_metadata['sample_id']].copy()
    test_expression = expression_df.loc[test_metadata['sample_id']].copy()
    
    y_train = train_metadata['gestational_age_numeric'].values
    y_test = test_metadata['gestational_age_numeric'].values
    
    print(f"Training samples: {len(train_expression)}")
    print(f"Test samples: {len(test_expression)}")
    print(f"Training gestational age range: {y_train.min():.1f} - {y_train.max():.1f}")
    print(f"Test gestational age range: {y_test.min():.1f} - {y_test.max():.1f}")
    
    # Feature selection and scaling pipeline
    print("\nPerforming feature selection and scaling...")
    
    # Select top features based on correlation with target
    feature_selector = SelectKBest(score_func=f_regression, k=min(1000, train_expression.shape[1]))
    scaler = StandardScaler()
    
    # Fit feature selector and scaler on training data
    X_train_selected = feature_selector.fit_transform(train_expression, y_train)
    X_train_scaled = scaler.fit_transform(X_train_selected)
    
    # Transform test data
    X_test_selected = feature_selector.transform(test_expression)
    X_test_scaled = scaler.transform(X_test_selected)
    
    print(f"Selected features: {X_train_scaled.shape[1]}")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Define models to try
    models = {
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42, max_iter=2000),
        'ElasticNet': ElasticNet(random_state=42, max_iter=2000),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    }
    
    best_model = None
    best_score = float('inf')
    best_model_name = None
    results = {}
    
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train_split, y_train_split)
        
        # Predict on validation set
        val_pred = grid_search.predict(X_val_split)
        val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
        val_r2 = r2_score(y_val_split, val_pred)
        
        results[name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation RMSE: {val_rmse:.3f}")
        print(f"  Validation R2: {val_r2:.3f}")
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = grid_search.best_estimator_
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} (Validation RMSE: {best_score:.3f})")
    
    # Retrain best model on full training data
    print(f"\nRetraining {best_model_name} on full training data...")
    best_model.fit(X_train_scaled, y_train)
    
    # Final predictions on test set
    y_pred = best_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test R2: {test_r2:.3f}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Gestational Age (weeks)', fontsize=12)
    plt.ylabel('Predicted Gestational Age (weeks)', fontsize=12)
    plt.title(f'Predicted vs Actual Gestational Age\n{best_model_name} Model (RMSE: {test_rmse:.3f}, R2: {test_r2:.3f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add perfect prediction line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.legend()
    
    # Add text box with statistics
    textstr = f'RMSE: {test_rmse:.3f}\nR2: {test_r2:.3f}\nSamples: {len(y_test)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('gestational_age_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model, test_rmse, test_r2, results

def main():
    """Main function"""
    print("Starting GSE149440 analysis...")
    
    # Download and process data
    metadata_df, expression_df = download_and_process_data()
    
    # Prepare data
    metadata_clean, expression_clean = prepare_data(metadata_df, expression_df)
    
    # Train and evaluate models
    best_model, test_rmse, test_r2, results = train_and_evaluate_models(metadata_clean, expression_clean)
    
    print(f"\nAnalysis complete!")
    print(f"Best model achieved Test RMSE: {test_rmse:.3f}")
    print(f"Results saved to 'gestational_age_prediction_results.png'")

if __name__ == "__main__":
    main()