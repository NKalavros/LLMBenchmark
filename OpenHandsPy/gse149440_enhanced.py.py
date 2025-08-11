#!/usr/bin/env python3
"""
Enhanced Gene Expression Analysis for Gestational Age Prediction
Dataset: GSE149440 from Gene Expression Omnibus

This script implements advanced feature selection and ensemble methods
to optimize the prediction of gestational age.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import requests
import gzip
import io
import gc
import warnings
warnings.filterwarnings('ignore')

def download_and_parse_data():
    """Download and parse GSE149440 data efficiently"""
    print("Downloading GSE149440 dataset...")
    
    matrix_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149440/matrix/GSE149440_series_matrix.txt.gz"
    
    try:
        response = requests.get(matrix_url, timeout=300)
        response.raise_for_status()
        
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
            content = gz_file.read().decode('utf-8')
        
        print("Successfully downloaded dataset. Parsing...")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None
    
    lines = content.split('\n')
    
    # Extract metadata
    sample_info = {}
    data_start = None
    
    for i, line in enumerate(lines):
        if line.startswith('!Sample_geo_accession'):
            accessions = line.split('\t')[1:]
            sample_info['accessions'] = [acc.strip('"') for acc in accessions if acc.strip()]
        elif line.startswith('!Sample_characteristics_ch1'):
            if 'characteristics' not in sample_info:
                sample_info['characteristics'] = []
            chars = line.split('\t')[1:]
            sample_info['characteristics'].append([char.strip('"') for char in chars if char.strip()])
        elif line.startswith('!series_matrix_table_begin'):
            data_start = i + 1
            break
    
    # Find data end
    data_end = None
    for i in range(data_start, len(lines)):
        if lines[i].startswith('!series_matrix_table_end'):
            data_end = i
            break
    
    if data_end is None:
        data_end = len(lines)
    
    # Parse expression data more selectively
    data_lines = [line for line in lines[data_start:data_end] if line.strip()]
    headers = data_lines[0].split('\t')
    sample_names = headers[1:]
    
    print(f"Found {len(sample_names)} samples")
    
    # Use variance-based gene selection during parsing
    print("Processing expression data with variance-based selection...")
    
    gene_data = []
    gene_ids = []
    
    # First pass: collect all data to calculate variance
    all_data = []
    all_genes = []
    
    for i, line in enumerate(data_lines[1:]):
        if i % 5 == 0:  # Sample every 5th gene for memory efficiency
            parts = line.split('\t')
            if len(parts) >= len(headers):
                try:
                    expr_values = [float(x) if x != '' and x != 'null' else np.nan 
                                 for x in parts[1:len(headers)]]
                    if not all(np.isnan(expr_values)):  # Skip genes with all NaN values
                        all_data.append(expr_values)
                        all_genes.append(parts[0])
                except:
                    continue
    
    # Calculate variance and select top genes
    variances = []
    for data_row in all_data:
        var = np.nanvar(data_row)
        variances.append(var if not np.isnan(var) else 0)
    
    # Select top 5000 most variable genes
    top_indices = np.argsort(variances)[-5000:]
    
    for idx in top_indices:
        gene_data.append(all_data[idx])
        gene_ids.append(all_genes[idx])
    
    print(f"Selected {len(gene_ids)} most variable genes")
    
    # Create expression DataFrame
    expr_df = pd.DataFrame(gene_data, columns=sample_names, index=gene_ids)
    
    # Extract metadata
    metadata = {}
    sample_names_meta = sample_info.get('accessions', [])
    
    for char_list in sample_info['characteristics']:
        for i, char in enumerate(char_list):
            if i >= len(sample_names_meta):
                break
            
            sample_name = sample_names_meta[i]
            if sample_name not in metadata:
                metadata[sample_name] = {}
            
            if ':' in char:
                key, value = char.split(':', 1)
                key = key.strip().strip('"')
                value = value.strip().strip('"')
                metadata[sample_name][key] = value
    
    metadata_df = pd.DataFrame.from_dict(metadata, orient='index')
    
    return expr_df, metadata_df

def main():
    print("Starting enhanced GSE149440 analysis...")
    
    # Load data
    expr_df, metadata_df = download_and_parse_data()
    if expr_df is None:
        return
    
    # Clean up memory
    gc.collect()
    
    # Find required columns
    age_cols = [col for col in metadata_df.columns if 'age' in col.lower() or 'gestational' in col.lower()]
    train_cols = [col for col in metadata_df.columns if 'train' in col.lower()]
    
    if not age_cols or not train_cols:
        print("Required columns not found")
        return
    
    metadata_df['gestational_age'] = pd.to_numeric(metadata_df[age_cols[0]], errors='coerce')
    metadata_df['train_flag'] = pd.to_numeric(metadata_df[train_cols[0]], errors='coerce')
    
    # Align data
    if len(expr_df.columns) == len(metadata_df):
        expr_df.columns = metadata_df.index
    
    # Filter valid samples
    valid_samples = metadata_df.dropna(subset=['gestational_age', 'train_flag']).index
    common_samples = list(set(expr_df.columns) & set(valid_samples))
    
    expr_df = expr_df[common_samples]
    metadata_df = metadata_df.loc[common_samples]
    
    print(f"Working with {len(common_samples)} samples and {expr_df.shape[0]} genes")
    
    # Split data
    train_mask = metadata_df['train_flag'] == 1
    test_mask = metadata_df['train_flag'] == 0
    
    X_train = expr_df.loc[:, train_mask].T
    y_train = metadata_df.loc[train_mask, 'gestational_age']
    X_test = expr_df.loc[:, test_mask].T
    y_test = metadata_df.loc[test_mask, 'gestational_age']
    
    print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Fill missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Create validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print("Building enhanced prediction models...")
    
    # Define enhanced models with different feature selection methods
    models = {
        'ElasticNet_F': Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_regression, k=1000)),
            ('regressor', ElasticNet(random_state=42))
        ]),
        'ElasticNet_MI': Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(mutual_info_regression, k=800)),
            ('regressor', ElasticNet(random_state=42))
        ]),
        'Ridge_F': Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_regression, k=1200)),
            ('regressor', Ridge(random_state=42))
        ]),
        'Lasso': Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_regression, k=800)),
            ('regressor', Lasso(random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selection', SelectKBest(f_regression, k=600)),
            ('regressor', GradientBoostingRegressor(random_state=42, n_estimators=100))
        ])
    }
    
    # Enhanced parameter grids
    param_grids = {
        'ElasticNet_F': {
            'regressor__alpha': [0.01, 0.1, 1.0, 5.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'ElasticNet_MI': {
            'regressor__alpha': [0.01, 0.1, 1.0, 5.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'Ridge_F': {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 50.0]
        },
        'Lasso': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 5.0]
        },
        'GradientBoosting': {
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__subsample': [0.8, 0.9, 1.0]
        }
    }
    
    best_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            grid_search = GridSearchCV(
                model, param_grids[name], 
                cv=3, scoring='neg_mean_squared_error',
                n_jobs=1, verbose=0
            )
            
            grid_search.fit(X_train_split, y_train_split)
            
            # Evaluate on validation set
            val_pred = grid_search.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            
            results[name] = {
                'model': grid_search.best_estimator_,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'best_params': grid_search.best_params_
            }
            
            best_models[name] = grid_search.best_estimator_
            
            print(f"{name} - Val RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}, MAE: {val_mae:.3f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    if not best_models:
        print("No models were successfully trained.")
        return
    
    # Create ensemble model
    print("Creating ensemble model...")
    
    # Select top 3 models for ensemble
    sorted_models = sorted(results.items(), key=lambda x: x[1]['val_rmse'])[:3]
    ensemble_models = [(name, results[name]['model']) for name, _ in sorted_models]
    
    ensemble = VotingRegressor(ensemble_models)
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble on validation set
    val_pred_ensemble = ensemble.predict(X_val)
    val_rmse_ensemble = np.sqrt(mean_squared_error(y_val, val_pred_ensemble))
    val_r2_ensemble = r2_score(y_val, val_pred_ensemble)
    
    print(f"Ensemble - Val RMSE: {val_rmse_ensemble:.3f}, R²: {val_r2_ensemble:.3f}")
    
    # Select best individual model
    best_name = sorted_models[0][0]
    best_model = results[best_name]['model']
    best_rmse = results[best_name]['val_rmse']
    
    # Choose between best individual model and ensemble
    if val_rmse_ensemble < best_rmse:
        final_model = ensemble
        final_name = "Ensemble"
        print(f"Using ensemble model (RMSE: {val_rmse_ensemble:.3f})")
    else:
        final_model = best_model
        final_name = best_name
        print(f"Using {best_name} model (RMSE: {best_rmse:.3f})")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_pred = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n=== ENHANCED RESULTS ===")
    print(f"Final Model: {final_name}")
    print(f"Test RMSE: {test_rmse:.3f} weeks")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.3f} weeks")
    
    if final_name != "Ensemble":
        print(f"Best parameters: {results[final_name]['best_params']}")
    
    # Create enhanced scatter plot
    plt.figure(figsize=(12, 10))
    
    # Main scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, test_pred, alpha=0.6, s=50, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Gestational Age (weeks)')
    plt.ylabel('Predicted Gestational Age (weeks)')
    plt.title(f'Predicted vs Actual Gestational Age\n{final_name} Model (RMSE: {test_rmse:.3f})')
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {test_r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = test_pred - y_test
    plt.scatter(test_pred, residuals, alpha=0.6, s=50, color='orange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Gestational Age (weeks)')
    plt.ylabel('Residuals (weeks)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Residuals (weeks)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Model comparison
    plt.subplot(2, 2, 4)
    model_names = list(results.keys())
    rmse_values = [results[name]['val_rmse'] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), rmse_values, color='lightcoral')
    plt.xlabel('Models')
    plt.ylabel('Validation RMSE')
    plt.title('Model Comparison')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Highlight best model
    best_idx = rmse_values.index(min(rmse_values))
    bars[best_idx].set_color('darkred')
    
    plt.tight_layout()
    plt.savefig('/workspace/gestational_age_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n=== ENHANCED SUMMARY ===")
    print(f"Dataset: GSE149440 (enhanced processing)")
    print(f"Total samples: {len(metadata_df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Genes used: {X_train.shape[1]} (top variable genes)")
    print(f"Final model: {final_name}")
    print(f"Test RMSE: {test_rmse:.3f} weeks")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.3f} weeks")
    
    print("\nModel Performance Summary:")
    for name, result in sorted(results.items(), key=lambda x: x[1]['val_rmse']):
        print(f"  {name}: RMSE={result['val_rmse']:.3f}, R²={result['val_r2']:.3f}")

if __name__ == "__main__":
    main()