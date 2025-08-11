#!/usr/bin/env python3
"""
Gene Expression Analysis for Gestational Age Prediction
Dataset: GSE149440 from Gene Expression Omnibus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import GEOparse
import warnings
warnings.filterwarnings('ignore')

def download_and_parse_data():
    """Download and parse GSE149440 dataset"""
    print("Downloading GSE149440 dataset...")
    gse = GEOparse.get_GEO(geo="GSE149440", destdir="./data")
    
    # Get expression data
    print("Extracting expression data...")
    expression_data = gse.pivot_samples('VALUE')
    
    # Get metadata
    print("Extracting metadata...")
    metadata = []
    for gsm_name, gsm in gse.gsms.items():
        sample_info = {
            'sample_id': gsm_name,
            'gestational_age': None,
            'train': None
        }
        
        # Extract gestational age and train/test split info
        for char in gsm.metadata.get('characteristics_ch1', []):
            if 'gestational age' in char.lower():
                try:
                    # Extract numeric value from string like "gestational age: 25.5"
                    age_str = char.split(':')[1].strip()
                    sample_info['gestational_age'] = float(age_str)
                except:
                    pass
            elif 'train' in char.lower():
                try:
                    # Extract train/test indicator
                    train_str = char.split(':')[1].strip()
                    sample_info['train'] = int(train_str)
                except:
                    pass
        
        metadata.append(sample_info)
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.set_index('sample_id', inplace=True)
    
    return expression_data, metadata_df

def preprocess_data(expression_data, metadata_df):
    """Preprocess expression and metadata"""
    print("Preprocessing data...")
    
    # Align samples between expression and metadata
    common_samples = expression_data.columns.intersection(metadata_df.index)
    expression_data = expression_data[common_samples]
    metadata_df = metadata_df.loc[common_samples]
    
    # Remove samples with missing gestational age or train/test info
    valid_samples = metadata_df.dropna().index
    expression_data = expression_data[valid_samples]
    metadata_df = metadata_df.loc[valid_samples]
    
    # Transpose expression data so samples are rows
    expression_data = expression_data.T
    
    # Remove genes with low variance or too many missing values
    gene_variance = expression_data.var()
    high_var_genes = gene_variance[gene_variance > gene_variance.quantile(0.1)].index
    expression_data = expression_data[high_var_genes]
    
    # Fill any remaining missing values with median
    expression_data = expression_data.fillna(expression_data.median())
    
    print(f"Final dataset shape: {expression_data.shape}")
    print(f"Number of samples: {len(metadata_df)}")
    print(f"Train samples: {sum(metadata_df['train'] == 1)}")
    print(f"Test samples: {sum(metadata_df['train'] == 0)}")
    
    return expression_data, metadata_df

def feature_selection(X_train, y_train, n_features=1000):
    """Select top features using univariate feature selection"""
    print(f"Selecting top {n_features} features...")
    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    
    print(f"Selected {len(selected_features)} features")
    return selector, selected_features

def train_models(X_train, y_train, X_val, y_val):
    """Train and compare different models"""
    print("Training models...")
    
    models = {
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(max_iter=2000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    param_grids = {
        'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'RandomForest': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    }
    
    best_models = {}
    best_scores = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = grid_search.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        best_models[name] = grid_search.best_estimator_
        best_scores[name] = val_rmse
        
        print(f"{name} - Best params: {grid_search.best_params_}")
        print(f"{name} - Validation RMSE: {val_rmse:.4f}")
    
    # Select best model
    best_model_name = min(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]
    
    print(f"\nBest model: {best_model_name} (RMSE: {best_scores[best_model_name]:.4f})")
    
    return best_model, best_model_name

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set"""
    print(f"\nEvaluating {model_name} on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return y_pred, test_rmse

def plot_results(y_test, y_pred, test_rmse, model_name):
    """Create scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual Gestational Age (weeks)', fontsize=12)
    plt.ylabel('Predicted Gestational Age (weeks)', fontsize=12)
    plt.title(f'Predicted vs Actual Gestational Age\n{model_name} - Test RMSE: {test_rmse:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('gestational_age_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis pipeline"""
    try:
        # Download and parse data
        expression_data, metadata_df = download_and_parse_data()
        
        # Preprocess data
        expression_data, metadata_df = preprocess_data(expression_data, metadata_df)
        
        # Split into train and test based on metadata
        train_mask = metadata_df['train'] == 1
        test_mask = metadata_df['train'] == 0
        
        X_train_full = expression_data[train_mask]
        y_train_full = metadata_df.loc[train_mask, 'gestational_age']
        X_test = expression_data[test_mask]
        y_test = metadata_df.loc[test_mask, 'gestational_age']
        
        # Create validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Feature selection
        selector, selected_features = feature_selection(X_train, y_train, n_features=1000)
        
        # Apply feature selection to all sets
        X_train_selected = selector.transform(X_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_val_selected = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train models
        best_model, model_name = train_models(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate on test set
        y_pred, test_rmse = evaluate_model(best_model, X_test_scaled, y_test, model_name)
        
        # Plot results
        plot_results(y_test, y_pred, test_rmse, model_name)
        
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Best Model: {model_name}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test Correlation: {np.corrcoef(y_test, y_pred)[0, 1]:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()