#!/usr/bin/env python3
"""
Enhanced Gene Expression Analysis for Gestational Age Prediction
Dataset: GSE149440 from Gene Expression Omnibus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import GEOparse
import warnings
warnings.filterwarnings('ignore')

def download_and_parse_data():
    """Download and parse GSE149440 dataset"""
    print("Loading GSE149440 dataset...")
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
                    age_str = char.split(':')[1].strip()
                    sample_info['gestational_age'] = float(age_str)
                except:
                    pass
            elif 'train' in char.lower():
                try:
                    train_str = char.split(':')[1].strip()
                    sample_info['train'] = int(train_str)
                except:
                    pass
        
        metadata.append(sample_info)
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.set_index('sample_id', inplace=True)
    
    return expression_data, metadata_df

def preprocess_data(expression_data, metadata_df):
    """Enhanced preprocessing with better outlier handling"""
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
    
    # More aggressive filtering of low-variance genes
    gene_variance = expression_data.var()
    high_var_genes = gene_variance[gene_variance > gene_variance.quantile(0.25)].index
    expression_data = expression_data[high_var_genes]
    
    # Remove genes with too many outliers (using IQR method)
    Q1 = expression_data.quantile(0.25)
    Q3 = expression_data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = 0.1  # Allow up to 10% outliers per gene
    
    good_genes = []
    for gene in expression_data.columns:
        outliers = ((expression_data[gene] < (Q1[gene] - 1.5 * IQR[gene])) | 
                   (expression_data[gene] > (Q3[gene] + 1.5 * IQR[gene]))).sum()
        if outliers / len(expression_data) <= outlier_threshold:
            good_genes.append(gene)
    
    expression_data = expression_data[good_genes]
    
    # Fill any remaining missing values with median
    expression_data = expression_data.fillna(expression_data.median())
    
    print(f"Final dataset shape: {expression_data.shape}")
    print(f"Number of samples: {len(metadata_df)}")
    print(f"Train samples: {sum(metadata_df['train'] == 1)}")
    print(f"Test samples: {sum(metadata_df['train'] == 0)}")
    
    return expression_data, metadata_df

def advanced_feature_selection(X_train, y_train, n_features=2000):
    """Multi-stage feature selection"""
    print(f"Advanced feature selection...")
    
    # Stage 1: Univariate selection (top 5000 features)
    selector1 = SelectKBest(score_func=f_regression, k=min(5000, X_train.shape[1]))
    X_stage1 = selector1.fit_transform(X_train, y_train)
    features_stage1 = X_train.columns[selector1.get_support()]
    
    # Stage 2: L1-based selection using Lasso
    X_stage1_df = pd.DataFrame(X_stage1, columns=features_stage1, index=X_train.index)
    scaler_temp = StandardScaler()
    X_scaled_temp = scaler_temp.fit_transform(X_stage1_df)
    
    lasso_selector = SelectFromModel(Lasso(alpha=0.01), max_features=n_features)
    X_stage2 = lasso_selector.fit_transform(X_scaled_temp, y_train)
    
    # Get final selected features
    selected_mask = lasso_selector.get_support()
    final_features = features_stage1[selected_mask]
    
    print(f"Selected {len(final_features)} features after 2-stage selection")
    
    return selector1, lasso_selector, final_features

def train_enhanced_models(X_train, y_train, X_val, y_val):
    """Train enhanced models with better hyperparameter tuning"""
    print("Training enhanced models...")
    
    models = {
        'ElasticNet': ElasticNet(max_iter=3000, random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(max_iter=3000, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
    }
    
    param_grids = {
        'ElasticNet': {
            'alpha': [0.01, 0.1, 1.0, 10.0], 
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        },
        'Lasso': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'GradientBoosting': {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'RandomForest': {
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
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
    
    # Try ensemble of top 3 models
    sorted_models = sorted(best_scores.items(), key=lambda x: x[1])[:3]
    print(f"Creating ensemble of top 3 models: {[name for name, _ in sorted_models]}")
    
    ensemble_models = [best_models[name] for name, _ in sorted_models]
    ensemble_weights = [1/score for _, score in sorted_models]
    ensemble_weights = np.array(ensemble_weights) / sum(ensemble_weights)
    
    # Evaluate ensemble
    ensemble_pred = np.zeros(len(X_val))
    for model, weight in zip(ensemble_models, ensemble_weights):
        ensemble_pred += weight * model.predict(X_val)
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    print(f"Ensemble validation RMSE: {ensemble_rmse:.4f}")
    
    if ensemble_rmse < best_scores[best_model_name]:
        print("Ensemble performs better, using ensemble model")
        return ensemble_models, ensemble_weights, "Ensemble"
    else:
        return best_model, None, best_model_name

def evaluate_model(model, weights, X_test, y_test, model_name):
    """Evaluate model on test set"""
    print(f"\nEvaluating {model_name} on test set...")
    
    # Make predictions
    if model_name == "Ensemble":
        y_pred = np.zeros(len(X_test))
        for m, w in zip(model, weights):
            y_pred += w * m.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return y_pred, test_rmse

def plot_results(y_test, y_pred, test_rmse, model_name):
    """Create enhanced scatter plot"""
    plt.figure(figsize=(12, 10))
    
    # Scatter plot with better styling
    plt.scatter(y_test, y_pred, alpha=0.6, s=60, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual Gestational Age (weeks)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Gestational Age (weeks)', fontsize=14, fontweight='bold')
    plt.title(f'Predicted vs Actual Gestational Age\n{model_name} - Test RMSE: {test_rmse:.4f}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    mae = np.mean(np.abs(y_test - y_pred))
    
    stats_text = f'Correlation: {correlation:.3f}\nMAE: {mae:.3f}\nRMSE: {test_rmse:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
             verticalalignment='top')
    
    # Add trend line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(sorted(y_test), p(sorted(y_test)), "g--", alpha=0.8, linewidth=2, label='Trend Line')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('gestational_age_prediction_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Enhanced analysis pipeline"""
    try:
        # Download and parse data (reuse existing data if available)
        expression_data, metadata_df = download_and_parse_data()
        
        # Enhanced preprocessing
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
        
        # Advanced feature selection
        selector1, selector2, selected_features = advanced_feature_selection(X_train, y_train, n_features=1500)
        
        # Apply feature selection to all sets
        X_train_stage1 = selector1.transform(X_train)
        X_val_stage1 = selector1.transform(X_val)
        X_test_stage1 = selector1.transform(X_test)
        
        # Scale features using RobustScaler (better for outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_stage1)
        X_val_scaled = scaler.transform(X_val_stage1)
        X_test_scaled = scaler.transform(X_test_stage1)
        
        # Apply second stage selection
        X_train_final = selector2.transform(X_train_scaled)
        X_val_final = selector2.transform(X_val_scaled)
        X_test_final = selector2.transform(X_test_scaled)
        
        # Train enhanced models
        best_model, weights, model_name = train_enhanced_models(X_train_final, y_train, X_val_final, y_val)
        
        # Evaluate on test set
        y_pred, test_rmse = evaluate_model(best_model, weights, X_test_final, y_test, model_name)
        
        # Plot results
        plot_results(y_test, y_pred, test_rmse, model_name)
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZED RESULTS")
        print(f"{'='*60}")
        print(f"Best Model: {model_name}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test Correlation: {np.corrcoef(y_test, y_pred)[0, 1]:.4f}")
        print(f"Test MAE: {np.mean(np.abs(y_test - y_pred)):.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()