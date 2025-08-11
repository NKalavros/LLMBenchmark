#!/usr/bin/env python3
"""
Optimized Gestational Age Prediction from Gene Expression Data
Dataset: GSE149440
This script uses advanced techniques to minimize RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import GEOparse
import warnings
warnings.filterwarnings('ignore')

def download_gse149440():
    """
    Download the GSE149440 dataset using GEOparse
    """
    print("Downloading GSE149440 dataset...")
    try:
        # Download the dataset
        gse = GEOparse.get_GEO(geo="GSE149440")
        
        # Get expression data
        expression_data = gse.pivot_samples('VALUE')
        
        # Get metadata
        metadata = gse.phenotype_data
        
        print(f"Dataset downloaded successfully!")
        print(f"Expression data shape: {expression_data.shape}")
        print(f"Metadata shape: {metadata.shape}")
        
        return expression_data, metadata
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None

def prepare_data(expression_data, metadata):
    """
    Prepare the data for modeling with advanced preprocessing
    """
    print("Preparing data for modeling...")
    
    # Extract gestational age and training split information
    # Use the second set of columns which have more data and proper train/test split
    gestational_age = metadata['characteristics_ch1.2.gestational age']
    train_split = metadata['characteristics_ch1.5.train']
    
    # Convert to numeric, handling any non-numeric values
    gestational_age = pd.to_numeric(gestational_age, errors='coerce')
    train_split = pd.to_numeric(train_split, errors='coerce')
    
    # Remove samples with missing values
    valid_mask = ~(gestational_age.isna() | train_split.isna())
    gestational_age = gestational_age[valid_mask]
    train_split = train_split[valid_mask]
    
    # Get the valid sample names
    valid_samples = gestational_age.index
    
    # Filter expression data to only include valid samples
    expression_data = expression_data.loc[:, valid_samples]
    
    print(f"Valid samples: {len(gestational_age)}")
    print(f"Training samples: {sum(train_split == 1)}")
    print(f"Test samples: {sum(train_split == 0)}")
    
    # Split data
    train_mask = train_split == 1
    test_mask = train_split == 0
    
    # Transpose expression data so samples are rows and genes are columns
    expression_data = expression_data.T
    
    X_train = expression_data.loc[train_mask.index[train_mask], :]
    y_train = gestational_age[train_mask]
    X_test = expression_data.loc[test_mask.index[test_mask], :]
    y_test = gestational_age[test_mask]
    
    return X_train, y_train, X_test, y_test

def advanced_feature_selection(X_train, y_train, X_test, max_features=2000):
    """
    Advanced feature selection using multiple methods
    """
    print(f"Performing advanced feature selection...")
    
    # Method 1: Correlation-based selection
    correlations = np.abs([np.corrcoef(X_train.iloc[:, i], y_train)[0, 1] 
                          for i in range(X_train.shape[1])])
    correlation_threshold = np.percentile(correlations, 95)  # Top 5%
    corr_features = correlations > correlation_threshold
    
    # Method 2: F-regression
    f_selector = SelectKBest(score_func=f_regression, k=min(max_features//2, X_train.shape[1]))
    f_selector.fit(X_train, y_train)
    f_features = f_selector.get_support()
    
    # Method 3: Random Forest feature importance
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    rf_importance = rf_selector.feature_importances_
    rf_threshold = np.percentile(rf_importance, 90)  # Top 10%
    rf_features = rf_importance > rf_threshold
    
    # Combine all methods (union)
    combined_features = corr_features | f_features | rf_features
    
    # Ensure we don't exceed max_features
    if combined_features.sum() > max_features:
        # Use correlation scores to select top features
        feature_scores = correlations * combined_features
        top_indices = np.argsort(feature_scores)[-max_features:]
        final_features = np.zeros_like(combined_features)
        final_features[top_indices] = True
        combined_features = final_features
    
    selected_features = X_train.columns[combined_features]
    
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]
    
    print(f"Selected {len(selected_features)} features from {X_train.shape[1]} total features")
    
    return X_train_selected, X_test_selected, selected_features

def create_advanced_models():
    """
    Create advanced model ensemble with optimized hyperparameters
    """
    models = {}
    
    # 1. Ridge with optimized parameters
    ridge_params = {
        'alpha': [0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    models['Ridge'] = (Ridge(), ridge_params)
    
    # 2. ElasticNet (combines L1 and L2 regularization)
    elastic_params = {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [2000]
    }
    models['ElasticNet'] = (ElasticNet(random_state=42, max_iter=2000), elastic_params)
    
    # 3. Huber Regressor (robust to outliers)
    huber_params = {
        'epsilon': [1.1, 1.2, 1.3, 1.4, 1.5],
        'alpha': [0.001, 0.01, 0.1, 1]
    }
    models['Huber'] = (HuberRegressor(max_iter=1000), huber_params)
    
    # 4. SVR with RBF kernel
    svr_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    }
    models['SVR'] = (SVR(), svr_params)
    
    # 5. Random Forest with more trees
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    models['RandomForest'] = (RandomForestRegressor(random_state=42, n_jobs=-1), rf_params)
    
    # 6. Gradient Boosting with optimized parameters
    gb_params = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10]
    }
    models['GradientBoosting'] = (GradientBoostingRegressor(random_state=42), gb_params)
    
    # 7. Extra Trees (more robust than Random Forest)
    et_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    models['ExtraTrees'] = (ExtraTreesRegressor(random_state=42, n_jobs=-1), et_params)
    
    # 8. Neural Network
    mlp_params = {
        'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (200, 100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [1000]
    }
    models['MLP'] = (MLPRegressor(random_state=42, early_stopping=True), mlp_params)
    
    return models

def train_optimized_models(X_train, y_train):
    """
    Train multiple optimized models and find the best ensemble
    """
    print("Training optimized models...")
    
    # Create advanced models
    models = create_advanced_models()
    
    # Scale the data using RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    best_models = {}
    cv_scores = {}
    
    # Train each model with cross-validation
    for name, (model, params) in models.items():
        print(f"Training {name}...")
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            model, 
            params, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best score (convert from negative MSE)
        score = -grid_search.best_score_
        cv_rmse = np.sqrt(score)
        
        best_models[name] = grid_search.best_estimator_
        cv_scores[name] = cv_rmse
        
        print(f"{name} - Best CV RMSE: {cv_rmse:.4f}")
    
    # Create ensemble of top models
    print("\nCreating ensemble...")
    
    # Select top 3 models based on CV performance
    top_models = sorted(cv_scores.items(), key=lambda x: x[1])[:3]
    print(f"Top 3 models: {[name for name, _ in top_models]}")
    
    # Create voting regressor with top models
    estimators = [(name, best_models[name]) for name, _ in top_models]
    ensemble = VotingRegressor(estimators=estimators, n_jobs=-1)
    
    # Train ensemble
    ensemble.fit(X_train_scaled, y_train)
    
    return ensemble, scaler, best_models, cv_scores

def evaluate_optimized_model(ensemble, scaler, X_test, y_test, best_models, cv_scores):
    """
    Evaluate the optimized ensemble model
    """
    print("Evaluating optimized ensemble model...")
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions with ensemble
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    
    # Calculate metrics for ensemble
    rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    
    print(f"Ensemble - Test RMSE: {rmse_ensemble:.4f}")
    print(f"Ensemble - Test R²: {r2_ensemble:.4f}")
    print(f"Ensemble - Test MAE: {mae_ensemble:.4f}")
    
    # Also evaluate individual best models
    individual_results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        individual_results[name] = {'RMSE': rmse, 'R²': r2}
        print(f"{name} - Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return y_pred_ensemble, rmse_ensemble, r2_ensemble, individual_results

def plot_optimized_results(y_test, y_pred, rmse, r2, individual_results):
    """
    Create comprehensive visualization of results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot of predicted vs actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=50)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Gestational Age (weeks)')
    axes[0, 0].set_ylabel('Predicted Gestational Age (weeks)')
    axes[0, 0].set_title(f'Ensemble: Predicted vs Actual\nRMSE: {rmse:.4f}, R²: {r2:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Gestational Age (weeks)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model comparison
    model_names = list(individual_results.keys())
    model_rmse = [individual_results[name]['RMSE'] for name in model_names]
    model_r2 = [individual_results[name]['R²'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, model_rmse, width, label='RMSE', alpha=0.8)
    axes[1, 0].bar(x + width/2, model_r2, width, label='R²', alpha=0.8)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Model Performance Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_gestational_age_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the optimized pipeline
    """
    print("=== Optimized Gestational Age Prediction from Gene Expression Data ===")
    print("Dataset: GSE149440\n")
    
    # Download dataset
    expression_data, metadata = download_gse149440()
    
    if expression_data is None or metadata is None:
        print("Failed to download dataset. Exiting.")
        return
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data(expression_data, metadata)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("No valid training or test data found. Exiting.")
        return
    
    # Advanced feature selection
    X_train_selected, X_test_selected, selected_features = advanced_feature_selection(X_train, y_train, X_test)
    
    # Train optimized models
    ensemble, scaler, best_models, cv_scores = train_optimized_models(X_train_selected, y_train)
    
    # Evaluate on test set
    y_pred, rmse, r2, individual_results = evaluate_optimized_model(ensemble, scaler, X_test_selected, y_test, best_models, cv_scores)
    
    # Plot results
    plot_optimized_results(y_test, y_pred, rmse, r2, individual_results)
    
    print(f"\n=== Final Optimized Results ===")
    print(f"Ensemble Test RMSE: {rmse:.4f}")
    print(f"Ensemble Test R²: {r2:.4f}")
    print(f"Number of features used: {len(selected_features)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    
    # Find best individual model
    best_individual = min(individual_results.items(), key=lambda x: x[1]['RMSE'])
    print(f"Best individual model: {best_individual[0]} (RMSE: {best_individual[1]['RMSE']:.4f})")

if __name__ == "__main__":
    main() 