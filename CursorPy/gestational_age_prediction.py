#!/usr/bin/env python3
"""
Gestational Age Prediction from Gene Expression Data
Dataset: GSE149440
This script downloads the dataset, builds a prediction model for gestational age,
and evaluates it on the test set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
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
    Prepare the data for modeling
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

def feature_selection(X_train, y_train, X_test, k=1000):
    """
    Select top k features based on correlation with target
    """
    print(f"Selecting top {k} features...")
    
    # Use f_regression to select features
    selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]
    
    print(f"Selected {len(selected_features)} features")
    
    return X_train_selected, X_test_selected, selected_features

def train_models(X_train, y_train):
    """
    Train multiple models and find the best one
    """
    print("Training models...")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define models to try
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Define parameter grids for hyperparameter tuning
    param_grids = {
        'Ridge': {'alpha': [0.1, 1, 10, 100]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10]},
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    best_model = None
    best_score = float('inf')
    best_model_name = None
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best score (convert from negative MSE)
        score = -grid_search.best_score_
        
        print(f"{name} - Best CV RMSE: {np.sqrt(score):.4f}")
        
        if score < best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with CV RMSE: {np.sqrt(best_score):.4f}")
    
    return best_model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the model on test set
    """
    print("Evaluating model on test set...")
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return y_pred, rmse, r2

def plot_results(y_test, y_pred, rmse, r2):
    """
    Create scatter plot of predicted vs actual values
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=50)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('Actual Gestational Age', fontsize=12)
    plt.ylabel('Predicted Gestational Age', fontsize=12)
    plt.title(f'Predicted vs Actual Gestational Age\nRMSE: {rmse:.4f}, R²: {r2:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\nN: {len(y_test)}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('gestational_age_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the entire pipeline
    """
    print("=== Gestational Age Prediction from Gene Expression Data ===")
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
    
    # Feature selection
    X_train_selected, X_test_selected, selected_features = feature_selection(X_train, y_train, X_test)
    
    # Train models
    best_model, scaler = train_models(X_train_selected, y_train)
    
    # Evaluate on test set
    y_pred, rmse, r2 = evaluate_model(best_model, scaler, X_test_selected, y_test)
    
    # Plot results
    plot_results(y_test, y_pred, rmse, r2)
    
    print(f"\n=== Final Results ===")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Number of features used: {len(selected_features)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

if __name__ == "__main__":
    main() 