import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import GEOparse
import warnings
warnings.filterwarnings('ignore')

def download_and_parse_geo_data(geo_id):
    """Download and parse GEO dataset"""
    print(f"Downloading GEO dataset {geo_id}...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir="./")
    
    # Extract expression data and metadata
    expression_data = None
    sample_metadata = []
    
    print("Extracting expression data and metadata...")
    for gsm_name, gsm in gse.gsms.items():
        if expression_data is None:
            expression_data = pd.DataFrame(index=gsm.table['ID_REF'])
        
        expression_data[gsm_name] = pd.to_numeric(gsm.table['VALUE'], errors='coerce')
        
        # Extract metadata
        metadata_row = {'sample_id': gsm_name}
        
        # Parse characteristics_ch1 which contains the key information
        if 'characteristics_ch1' in gsm.metadata:
            characteristics = gsm.metadata['characteristics_ch1']
            for char in characteristics:
                if ':' in char:
                    key, value = char.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Convert numeric values
                    try:
                        if key in ['gestational_age', 'train', 'gadelivery', 'batch']:
                            value = float(value) if '.' in value else int(value)
                    except:
                        pass
                    
                    metadata_row[key] = value
        
        # Add other relevant metadata
        for key, value in gsm.metadata.items():
            if key not in ['characteristics_ch1']:
                if isinstance(value, list):
                    metadata_row[key] = value[0] if len(value) > 0 else ''
                else:
                    metadata_row[key] = value
        
        sample_metadata.append(metadata_row)
    
    metadata_df = pd.DataFrame(sample_metadata)
    metadata_df.set_index('sample_id', inplace=True)
    
    print(f"Expression data shape: {expression_data.shape}")
    print(f"Metadata shape: {metadata_df.shape}")
    
    return expression_data, metadata_df

def extract_metadata_variables(metadata_df):
    """Extract gestational age and training set indicators from metadata"""
    print("Extracting metadata variables...")
    
    # Print available columns
    print("Available metadata columns:")
    for col in metadata_df.columns:
        print(f"  {col}")
    
    gestational_age = None
    train_indicator = None
    
    # Look for gestational age
    if 'gestational_age' in metadata_df.columns:
        gestational_age = metadata_df['gestational_age']
        print("Found gestational_age column")
    
    # Look for training indicator
    if 'train' in metadata_df.columns:
        train_indicator = metadata_df['train']
        print("Found train column")
    
    if gestational_age is not None:
        gestational_age = pd.to_numeric(gestational_age, errors='coerce')
        print(f"Gestational age stats: min={gestational_age.min():.2f}, max={gestational_age.max():.2f}, mean={gestational_age.mean():.2f}")
    
    if train_indicator is not None:
        train_indicator = pd.to_numeric(train_indicator, errors='coerce')
        print(f"Training indicator distribution: {train_indicator.value_counts().to_dict()}")
    
    return gestational_age, train_indicator

def preprocess_data(expression_data, gestational_age, train_indicator):
    """Preprocess the data for modeling"""
    print("Preprocessing data...")
    
    # Align data
    common_samples = expression_data.columns.intersection(gestational_age.index).intersection(train_indicator.index)
    print(f"Common samples: {len(common_samples)}")
    
    expression_filtered = expression_data[common_samples]
    gestational_age_filtered = gestational_age[common_samples]
    train_indicator_filtered = train_indicator[common_samples]
    
    # Remove samples with missing values
    valid_mask = gestational_age_filtered.notna() & train_indicator_filtered.notna()
    expression_filtered = expression_filtered.loc[:, valid_mask]
    gestational_age_filtered = gestational_age_filtered[valid_mask]
    train_indicator_filtered = train_indicator_filtered[valid_mask]
    
    print(f"Samples after removing missing values: {len(gestational_age_filtered)}")
    print(f"Training samples: {sum(train_indicator_filtered == 1)}")
    print(f"Test samples: {sum(train_indicator_filtered == 0)}")
    
    # Filter genes
    print("Filtering genes...")
    print(f"Initial genes: {expression_filtered.shape[0]}")
    print(f"Initial samples: {expression_filtered.shape[1]}")
    
    # Check for missing values
    missing_per_gene = expression_filtered.isnull().sum(axis=1)
    print(f"Genes with missing values: {sum(missing_per_gene > 0)}")
    
    # More lenient missing value filtering - keep genes with at least 20% non-missing values
    min_samples = max(1, int(len(gestational_age_filtered) * 0.2))
    expression_filtered = expression_filtered.dropna(thresh=min_samples)
    print(f"Genes after missing value filtering (thresh={min_samples}): {expression_filtered.shape[0]}")
    
    # If still no genes, try without missing value filtering
    if expression_filtered.shape[0] == 0:
        print("No genes passed missing value filter, using all genes...")
        expression_filtered = expression_data[common_samples].loc[:, valid_mask]
    
    # Fill remaining missing values with gene mean, then with 0 if still NaN
    print("Filling missing values...")
    expression_filtered = expression_filtered.fillna(expression_filtered.mean(axis=1))
    expression_filtered = expression_filtered.fillna(0)  # Fill any remaining NaN with 0
    
    # Verify no NaN values remain
    nan_count = expression_filtered.isnull().sum().sum()
    print(f"Remaining NaN values: {nan_count}")
    
    # Check for constant genes (zero variance)
    gene_var = expression_filtered.var(axis=1)
    non_constant_genes = gene_var[gene_var > 0]
    print(f"Non-constant genes: {len(non_constant_genes)}")
    
    if len(non_constant_genes) > 0:
        expression_filtered = expression_filtered.loc[non_constant_genes.index]
        
        # Only apply variance filtering if we have many genes
        if len(non_constant_genes) > 1000:
            # Keep top 90% by variance
            high_var_genes = gene_var[gene_var > gene_var.quantile(0.1)].index
            expression_filtered = expression_filtered.loc[high_var_genes]
            print(f"Genes after variance filtering: {expression_filtered.shape[0]}")
    else:
        print("All genes have zero variance, keeping all genes...")
    
    print(f"Final genes after filtering: {expression_filtered.shape[0]}")
    
    return expression_filtered.T, gestational_age_filtered, train_indicator_filtered

def build_and_evaluate_model(X, y, train_indicator):
    """Build and evaluate prediction models"""
    print("Building and evaluating models...")
    
    train_mask = train_indicator == 1
    test_mask = train_indicator == 0
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Feature selection - select top features
    print("Performing feature selection...")
    k_features = min(1000, X_train.shape[1])  # Select top 1000 features or all if less
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Selected features: {X_train_selected.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Try different models with hyperparameter tuning
    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
        },
        'Lasso': {
            'model': Lasso(max_iter=2000),
            'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}
        },
        'ElasticNet': {
            'model': ElasticNet(max_iter=2000),
            'params': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'params': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'params': {'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5]}
        }
    }
    
    best_model = None
    best_score = float('inf')
    best_name = None
    
    for name, model_info in models.items():
        print(f"Training {name}...")
        
        try:
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                model_info['model'], 
                model_info['params'], 
                cv=5, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1
            )
            grid_search.fit(X_train_split, y_train_split)
            
            # Get best model
            model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            
            # Evaluate on validation set
            val_pred = model.predict(X_val_split)
            val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
            print(f"{name} validation RMSE: {val_rmse:.4f}")
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    if best_model is None:
        print("No model could be trained successfully!")
        return None, None, None, None, None
    
    print(f"Best model: {best_name} with validation RMSE: {best_score:.4f}")
    
    # Retrain best model on full training set
    print("Retraining best model on full training set...")
    best_model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    test_pred = best_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return best_model, test_pred, y_test, test_rmse, best_name

def create_scatter_plot(y_true, y_pred, rmse, model_name):
    """Create scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line for perfect prediction
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('Actual Gestational Age (weeks)', fontsize=12)
    plt.ylabel('Predicted Gestational Age (weeks)', fontsize=12)
    plt.title(f'Predicted vs Actual Gestational Age\n{model_name} Model (Test RMSE: {rmse:.4f})', fontsize=14)
    plt.legend()
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=11)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gestational_age_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Scatter plot saved as 'gestational_age_prediction.png'")

def main():
    """Main function to run the analysis"""
    try:
        print("Starting gestational age prediction analysis...")
        print("=" * 50)
        
        # Download and parse GEO data
        expression_data, metadata_df = download_and_parse_geo_data('GSE149440')
        
        if expression_data is None or metadata_df is None:
            print("Failed to download or parse data!")
            return
        
        # Extract metadata variables
        gestational_age, train_indicator = extract_metadata_variables(metadata_df)
        
        if gestational_age is None:
            print("Could not find gestational age variable!")
            return
        if train_indicator is None:
            print("Could not find training indicator variable!")
            return
        
        # Preprocess data
        X, y, train_ind = preprocess_data(expression_data, gestational_age, train_indicator)
        
        if len(y) == 0:
            print("No valid samples found after preprocessing!")
            return
        
        # Check if we have both training and test samples
        if sum(train_ind == 1) == 0:
            print("No training samples found!")
            return
        if sum(train_ind == 0) == 0:
            print("No test samples found!")
            return
        
        # Build and evaluate model
        model, test_pred, y_test, test_rmse, model_name = build_and_evaluate_model(
            X.values, y.values, train_ind.values
        )
        
        if model is None:
            print("Model training failed!")
            return
        
        # Create scatter plot
        create_scatter_plot(y_test, test_pred, test_rmse, model_name)
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS:")
        print(f"Best Model: {model_name}")
        print(f"Test RMSE: {test_rmse:.4f} weeks")
        print(f"Test samples: {len(y_test)}")
        print(f"Correlation: {np.corrcoef(y_test, test_pred)[0, 1]:.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
