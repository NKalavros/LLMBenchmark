import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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
    
    # Get the expression data
    print("Extracting expression data...")
    expression_data = None
    sample_metadata = None
    
    # GSE data structure - get the first platform
    for gpl_name, gpl in gse.gpls.items():
        print(f"Found platform: {gpl_name}")
        
    # Get expression matrix and sample info
    for gsm_name, gsm in gse.gsms.items():
        if expression_data is None:
            # Initialize with first sample
            expression_data = pd.DataFrame(index=gsm.table['ID_REF'])
            sample_metadata = pd.DataFrame()
        
        # Add expression values for this sample
        expression_data[gsm_name] = gsm.table['VALUE'].values
        
        # Extract metadata for this sample
        metadata_row = {}
        for key, value in gsm.metadata.items():
            if isinstance(value, list):
                metadata_row[key] = value[0] if len(value) > 0 else ''
            else:
                metadata_row[key] = value
        
        sample_metadata = pd.concat([sample_metadata, pd.DataFrame([metadata_row])], ignore_index=True)
    
    # Convert expression data to numeric
    expression_data = expression_data.apply(pd.to_numeric, errors='coerce')
    
    print(f"Expression data shape: {expression_data.shape}")
    print(f"Sample metadata shape: {sample_metadata.shape}")
    
    return expression_data, sample_metadata, gse

def extract_metadata_variables(sample_metadata):
    """Extract gestational age and training set indicators from metadata"""
    print("Extracting metadata variables...")
    
    # Look for gestational age and training set variables
    gestational_age = None
    train_indicator = None
    
    # Print available columns to understand the data structure
    print("Available metadata columns:")
    for col in sample_metadata.columns:
        print(f"  {col}")
        if 'gestational' in col.lower() or 'age' in col.lower():
            print(f"    Potential gestational age column: {col}")
            print(f"    Sample values: {sample_metadata[col].head()}")
        if 'train' in col.lower():
            print(f"    Potential training indicator column: {col}")
            print(f"    Sample values: {sample_metadata[col].head()}")
    
    # Try to find gestational age column
    for col in sample_metadata.columns:
        if 'gestational age:ch1' in col or 'gestational_age' in col.lower():
            gestational_age = sample_metadata[col]
            print(f"Found gestational age in column: {col}")
            break
    
    # Try to find training indicator column
    for col in sample_metadata.columns:
        if 'train:ch1' in col or 'train' in col.lower():
            train_indicator = sample_metadata[col]
            print(f"Found training indicator in column: {col}")
            break
    
    # If not found, look in characteristics columns
    if gestational_age is None or train_indicator is None:
        print("Looking in characteristics columns...")
        for col in sample_metadata.columns:
            if 'characteristics' in col.lower():
                print(f"Checking characteristics column: {col}")
                sample_chars = sample_metadata[col].head()
                print(f"Sample characteristics: {sample_chars}")
                
                # Parse characteristics if they contain the needed info
                for idx, chars in enumerate(sample_metadata[col]):
                    if pd.isna(chars):
                        continue
                    if isinstance(chars, str):
                        if 'gestational age' in chars.lower() and gestational_age is None:
                            # Extract gestational age values
                            gestational_age = sample_metadata[col].apply(
                                lambda x: extract_value_from_characteristics(x, 'gestational age') if pd.notna(x) else None
                            )
                            print("Extracted gestational age from characteristics")
                        if 'train' in chars.lower() and train_indicator is None:
                            # Extract training indicator
                            train_indicator = sample_metadata[col].apply(
                                lambda x: extract_value_from_characteristics(x, 'train') if pd.notna(x) else None
                            )
                            print("Extracted training indicator from characteristics")
    
    return gestational_age, train_indicator

def extract_value_from_characteristics(characteristics_str, key):
    """Extract value from characteristics string"""
    if pd.isna(characteristics_str) or not isinstance(characteristics_str, str):
        return None
    
    # Look for key: value pattern
    lines = characteristics_str.split('\n') if '\n' in characteristics_str else [characteristics_str]
    for line in lines:
        if key.lower() in line.lower():
            if ':' in line:
                value = line.split(':', 1)[1].strip()
                # Try to convert to numeric if possible
                try:
                    return float(value)
                except:
                    return value
    return None

def preprocess_data(expression_data, gestational_age, train_indicator):
    """Preprocess the data for modeling"""
    print("Preprocessing data...")
    
    # Convert to numeric and handle missing values
    gestational_age = pd.to_numeric(gestational_age, errors='coerce')
    train_indicator = pd.to_numeric(train_indicator, errors='coerce')
    
    # Create a combined dataframe
    data_df = pd.DataFrame({
        'sample_id': expression_data.columns,
        'gestational_age': gestational_age.values,
        'train_indicator': train_indicator.values
    })
    
    # Remove samples with missing values
    data_df = data_df.dropna()
    
    print(f"Samples after removing missing values: {len(data_df)}")
    print(f"Training samples: {sum(data_df['train_indicator'] == 1)}")
    print(f"Test samples: {sum(data_df['train_indicator'] == 0)}")
    
    # Filter expression data to match
    valid_samples = data_df['sample_id'].tolist()
    expression_filtered = expression_data[valid_samples]
    
    # Remove genes with too many missing values or low variance
    expression_filtered = expression_filtered.dropna(thresh=len(valid_samples) * 0.8)  # Keep genes with <20% missing
    expression_filtered = expression_filtered.fillna(expression_filtered.mean(axis=1))  # Fill remaining with mean
    
    # Remove low variance genes
    gene_var = expression_filtered.var(axis=1)
    high_var_genes = gene_var[gene_var > gene_var.quantile(0.1)].index  # Keep top 90% by variance
    expression_filtered = expression_filtered.loc[high_var_genes]
    
    print(f"Genes after filtering: {expression_filtered.shape[0]}")
    
    return expression_filtered.T, data_df  # Transpose so samples are rows

def build_and_evaluate_model(X, y, train_mask, test_mask):
    """Build and evaluate prediction models"""
    print("Building and evaluating models...")
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Feature selection - select top k features
    print("Performing feature selection...")
    selector = SelectKBest(score_func=f_regression, k=min(1000, X_train.shape[1]))
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
    
    # Try different models
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('inf')
    best_name = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name in ['Ridge', 'Lasso', 'ElasticNet']:
            # Grid search for regularization models
            if name == 'Ridge':
                param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            elif name == 'Lasso':
                param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
            else:  # ElasticNet
                param_grid = {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
            
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train_split, y_train_split)
            model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            # For tree-based models, use default parameters to save time
            model.fit(X_train_split, y_train_split)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val_split)
        val_rmse = np.sqrt(mean_squared_error(y_val_split, val_pred))
        print(f"{name} validation RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_name = name
    
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
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Add diagonal line for perfect prediction
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('Actual Gestational Age', fontsize=12)
    plt.ylabel('Predicted Gestational Age', fontsize=12)
    plt.title(f'Predicted vs Actual Gestational Age\n{model_name} Model (Test RMSE: {rmse:.4f})', fontsize=14)
    plt.legend()
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gestational_age_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the analysis"""
    try:
        # Download and parse GEO data
        expression_data, sample_metadata, gse = download_and_parse_geo_data('GSE149440')
        
        # Extract metadata variables
        gestational_age, train_indicator = extract_metadata_variables(sample_metadata)
        
        if gestational_age is None:
            print("Could not find gestational age variable!")
            return
        if train_indicator is None:
            print("Could not find training indicator variable!")
            return
        
        # Preprocess data
        X, data_df = preprocess_data(expression_data, gestational_age, train_indicator)
        
        # Create train/test masks
        train_mask = data_df['train_indicator'] == 1
        test_mask = data_df['train_indicator'] == 0
        
        if sum(train_mask) == 0:
            print("No training samples found!")
            return
        if sum(test_mask) == 0:
            print("No test samples found!")
            return
        
        # Build and evaluate model
        model, test_pred, y_test, test_rmse, model_name = build_and_evaluate_model(
            X.values, data_df['gestational_age'].values, train_mask.values, test_mask.values
        )
        
        # Create scatter plot
        create_scatter_plot(y_test, test_pred, test_rmse, model_name)
        
        print(f"\nFinal Results:")
        print(f"Model: {model_name}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Scatter plot saved as 'gestational_age_prediction.png'")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
