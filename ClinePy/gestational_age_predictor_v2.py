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
import warnings
warnings.filterwarnings('ignore')

try:
    import GEOparse
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    print("GEOparse not available, will try alternative approach")

def download_geo_data_alternative(geo_id):
    """Alternative method to download GEO data using direct URLs"""
    import urllib.request
    import gzip
    import os
    
    print(f"Downloading GEO dataset {geo_id} using alternative method...")
    
    # URLs for GSE149440
    series_matrix_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"
    
    try:
        # Download series matrix file
        print("Downloading series matrix file...")
        urllib.request.urlretrieve(series_matrix_url, f"{geo_id}_series_matrix.txt.gz")
        
        # Extract the file
        with gzip.open(f"{geo_id}_series_matrix.txt.gz", 'rt') as f:
            content = f.read()
        
        with open(f"{geo_id}_series_matrix.txt", 'w') as f:
            f.write(content)
        
        print("File downloaded and extracted successfully")
        return f"{geo_id}_series_matrix.txt"
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def parse_series_matrix(filename):
    """Parse the series matrix file"""
    print("Parsing series matrix file...")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the data matrix
    data_start = None
    sample_info = {}
    
    for i, line in enumerate(lines):
        if line.startswith('!Sample_title'):
            sample_titles = line.strip().split('\t')[1:]
        elif line.startswith('!Sample_geo_accession'):
            sample_ids = line.strip().split('\t')[1:]
        elif line.startswith('!Sample_characteristics_ch1'):
            # Parse characteristics
            chars = line.strip().split('\t')[1:]
            for j, char in enumerate(chars):
                if j < len(sample_ids):
                    if sample_ids[j] not in sample_info:
                        sample_info[sample_ids[j]] = {}
                    
                    # Parse characteristics string
                    if 'gestational age' in char.lower():
                        try:
                            age_val = float(char.split(':')[1].strip().split()[0])
                            sample_info[sample_ids[j]]['gestational_age'] = age_val
                        except:
                            pass
                    elif 'train' in char.lower():
                        try:
                            train_val = int(char.split(':')[1].strip())
                            sample_info[sample_ids[j]]['train'] = train_val
                        except:
                            pass
        elif line.startswith('!series_matrix_table_begin'):
            data_start = i + 1
            break
    
    if data_start is None:
        print("Could not find data matrix start")
        return None, None
    
    # Read the expression data
    print("Reading expression data...")
    data_lines = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line.startswith('!series_matrix_table_end'):
            break
        if line and not line.startswith('!'):
            data_lines.append(line.split('\t'))
    
    if not data_lines:
        print("No data found")
        return None, None
    
    # Create expression dataframe
    header = data_lines[0]
    gene_ids = [row[0] for row in data_lines[1:]]
    expression_values = [row[1:] for row in data_lines[1:]]
    
    expression_df = pd.DataFrame(expression_values, index=gene_ids, columns=header[1:])
    expression_df = expression_df.apply(pd.to_numeric, errors='coerce')
    
    # Create metadata dataframe
    metadata_df = pd.DataFrame.from_dict(sample_info, orient='index')
    metadata_df.index.name = 'sample_id'
    
    print(f"Expression data shape: {expression_df.shape}")
    print(f"Metadata shape: {metadata_df.shape}")
    
    return expression_df, metadata_df

def download_and_parse_geo_data(geo_id):
    """Download and parse GEO dataset"""
    if GEO_AVAILABLE:
        try:
            print(f"Downloading GEO dataset {geo_id} using GEOparse...")
            gse = GEOparse.get_GEO(geo=geo_id, destdir="./")
            
            # Extract expression data and metadata
            expression_data = None
            sample_metadata = []
            
            for gsm_name, gsm in gse.gsms.items():
                if expression_data is None:
                    expression_data = pd.DataFrame(index=gsm.table['ID_REF'])
                
                expression_data[gsm_name] = pd.to_numeric(gsm.table['VALUE'], errors='coerce')
                
                # Extract metadata
                metadata_row = {'sample_id': gsm_name}
                for key, value in gsm.metadata.items():
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
            
        except Exception as e:
            print(f"GEOparse failed: {e}")
            print("Trying alternative download method...")
    
    # Alternative download method
    matrix_file = download_geo_data_alternative(geo_id)
    if matrix_file:
        return parse_series_matrix(matrix_file)
    else:
        return None, None

def extract_metadata_variables(metadata_df):
    """Extract gestational age and training set indicators from metadata"""
    print("Extracting metadata variables...")
    
    gestational_age = None
    train_indicator = None
    
    # Print available columns
    print("Available metadata columns:")
    for col in metadata_df.columns:
        print(f"  {col}")
    
    # Look for gestational age
    if 'gestational_age' in metadata_df.columns:
        gestational_age = metadata_df['gestational_age']
        print("Found gestational_age column")
    else:
        # Look in characteristics columns
        for col in metadata_df.columns:
            if 'characteristics' in col.lower():
                print(f"Checking {col} for gestational age...")
                sample_values = metadata_df[col].dropna().head()
                print(f"Sample values: {sample_values.tolist()}")
                
                # Try to extract gestational age from characteristics
                def extract_gestational_age(char_str):
                    if pd.isna(char_str):
                        return None
                    if 'gestational age' in str(char_str).lower():
                        try:
                            # Extract numeric value
                            parts = str(char_str).split(':')
                            if len(parts) > 1:
                                age_str = parts[1].strip().split()[0]
                                return float(age_str)
                        except:
                            pass
                    return None
                
                extracted_ages = metadata_df[col].apply(extract_gestational_age)
                if extracted_ages.notna().sum() > 0:
                    gestational_age = extracted_ages
                    print(f"Extracted gestational age from {col}")
                    break
    
    # Look for training indicator
    if 'train' in metadata_df.columns:
        train_indicator = metadata_df['train']
        print("Found train column")
    else:
        # Look in characteristics columns
        for col in metadata_df.columns:
            if 'characteristics' in col.lower():
                print(f"Checking {col} for training indicator...")
                
                def extract_train_indicator(char_str):
                    if pd.isna(char_str):
                        return None
                    if 'train' in str(char_str).lower():
                        try:
                            parts = str(char_str).split(':')
                            if len(parts) > 1:
                                train_str = parts[1].strip()
                                return int(train_str)
                        except:
                            pass
                    return None
                
                extracted_train = metadata_df[col].apply(extract_train_indicator)
                if extracted_train.notna().sum() > 0:
                    train_indicator = extracted_train
                    print(f"Extracted training indicator from {col}")
                    break
    
    if gestational_age is not None:
        print(f"Gestational age stats: min={gestational_age.min():.2f}, max={gestational_age.max():.2f}, mean={gestational_age.mean():.2f}")
    if train_indicator is not None:
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
    # Remove genes with too many missing values
    expression_filtered = expression_filtered.dropna(thresh=len(gestational_age_filtered) * 0.8)
    # Fill remaining missing values with gene mean
    expression_filtered = expression_filtered.fillna(expression_filtered.mean(axis=1))
    
    # Remove low variance genes
    gene_var = expression_filtered.var(axis=1)
    high_var_genes = gene_var[gene_var > gene_var.quantile(0.2)].index  # Keep top 80% by variance
    expression_filtered = expression_filtered.loc[high_var_genes]
    
    print(f"Genes after filtering: {expression_filtered.shape[0]}")
    
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
    
    # Feature selection
    print("Performing feature selection...")
    k_features = min(500, X_train.shape[1])  # Select top 500 features or all if less
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
    
    # Try different models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }
    
    best_model = None
    best_score = float('inf')
    best_name = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            model.fit(X_train_split, y_train_split)
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
        model, test_pred, y_test, test_rmse, model_name = build_and_evaluate_model(X.values, y.values, train_ind.values)
        
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
