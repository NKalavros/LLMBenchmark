#!/usr/bin/env python3
"""
Debug script for GSE149440 dataset
"""

import GEOparse
import pandas as pd
import numpy as np

def debug_dataset():
    """
    Debug the GSE149440 dataset to understand its structure
    """
    print("=== Debugging GSE149440 Dataset ===\n")
    
    try:
        # Download the dataset
        print("Downloading dataset...")
        gse = GEOparse.get_GEO(geo="GSE149440")
        
        # Get expression data
        expression_data = gse.pivot_samples('VALUE')
        
        # Get metadata
        metadata = gse.phenotype_data
        
        print(f"Expression data shape: {expression_data.shape}")
        print(f"Metadata shape: {metadata.shape}")
        print(f"Expression data columns (samples): {len(expression_data.columns)}")
        print(f"Expression data index (genes): {len(expression_data.index)}")
        print(f"Metadata index (samples): {len(metadata.index)}")
        
        # Check if sample names match
        expression_samples = set(expression_data.columns)
        metadata_samples = set(metadata.index)
        
        print(f"\nSamples in expression data: {len(expression_samples)}")
        print(f"Samples in metadata: {len(metadata_samples)}")
        print(f"Intersection: {len(expression_samples & metadata_samples)}")
        print(f"Expression only: {len(expression_samples - metadata_samples)}")
        print(f"Metadata only: {len(metadata_samples - expression_samples)}")
        
        # Check metadata columns
        print(f"\nMetadata columns:")
        for i, col in enumerate(metadata.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Check specific columns
        print(f"\nChecking gestational age column...")
        if 'characteristics_ch1.1.gestational age' in metadata.columns:
            ga_col = metadata['characteristics_ch1.1.gestational age']
            print(f"Gestational age column found")
            print(f"  Unique values: {ga_col.unique()}")
            print(f"  Non-null values: {ga_col.notna().sum()}")
            print(f"  Null values: {ga_col.isna().sum()}")
            print(f"  Sample values: {ga_col.head(10).tolist()}")
        else:
            print("Gestational age column NOT found!")
            
        print(f"\nChecking train column...")
        if 'characteristics_ch1.4.train' in metadata.columns:
            train_col = metadata['characteristics_ch1.4.train']
            print(f"Train column found")
            print(f"  Unique values: {train_col.unique()}")
            print(f"  Non-null values: {train_col.notna().sum()}")
            print(f"  Null values: {train_col.isna().sum()}")
            print(f"  Value counts:")
            print(train_col.value_counts(dropna=False))
        else:
            print("Train column NOT found!")
        
        # Check for alternative column names
        print(f"\nSearching for alternative column names...")
        ga_keywords = ['gestational', 'age', 'gest']
        train_keywords = ['train', 'split', 'set']
        
        for col in metadata.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ga_keywords):
                print(f"  Potential gestational age column: {col}")
                print(f"    Values: {metadata[col].unique()}")
            if any(keyword in col_lower for keyword in train_keywords):
                print(f"  Potential train column: {col}")
                print(f"    Values: {metadata[col].unique()}")
        
        # Try to find any numeric columns that could be gestational age
        print(f"\nChecking all numeric columns...")
        for col in metadata.columns:
            try:
                numeric_data = pd.to_numeric(metadata[col], errors='coerce')
                non_null_count = numeric_data.notna().sum()
                if non_null_count > 0:
                    print(f"  {col}: {non_null_count} non-null numeric values")
                    print(f"    Range: {numeric_data.min():.2f} - {numeric_data.max():.2f}")
                    print(f"    Sample values: {numeric_data.dropna().head(5).tolist()}")
            except:
                pass
        
        # Check if we can find any train/test split
        print(f"\nChecking for any binary columns that could be train/test split...")
        for col in metadata.columns:
            unique_vals = metadata[col].unique()
            if len(unique_vals) <= 3:  # Binary or ternary
                print(f"  {col}: {unique_vals}")
                print(f"    Value counts: {metadata[col].value_counts(dropna=False)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset() 