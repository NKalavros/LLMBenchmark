#!/usr/bin/env python3
"""
Check metadata columns for GSE149440 dataset
"""

import GEOparse
import pandas as pd

def check_metadata():
    """
    Check what columns are available in the metadata
    """
    print("Checking GSE149440 metadata columns...")
    
    try:
        # Download the dataset
        gse = GEOparse.get_GEO(geo="GSE149440")
        
        # Get metadata
        metadata = gse.phenotype_data
        
        # Get relevant columns
        gestational_age = metadata['characteristics_ch1.1.gestational age']
        train_split = metadata['characteristics_ch1.4.train']
        
        # Convert to numeric
        gestational_age = pd.to_numeric(gestational_age, errors='coerce')
        train_split = pd.to_numeric(train_split, errors='coerce')
        
        print("\nValue counts for train variable:")
        print(train_split.value_counts(dropna=False))
        
        print("\nNumber of non-missing gestational age values for each train group:")
        for val in train_split.unique():
            count = ((train_split == val) & (~gestational_age.isna())).sum()
            print(f"train == {val}: {count} samples with non-missing gestational age")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_metadata() 