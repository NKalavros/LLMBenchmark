#!/usr/bin/env python3
"""
Simple test script to check GEOparse functionality
"""

import sys
import os

def install_package(package):
    """Install package if not already installed"""
    try:
        __import__(package)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        os.system(f"{sys.executable} -m pip install {package}")

# Install required packages
required_packages = [
    'GEOparse',
    'pandas',
    'numpy'
]

print("=== Checking packages ===")
for package in required_packages:
    install_package(package)

print("\n=== Testing GEO download ===")
try:
    import GEOparse
    import pandas as pd
    import numpy as np
    
    print("Packages imported successfully!")
    
    # Test downloading a smaller dataset first
    print("Attempting to download GSE149440...")
    gse = GEOparse.get_GEO(geo="GSE149440", destdir="./data", silent=False)
    
    print(f"Dataset type: {type(gse)}")
    print(f"Available attributes: {dir(gse)}")
    
    if hasattr(gse, 'gsms'):
        print(f"Number of samples: {len(gse.gsms)}")
        sample_names = list(gse.gsms.keys())[:3]  # First 3 samples
        print(f"First few samples: {sample_names}")
        
        # Check one sample
        if sample_names:
            sample = gse.gsms[sample_names[0]]
            print(f"Sample metadata keys: {list(sample.metadata.keys())}")
            print(f"Characteristics: {sample.metadata.get('characteristics_ch1', [])}")
    
    print("Basic download test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()