import GEOparse
import pandas as pd

# Load the dataset
gse = GEOparse.get_GEO(geo='GSE149440', destdir="./")

# Examine the first few samples to understand the metadata structure
sample_names = list(gse.gsms.keys())[:5]

print("Examining metadata structure:")
print("=" * 50)

for sample_name in sample_names:
    gsm = gse.gsms[sample_name]
    print(f"\nSample: {sample_name}")
    print("Characteristics:")
    
    for key, value in gsm.metadata.items():
        if 'characteristics' in key.lower():
            print(f"  {key}: {value}")
    
    # Also check other potentially relevant fields
    for key, value in gsm.metadata.items():
        if any(term in key.lower() for term in ['age', 'train', 'test', 'group']):
            print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("All metadata keys for first sample:")
first_sample = gse.gsms[sample_names[0]]
for key in first_sample.metadata.keys():
    print(f"  {key}")
