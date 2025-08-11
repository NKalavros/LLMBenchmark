import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Step 2: Download and parse the GSE149440 Series Matrix file
gse = GEOparse.get_GEO(geo="GSE149440", destdir="./")

# Debug: print type and attributes of gse object
print(f"Type of gse object: {type(gse)}")
print(f"Attributes of gse object: {dir(gse)}")
if hasattr(gse, 'gsms'):
    print(f"Number of samples in gse.gsms: {len(gse.gsms)}")
else:
    print("gse object does not have 'gsms' attribute.")

# Step 3: Extract expression matrix and relevant metadata
sample_ids = list(gse.gsms.keys())


# Extract metadata and build a DataFrame
metadata = []
for i, gsm_id in enumerate(sample_ids):
    gsm = gse.gsms[gsm_id]
    meta = gsm.metadata
    if i < 5:
        print(f"Metadata keys for sample {gsm_id}: {list(meta.keys())}")
        print(f"Metadata for sample {gsm_id}: {meta}")
    # Extract gestational age and train from characteristics_ch1
    gest_age = np.nan
    train_flag = np.nan
    characteristics = meta.get('characteristics_ch1', [])
    for item in characteristics:
        if item.lower().startswith('gestational age:'):
            try:
                gest_age = float(item.split(':', 1)[1].strip())
            except Exception:
                gest_age = np.nan
        if item.lower().startswith('train:'):
            try:
                train_flag = int(item.split(':', 1)[1].strip())
            except Exception:
                train_flag = np.nan
    metadata.append({
        'gsm_id': gsm_id,
        'gestational_age': gest_age,
        'train': train_flag
    })
metadata_df = pd.DataFrame(metadata)

# Convert gestational_age and train to numeric
metadata_df['gestational_age'] = pd.to_numeric(metadata_df['gestational_age'], errors='coerce')
metadata_df['train'] = pd.to_numeric(metadata_df['train'], errors='coerce')

# Build expression matrix (genes x samples)
exprs = []
gene_ids = None
for gsm_id in sample_ids:
    gsm = gse.gsms[gsm_id]
    table = gsm.table
    if gene_ids is None:
        gene_ids = table['ID_REF'].values
    exprs.append(table['VALUE'].values)
exprs = np.array(exprs).T  # shape: genes x samples
exprs_df = pd.DataFrame(exprs, index=gene_ids, columns=sample_ids)

# Save intermediate files for inspection
metadata_df.to_csv('metadata_gse149440.csv', index=False)
exprs_df.to_csv('expression_gse149440.csv')
 
# Feature selection: select top 500 most variable genes
gene_variances = exprs_df.var(axis=1)
top_n = 500
top_genes = gene_variances.sort_values(ascending=False).head(top_n).index
exprs_df = exprs_df.loc[top_genes]




# Step 4: Build training and test sets
print('Unique values in metadata_df["train"]:', metadata_df['train'].unique())
print('Value counts in metadata_df["train"]:', metadata_df['train'].value_counts())
train_samples = metadata_df[metadata_df['train'] == 1]['gsm_id'].values
test_samples = metadata_df[metadata_df['train'] == 0]['gsm_id'].values



X_train = exprs_df[train_samples].T
y_train = metadata_df.set_index('gsm_id').reindex(list(train_samples))['gestational_age']
X_test = exprs_df[test_samples].T
y_test = metadata_df.set_index('gsm_id').reindex(list(test_samples))['gestational_age']

# Step 5: Preprocess data (impute missing, standardize)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 6-7: Fit and optimize regression model (RidgeCV with internal CV)
alphas = np.logspace(-2, 2, 10)
ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
ridge.fit(X_train_scaled, y_train)

# Step 8: Predict and evaluate RMSE
y_pred = ridge.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE) on test set: {rmse:.3f}')

# Step 9: Scatter plot predicted vs actual gestational age
plt.figure(figsize=(7,7))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Predicted vs Actual Gestational Age (Test Set)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.savefig('predicted_vs_actual_gestational_age.png')
plt.show()
