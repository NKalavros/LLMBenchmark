# Gestational Age Prediction from Gene Expression Data

## Overview
This project implements a machine learning pipeline to predict gestational age from gene expression data using the GSE149440 dataset from the Gene Expression Omnibus (GEO).

## Dataset Information
- **Dataset**: GSE149440
- **Expression Data**: 32,830 genes × 735 samples
- **Training Samples**: 367
- **Test Samples**: 368
- **Target Variable**: Gestational age (weeks)

## Methodology

### Data Processing
1. **Download**: Downloaded GSE149440 dataset using GEOquery
2. **Feature Selection**: Selected top 2,000 genes based on correlation with gestational age
3. **Data Split**: Used predefined train/test split from metadata (`train:ch1` variable)

### Machine Learning Models
The analysis implemented four different models:

1. **Random Forest** (RF)
   - RMSE: 6.12 weeks
   - R²: 0.429
   - MAE: 5.08 weeks

2. **XGBoost** (XGB)
   - RMSE: 5.60 weeks
   - R²: 0.523
   - MAE: 4.46 weeks

3. **Elastic Net** (ENET)
   - RMSE: 5.68 weeks
   - R²: 0.509
   - MAE: 4.50 weeks

4. **Support Vector Regression** (SVR)
   - RMSE: 5.55 weeks
   - R²: 0.530
   - MAE: 4.44 weeks

### Ensemble Model
- **Method**: Simple averaging of all four model predictions
- **RMSE**: 5.42 weeks
- **R²**: 0.553
- **MAE**: 4.37 weeks

## Results Summary

| Model | RMSE (weeks) | R² | MAE (weeks) |
|-------|-------------|----|-------------|
| Random Forest | 6.12 | 0.429 | 5.08 |
| XGBoost | 5.60 | 0.523 | 4.46 |
| Elastic Net | 5.68 | 0.509 | 4.50 |
| SVR | 5.55 | 0.530 | 4.44 |
| **Ensemble** | **5.42** | **0.553** | **4.37** |

## Key Findings

1. **Best Individual Model**: Support Vector Regression achieved the best performance among individual models with RMSE = 5.55 weeks.

2. **Ensemble Improvement**: The ensemble model improved performance by 0.13 RMSE units compared to the best individual model.

3. **Model Performance**: All models achieved reasonable prediction accuracy, with the ensemble model explaining 55.3% of the variance in gestational age.

## Files Generated

- `gestational_age_prediction.R`: Main analysis script
- `final_results.R`: Results summary and visualization
- `final_results_scatter.png`: Scatter plot of predicted vs actual gestational age
- `model_evaluation_results.csv`: Detailed model performance metrics
- `plots/`: Directory containing individual model scatter plots and comparison charts

## Technical Details

### Dependencies
- R packages: GEOquery, Biobase, dplyr, ggplot2, caret, randomForest, glmnet, xgboost, e1071

### Feature Selection
- Selected top 2,000 genes based on absolute correlation with gestational age
- Used only training data for feature selection to avoid data leakage

### Model Parameters
- **Random Forest**: 500 trees, mtry = sqrt(n_features)
- **XGBoost**: max_depth=6, eta=0.1, early stopping
- **Elastic Net**: alpha=0.5, 5-fold CV
- **SVR**: radial kernel, default parameters

## Conclusion

The ensemble model successfully predicted gestational age from gene expression data with an RMSE of 5.42 weeks, demonstrating the potential of machine learning approaches for gestational age prediction from transcriptomic data. The ensemble approach provided consistent improvements over individual models, highlighting the value of model combination in genomic prediction tasks. 