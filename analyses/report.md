# Agent Benchmark Report

Indexed Agents: BeastModev3, BeastModev3Py, ClaudeCode, Cline, ClinePy, Cursor, CursorPy, Kiro, KiroPy, OpenHandsPy, RooCode, RooCodePy, RovoDev, RovoDevPy
Total Code Files: 48

_No RMSE results available yet._

## Quality Heuristic (Top 5)
- RooCode :: RooCode/test_r.R => 3.5
- BeastModev3 :: BeastModev3/gse149440_metadata_check.R => 3.39
- BeastModev3 :: BeastModev3/gse149440_download.R => 3.33
- BeastModev3 :: BeastModev3/gse149440_preprocess.R => 3.3
- Cline :: Cline/install_packages.R => 3.27

## LLM Quality Metrics (Averages)
- readability: 0.866 (n=48)
- modularity: 0.367 (n=48)

## Sample Semantic Summaries
- BeastModev3/gse149440_extract.R: # Step 2: Extract expression matrix and sample metadata | # Load the ExpressionSet object | data_file <- "gse149440_expr...
- BeastModev3/gse149440_preprocess.R: # Step 4: Preprocess data (normalization, filtering) | # Load data | load("gse149440_model_data.RData") | # Filter to sa...
- [random_forest] BeastModev3/gse149440_evaluate_plot.R: # Load results | load("gse149440_model_results.RData") | # Compute RMSEs | rmse_enet <- sqrt(mean((pred_test_enet - y_te...
- BeastModev3/gse149440_download.R: # Step 1: Download and load the GSE149440 Series Matrix file in R using GEOquery | # Install GEOquery if not already ins...
- [elastic_net] BeastModev3/gse149440_model_fit.R: # Step 5: Fit Elastic Net regression model with cross-validation | # Load preprocessed data | load("gse149440_preprocess...
