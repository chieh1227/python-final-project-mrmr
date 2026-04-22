# AutoPrognosis + MRMR Feature Selection

Extended the [AutoPrognosis](https://github.com/vanderschaarlab/autoprognosis) framework with **MRMR (Minimum Redundancy Maximum Relevance)** feature selection for automated clinical prognosis modeling. This was a group course project — my contribution focused on implementing and evaluating the MRMR feature selection plugin.

## What I Built

### MRMR Feature Selection Plugin

Added a custom sklearn-compatible MRMR feature selector as an AutoPrognosis plugin, enabling automated ML pipelines to leverage information-theoretic feature selection.

**Algorithm:**
- **Relevance**: Measured via `mutual_info_classif` (mutual information between each feature and target)
- **Redundancy**: Mean absolute Pearson correlation with already-selected features
- **Selection**: Greedy forward selection maximizing `relevance - redundancy`

**Implementation highlights:**
- Fully integrated into AutoPrognosis's plugin architecture (auto-discovered, no core modifications needed)
- Hyperparameter tuning support for `n_features` (number of features to select)
- Proper cross-validation handling — feature selection runs per fold to prevent data leakage

### Experiment: MRMR vs. Baseline on Pima Diabetes Dataset

Compared MRMR feature selection against baseline approaches (no selection, PCA, Variance Threshold) using:
- 5 classifiers: XGBoost, Random Forest, Logistic Regression, CatBoost, Linear SVM
- 5 random seeds × 5-fold cross-validation
- Metrics: ROC-AUC, F1, Accuracy

## Project Structure

```
├── src/autoprognosis/plugins/preprocessors/
│   └── dimensionality_reduction/
│       └── plugin_mrmr.py          # MRMR plugin implementation
├── notebooks/
│   └── pima_mrmr_experiment.ipynb   # Comparative experiments
└── 演算法查詢_驗證_wayne.py           # Model analysis & parameter extraction tool
```

## Key Files (My Contributions)

| File | Description |
|------|-------------|
| `plugin_mrmr.py` | MRMR feature selector + AutoPrognosis plugin wrapper |
| `pima_mrmr_experiment.ipynb` | Reproducible experiments comparing MRMR with baselines |
| `演算法查詢_驗證_wayne.py` | Utility for extracting hyperparameters from fitted AutoPrognosis models |

## Tech Stack

Python · AutoPrognosis · scikit-learn · XGBoost · CatBoost · Jupyter Notebook

## About AutoPrognosis

[AutoPrognosis](https://github.com/vanderschaarlab/autoprognosis) is an AutoML framework for clinical prognosis — it automatically searches for optimal ML pipeline configurations including imputation, preprocessing, and prediction.
