# ROGII - Wellbore Geology Prediction

Kaggle competition solution: building machine learning models to predict geology along horizontal wellbores, contributing to the automation of drilling operations in the oil and gas industry.

**Competition:** [ROGII - Wellbore Geology Prediction](https://kaggle.com/competitions/rogii-wellbore-geology-prediction)
---

## Objective

Develop ML model that predict the geology encountered along a horizontal wellbore; specifically the target variable `tvt`, using drilling sensor and logging data.

Accurate predictions help:
- **Reduce resource waste** by minimizing redundant or misdirected drilling
- **Improve operational safety** by anticipating geological hazards
- **Enable automation** of well-steering decisions that currently depend on expert interpretation

---

## Competition Details

| Detail | Value |
|---|---|
| Start Date | May 5, 2026 |
| Entry / Team Merger Deadline | July 29, 2026 |
| Final Submission Deadline | August 5, 2026 |
| Submission Type | Kaggle Notebook (CPU or GPU, ≤ 9 hrs) |
| Internet Access | Disabled during submission |
| External Data | Allowed (freely & publicly available) |
| Submission File | `submission.csv` with columns `id`, `tvt` |

---

## Project Structure

```
kaggle-rogii-wellbore-geology-prediction/
├── data/
│   ├── raw/                    # Original competition data (train, test, sample_submission)
│   └── processed/              # Feature-engineered datasets ready for modeling
├── notebooks/
├── src/
│   ├── features.py             # Feature engineering utilities
│   ├── models.py               # Model definitions and training logic
│   └── utils.py                # Shared helpers (data loading, evaluation, etc.)
├── models/
│   └── ...                     # Serialized trained models
├── submissions/
│   └── submission.csv          # Latest submission file
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Solution Approach

### Problem Framing

This is a **regression task** on time-series / depth-series well data. Each row corresponds to a depth point along a horizontal wellbore, and the goal is to predict `tvt` (True Vertical Thickness), a measure of the geological layer thickness at that point.

### Key Challenges

- **Spatial autocorrelation:** measurements along the same wellbore are strongly correlated, requiring careful cross-validation (well-based splits rather than random).
- **Geological heterogeneity:** each well traverses different rock sequences, so models must generalize across varying lithologies.
- **Limited direct observations:** sensor data provides indirect proxies for rock properties; feature engineering is critical.

### Methodology

1. **Exploratory Data Analysis** — distribution of `tvt`, per-well statistics, sensor correlations, and depth-wise trends.
2. **Feature Engineering** — lag/lead features along the wellbore depth axis, rolling statistics, rate-of-change features, and cross-sensor interaction terms.
3. **Baseline Models** — LightGBM / XGBoost gradient boosting with well-based cross-validation folds.
4. **Advanced Models** — sequence models (LSTM / Transformer) to capture depth-sequential patterns; potentially stacked ensembles.
5. **Validation Strategy** — group k-fold by well ID to avoid data leakage and ensure realistic out-of-fold RMSE estimates.
6. **Ensemble** — blend tree-based and sequence models via weighted averaging or stacking.

### Libraries

- `lightgbm`, `xgboost`, `catboost` — gradient boosting
- `scikit-learn` — preprocessing, cross-validation, metrics
- `pytorch` / `tensorflow` — deep sequence models
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn`, `plotly` — visualization

---

## Submission Format

```
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
...
```

Each `id` is a composite key (`well_id` + `depth_index`). One prediction per row in the test set.

---

## Citation

Igor Kuvaev, Rafael Aguilar, John Granmayeh, Ryan Holbrook, María Cruz, and Ashley Oldacre. *ROGII - Wellbore Geology Prediction*. https://kaggle.com/competitions/rogii-wellbore-geology-prediction, 2026. Kaggle.
