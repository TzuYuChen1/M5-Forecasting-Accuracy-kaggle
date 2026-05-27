# M5 Sales Forecasting — Retail Demand Prediction at Scale

This project applies the M5 Forecasting Accuracy competition dataset — 30,490 Walmart product-store time series across California, Texas, and Wisconsin — to build a 28-day demand forecasting pipeline. The workflow covers data cleaning and memory optimization, external data integration, systematic feature engineering experimentation (16 controlled ablations from a 14-feature baseline), multi-model comparison (LightGBM, Random Forest, Ridge Regression, GRU), and two-phase hyperparameter tuning using grid search and Optuna. The final submission is a 3-seed LightGBM ensemble using recursive forecasting, achieving a private leaderboard WRMSSE of **0.59212**.

📍 University of Minnesota · Carlson School of Management · MSBA Program

---

## Business Problem

Accurate retail demand forecasting directly enables:

- **Inventory planning** — avoid stockouts on high-velocity items and reduce overstock on slow movers
- **Promotional response modeling** — quantify demand lift from SNAP benefit days and calendar events
- **Store-level resource allocation** — align replenishment schedules to actual expected sales patterns

This project simulates the core analytical workflow a demand forecasting team would execute: starting from raw historical sales data, engineering predictive features, selecting and tuning a model, and producing a competition-style 28-day forecast.

---

## Results

### Model Comparison

![Model Comparison](results/images/model_comparison.png)

| Model | RMSE | MAE | Notes |
|---|---|---|---|
| **LightGBM (Tuned)** | **2.0025** | **1.011** | Optuna trial 37; 18 features; 152 iterations |
| Random Forest | 2.0802 | 1.032 | n_estimators=300, max_depth=20 |
| Ridge Regression | 2.1180 | 1.051 | alpha=0.5 |
| GRU (RNN) | 2.2584 | 1.149 | seq_len=28, units=32; stopped at epoch 3 |
| LightGBM Baseline (E0) | 2.1619 | 1.017 | 14 features; no tuning |

### Kaggle Leaderboard

![Kaggle Score](results/kaggle_best_score.png)

| Metric | Score |
|---|---|
| **Private score (WRMSSE)** | **0.59212** |
| Public score (WRMSSE) | 0.62949 |

### Feature Importance (Final Model)

![Feature Importance](results/images/feature_importance.png)

The top features by information gain are all time-series history signals — 7-day rolling mean (`rmean_7`), item identity (`item_id`), and 28-day rolling mean (`rmean_28`) — confirming that recent demand momentum dominates over external signals.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_data_cleaning_and_preparation.ipynb      # Parse, downcast, melt wide→long, merge tables
│   ├── 02_external_data_integration.ipynb          # Weather/external data exploration
│   ├── 03_feature_engineering_experiments.ipynb    # Systematic FE experiments (E0–E10, combinations)
│   ├── 04a_model_exploration_rf_ridge.ipynb        # Random Forest & Ridge Regression
│   ├── 04b_model_exploration_gru.ipynb             # GRU (RNN) model
│   ├── 04c_lgbm_hyperparameter_tuning.ipynb        # 2-phase grid search + Optuna tuning
│   ├── 05_final_forecast_submission.ipynb          # 3-seed ensemble, recursive 28-day forecast
│   └── experiments/
│       ├── exp01_baseline_lgbm_pipeline.ipynb      # Initial LightGBM baseline on cleaned data
│       ├── exp02_feature_experiments_E1_E4.ipynb   # Individual feature group tests E1–E4
│       └── exp03_feature_experiments_E5_E9.ipynb   # Individual feature group tests E5–E9
│
├── results/
│   ├── model_performance_summary.csv               # All models with RMSE, MAE
│   ├── experiment_results_summary.csv              # All FE experiments vs. baseline
│   ├── feature_engineering_table.csv               # Feature descriptions and leakage-safety notes
│   ├── final_model_feature_importance.csv          # LightGBM gain-based importance (final model)
│   ├── best_lgbm_params.json                       # Optuna best hyperparameters (trial 37)
│   ├── kaggle_best_score.png                       # Kaggle leaderboard screenshot
│   └── images/
│       ├── model_comparison.png
│       ├── feature_importance.png
│       └── fe_experiments.png
│
├── docs/
│   ├── presentation.pdf                            # Team slide deck
│   └── project_report.pdf                         # Full write-up with methodology tables & figures
│
├── data/
│   └── README.md                                   # Kaggle download instructions & expected layout
│
├── requirements.txt
└── .gitignore
```

---

## Methodology

### 1. Data Cleaning & Preparation (`01`)

- **Type parsing & downcasting** — reduces memory footprint by ~440 MB before the melt step
- **Leading-zero detection** — items not yet on shelf are flagged as inactive to avoid leaking zero demand into training
- **Wide → long melt** — converts 1,941 daily columns into a single `demand` column; merges `calendar` and `sell_prices`

### 2. External Data Integration (`02`)

Weather observations (max/min temperature, z-scores) and additional SNAP/event indicators were explored. Weather features produced no meaningful RMSE improvement (< 0.0001 vs. baseline) and are **not part of the final model**. The code is retained for reproducibility.

### 3. Feature Engineering (`03`)

Starting from a **14-feature baseline (E0)**, 16 controlled experiments (E0–E10 plus named combinations) were run to measure the incremental RMSE impact of each feature group.

**Baseline features (E0) — 14 total:**

| Group | Features |
|---|---|
| Categorical | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` |
| Calendar | `dow`, `month` |
| Price | `sell_price`, `price_change`, `is_promo` |
| Lag & Rolling | `lag_7`, `lag_28`, `rmean_7`, `rmean_28` |

**Experiment results:**

| Experiment | Added Features | RMSE | vs Baseline |
|---|---|---|---|
| E0 — Baseline | — | 2.1619 | — |
| E7 — Event & SNAP | `snap_flag`, `is_event` | 2.1421 | −0.0198 |
| E1 — Demand lag | `lag_56` | 2.1578 | −0.0041 |
| **E7+E1 — Combination ★** | **`snap_flag`, `is_event`, `lag_56`** | **2.1322** | **−0.0297** |
| E9 — All E5~E8 | All price/promo/event/hierarchy | 2.1364 | −0.0255 |
| E2 — Demand lag | `lag_365` | 2.1864 | +0.0245 |

![Feature Engineering Experiments](results/images/fe_experiments.png)

Key leakage-safety principle: all lag/rolling features use a minimum shift equal to or beyond the 28-day forecast horizon. `lag_56` is the shortest safe additional lag.

**Features tested but not selected:**
- `lag_365` — too many NaNs at the start of series (RMSE worsened)
- Price enhancement features (`price_vs_mean`, `price_vs_dept_avg`) — marginal gain, added noise
- Weather features — no meaningful improvement over baseline (RMSE delta < 0.0001)

### 4. Model Exploration (`04a`, `04b`, `04c`)

RF, Ridge, and GRU models were trained on the 17-feature E7+E1 feature set. The final LightGBM model includes `is_active` as an additional indicator, bringing the total to 18 features.

**LightGBM Hyperparameter Tuning** used a two-phase strategy:

- **Phase 1 — Grid search** (12 configurations): identifies which parameter directions help. Best: `set_06_tweedie_low` (RMSE 2.0065), lowering `tweedie_variance_power` from 1.5 → 1.2
- **Phase 2 — Focused Optuna search** (40 trials): zooms into the Phase 1 winner region. Best: trial 37 (RMSE **2.0025**)

Best hyperparameters ([`results/best_lgbm_params.json`](results/best_lgbm_params.json)):

```json
{
  "objective": "tweedie",
  "boosting_type": "gbdt",
  "learning_rate": 0.0419,
  "num_leaves": 91,
  "min_data_in_leaf": 85,
  "tweedie_variance_power": 1.277,
  "lambda_l1": 0.868,
  "lambda_l2": 0.360,
  "feature_fraction": 0.699,
  "bagging_fraction": 0.869,
  "best_iteration": 152
}
```

### 5. Final 28-Day Recursive Forecast (`05`)

The final submission uses **recursive forecasting** with a **3-seed LightGBM ensemble** (seeds 42, 123, 2024 — same hyperparameters, predictions averaged). Each of the 28 days is predicted in order, with predicted values appended to the history so they can serve as lag/rolling inputs for subsequent days.

- Forecast period: days 1942–1969 (the Kaggle evaluation window)
- Total predictions: 853,720 (30,490 series × 28 days)
- Submission format: 60,980 rows (30,490 validation + 30,490 evaluation IDs × 28 columns F1–F28)

### Experiments (`experiments/`)

Three exploratory notebooks that preceded the consolidated pipeline:

- **`exp01_baseline_lgbm_pipeline`** — initial streamlined LightGBM baseline on the cleaned long-format dataset; establishes the reference RMSE before feature experimentation
- **`exp02_feature_experiments_E1_E4`** — individual ablation tests for feature groups E1–E4 (demand lags, rolling stats, price features)
- **`exp03_feature_experiments_E5_E9`** — individual ablation tests for feature groups E5–E9 (promo indicators, event/SNAP flags, hierarchy aggregates)

Results from these notebooks were consolidated into `03_feature_engineering_experiments.ipynb`.

---

## Dataset

**Source:** [Kaggle M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

| File | Description | Size |
|---|---|---|
| `sales_train_evaluation.csv` | Daily unit sales in wide format — 3,049 items × 1,941 days | ~120 MB |
| `sell_prices.csv` | Weekly item sell price per store | ~200 MB |
| `calendar.csv` | Date metadata — day-of-week, SNAP flags, event names | ~100 KB |

Raw data is not included in this repository due to file size. See [`data/README.md`](data/README.md) for download instructions.

Key data facts after cleaning:
- **30,490 time series** (3,049 items × 10 stores), 1,941 days of history
- Memory reduced by ~440 MB via downcasting before the melt step

---

## How to Run

### Environment Setup

```bash
pip install -r requirements.txt
```

### Data Download

```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

### Configuration

Each notebook has a configuration cell near the top where you set the data path for your environment:

```python
from pathlib import Path

DATA_DIR    = Path("../data/raw")                              # Kaggle raw CSVs
OUTPUT_DIR  = Path("./cleaned_data")                           # output of notebook 01
DATA_PATH   = Path("../results/long_df_with_features.parquet") # output of notebook 03
```

> **Google Colab users:** Uncomment the `drive.mount` cell at the top of each notebook and update the path variables to your Google Drive locations.

### Recommended Execution Order

```
01 → 02 (optional) → 03 → 04c → 05
```

Notebooks `04a` and `04b` can be run independently after `03` and are not required to reproduce the final submission. The `experiments/` notebooks are standalone and document the pre-consolidation ablation runs.

---

## Project Highlights

- **Systematic feature ablation** — 16 controlled experiments with a consistent train/validation split, each reporting RMSE, rather than ad-hoc feature addition
- **Leakage-safe feature design** — all lag and rolling features explicitly account for the 28-day forecast horizon, with documented shift values
- **Two-phase hyperparameter search** — grid search for direction, Optuna for precision, with full logging of all 40+ trials
- **Seed ensemble** — final submission averages predictions across 3 random seeds to reduce variance
- **Recursive forecasting implementation** — correctly handles the M5 evaluation setup where future demand is unknown at inference time
- **Multi-model comparison** — four model families (tree ensemble, linear, RNN, gradient boosting) evaluated on consistent data splits

---

## Tools & Technologies

| Layer | Technology |
|---|---|
| Data processing | Pandas · NumPy · PyArrow |
| Modeling | LightGBM · scikit-learn · TensorFlow/Keras |
| Hyperparameter tuning | Optuna |
| Utilities | tqdm · matplotlib |

---

## Acknowledgments

- [Kaggle M5 Forecasting — Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) for the dataset and evaluation framework
- [LightGBM](https://lightgbm.readthedocs.io), [Optuna](https://optuna.org), [TensorFlow/Keras](https://www.tensorflow.org), and [scikit-learn](https://scikit-learn.org) teams for open-source tooling

---

## Usage and License Note

This repository is shared for academic and portfolio purposes. Please contact the author before reusing or redistributing the code.
