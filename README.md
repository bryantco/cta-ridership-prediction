# CTA Ridership Prediction

A machine-learning pipeline that predicts daily ridership at Chicago Transit Authority (CTA) 'L' train stations using publicly available data from the City of Chicago's open data portal.

---

## Data sources

| Dataset | Socrata ID | Description |
|---------|-----------|-------------|
| CTA - Ridership - 'L' Station Entries - Daily Totals | `5neh-572f` | Daily ridership counts per station since 2001 |
| CTA - System Information - List of 'L' Stops | `8pix-ypme` | Station metadata including line, location, and coordinates |

---

## Pipeline

```
extract_ridership_data/   ← downloads & caches raw data from Socrata
feature_engineer/         ← joins station metadata, engineers date & line features
predict_ridership/        ← trains and evaluates regression models
```

### 1. Data extraction (`extract_ridership_data/extract_ridership_data.ipynb`)

- Downloads the ridership and station datasets from the Chicago open data portal via the Socrata API.
- Saves results as Parquet files for efficient re-use.
- Incrementally updates the local cache when new records are available.
- Shared fetching logic lives in `extract_ridership_data/utils.py`.

### 2. Feature engineering (`feature_engineer/feature_engineer.ipynb`)

- Merges station metadata (line colour, latitude, longitude) onto ridership records.
- Derives a categorical `line` feature from the per-line boolean columns.
- Extracts date components: `year`, `month`, `day`, `day_of_week_num`, `day_of_week_name`.
- Drops rows for permanently closed stations that are absent from the station metadata.

### 3. Modelling (`predict_ridership/predict_ridership.ipynb`)

**Pre-processing**

- Categorical features (line, day name) are one-hot encoded.
- Numeric features are standardised with `StandardScaler`.
- The target (`rides`) is log-transformed via `np.log1p` to reduce skew.

**Models evaluated**

| Model | Test R² |
|-------|---------|
| OLS (linear regression) | 0.131 |
| Neural network (PyTorch, 2-layer MLP) | 0.378 |
| LightGBM | 0.885 |
| XGBoost | 0.938 |
| Tuned XGBoost (Optuna, 20 trials) | 0.941 |
| Tuned LightGBM (Optuna, 10 trials) | 0.914 |
| **Random Forest** | **0.968** |

The **Random Forest** (10 estimators, no tuning) achieved the highest R² of **0.968** on the held-out test set (33 % of data, random seed 42).

Shared objective functions for Optuna hyperparameter tuning are in `predict_ridership/utils.py`.  
The PyTorch model architecture is in `predict_ridership/model.py`.

---

## Project structure

```
cta-ridership-prediction/
├── extract_ridership_data/
│   ├── extract_ridership_data.ipynb   # data download notebook
│   └── utils.py                       # Socrata fetch helper
├── feature_engineer/
│   └── feature_engineer.ipynb         # feature engineering notebook
├── predict_ridership/
│   ├── predict_ridership.ipynb        # modelling notebook
│   ├── model.py                       # PyTorch Regressor class
│   └── utils.py                       # Optuna objective functions & scorer
├── pyproject.toml
└── README.md
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Launch Jupyter to run notebooks
uv run jupyter notebook
```

Run the notebooks **in order**:

1. `extract_ridership_data/extract_ridership_data.ipynb`
2. `feature_engineer/feature_engineer.ipynb`
3. `predict_ridership/predict_ridership.ipynb`
