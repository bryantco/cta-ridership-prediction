# CTA Ridership Prediction

This repo contains a machine learning project that attempts to predict daily ridership of Chicago Transit Authority (CTA) 'L' train stations using publicly available data from the City of Chicago's open data portal. I found that a simple out-of-the-box random forest model performed the best (R-squared = 0.968). See the [results](#results) section for more details.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Once `uv` is installed on your machine, run the following to run the code in each notebook:

```bash
# Install dependencies
uv sync

# Launch Jupyter to run notebooks
uv run jupyter notebook
```

The order to run the notebooks is as follows:

1. `extract_ridership_data/extract_ridership_data.ipynb`
2. `feature_engineer/feature_engineer.ipynb`
3. `predict_ridership/predict_ridership.ipynb`

## Data sources

| Dataset | Socrata ID | Description |
|---------|-----------|-------------|
| CTA - Ridership - 'L' Station Entries - Daily Totals | `5neh-572f` | Daily ridership counts per station since 2001 |
| CTA - System Information - List of 'L' Stops | `8pix-ypme` | Station metadata including line, location, and coordinates |

---

## Results

The below table summarizes the R-squared from each model that I trained. A random forest (without any hyperparameter tuning) performed the best.

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

