import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cta_ridership_test", type=str, required=True)
parser.add_argument("--mod_rf", type=str, required=True)
parser.add_argument("--actual_vs_predicted_ridership", type=str, required=True)
args = parser.parse_args()

# Load test data ----
test_df = pd.read_parquet(args.cta_ridership_test)
mod_rf = load(args.mod_rf)

y_test = test_df["rides"]
X_test = test_df.drop(columns=["rides"]).values

y_pred_rf = mod_rf.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot ----
hb = ax.hexbin(y_test, y_pred_rf, gridsize=50, cmap="Blues", extent=(0, 1e4, 0, 1e4))
fig.colorbar(hb, ax=ax, label="Count")
ax.set_xlabel("Actual Ridership")
ax.set_ylabel("Predicted Ridership")
ax.set_title("Actual vs. Predicted Ridership (Test Set)")

plt.tight_layout()
plt.savefig(args.actual_vs_predicted_ridership)

