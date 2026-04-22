import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import optuna

import xgboost as xgb
import lightgbm as lgb

from model import Regressor
from utils import xgb_objective, lgb_objective
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from joblib import dump

cta_df = pd.read_parquet('../feature_engineer/output/cta_ridership_with_features.parquet')
cta_df = cta_df.reset_index(drop=True)

cta_df.head(10)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
onehot_encoder = OneHotEncoder()
scaler = StandardScaler()

preprocessor = make_column_transformer(
    # (ordinal_encoder, make_column_selector(dtype_include=object)),
    (onehot_encoder, make_column_selector(dtype_include=object)),
    (scaler, make_column_selector(dtype_include='number')),
    remainder='passthrough'
)

X = preprocessor.fit_transform(cta_df[['line', 'year', 'month', 'day', 'day_of_week_num', 'day_of_week_name', 'lat', 'lon']])
y = cta_df['rides']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

import scipy.sparse as sp

feature_names = preprocessor.get_feature_names_out()
X_test_dense = X_test.toarray() if sp.issparse(X_test) else X_test

test_df = pd.DataFrame(X_test_dense, columns=feature_names)
test_df["rides"] = y_test.values

test_df.to_parquet("output/cta_ridership_test.parquet", index=False)

test_df.size

mod_ols = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1
)

mod_ols.fit(X_train, y_train)
y_pred_ols = mod_ols.predict(X_test)
print("R squared: ", r2_score(y_test, y_pred_ols))

mod_rf = TransformedTargetRegressor(
    regressor=RandomForestRegressor(n_estimators=10, random_state=42),
    func=np.log1p,
    inverse_func=np.expm1
)

mod_rf.fit(X_train, y_train)
y_pred_rf = mod_rf.predict(X_test)
print("R squared: ", r2_score(y_test, y_pred_rf))

mod_xgb = TransformedTargetRegressor(
    regressor=xgb.XGBRegressor(random_state=42, tree_method='hist'),
    func=np.log1p,
    inverse_func=np.expm1
)

mod_xgb.fit(X_train, y_train)
y_pred_xgb = mod_xgb.predict(X_test)
print("R squared: ", r2_score(y_test, y_pred_xgb))

torch.manual_seed(42)

# Preprocess ----
# Log transform
y_train_log = np.log1p(y_train).values.reshape(-1, 1)
y_test_log = np.log1p(y_test).values.reshape(-1, 1)

y_train_scaled = scaler.fit_transform(y_train_log)
y_test_scaled = scaler.transform(y_test_log)

# Initialize
model = Regressor(n_in=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train_scaled).float()

for epoch in tqdm(range(500)):
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    loss = criterion(preds, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()

with torch.no_grad():
    y_pred_scaled = model(torch.from_numpy(X_test).float()).numpy()
    y_pred = np.expm1(scaler.inverse_transform(y_pred_scaled))
    r2_nn = r2_score(y_test, y_pred)
    print("R squared: ", r2_nn)

mod_lgb = TransformedTargetRegressor(
    regressor=lgb.LGBMRegressor(random_state=42),
    func=np.log1p,
    inverse_func=np.expm1
)

mod_lgb.fit(X_train, y_train)
y_pred_lgb = mod_lgb.predict(X_test)
print("R squared: ", r2_score(y_test, y_pred_lgb))

y_log = np.log1p(y.astype(float))

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda trial: xgb_objective(trial, X, y_log), n_trials=20)

print(f"Best params is {study.best_params} with value {study.best_value}")

# Predict using the best set of hyperparameters
mod_xgb_tuned = TransformedTargetRegressor(
    regressor=xgb.XGBRegressor(**study.best_params),
    func=np.log1p,
    inverse_func=np.expm1
)

mod_xgb_tuned.fit(X_train, y_train)
y_pred_xgb_tuned = mod_xgb_tuned.predict(X_test)
print("R squared: ", r2_score(y_test, y_pred_xgb_tuned))

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lambda trial: lgb_objective(trial, X, y_log), n_trials=10)

print(f"Best params for LGB is {study_lgb.best_params} with value {study_lgb.best_value}")

# Predict using the best set of hyperparameters
mod_lgb_tuned = TransformedTargetRegressor(
    regressor=lgb.LGBMRegressor(**study_lgb.best_params),
    func=np.log1p,
    inverse_func=np.expm1
)

mod_lgb_tuned.fit(X_train, y_train)
y_pred_lgb_tuned = mod_lgb_tuned.predict(X_test)
print("R squared: ", r2_score(y_test, y_pred_lgb_tuned))

dump(mod_rf, 'output/mod_rf.joblib')