

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from pathlib import Path

repo_root = Path("..")

print("=== Car Price Prediction Model Trainer ===")
print("=== Language: Python ===")

# load data
print("Loading data...")
dataPath = repo_root.joinpath('data/car-prediction/train.csv')
data_df = pd.read_csv(dataPath)
print("Sample data:")
print(data_df.head())
print(f"Number of rows: {len(data_df)}")


# filter data
print("Filtering data...")
print(f"Rows before filtering: {len(data_df)}")
data_df = data_df[(data_df['Price'] >= 100) & (data_df['Price'] <= 200000)]
print(f"Rows after filtering: {len(data_df)}")


# data cleanup
data_df['Mileage'] = data_df['Mileage'].str.replace(' km', '').astype('float32')

# one hot encoding
columns = [col for col in data_df.columns if col not in ['ID', 'Levy']]
data_encoded_df = pd.get_dummies(data_df[columns])

# shuffle
# data_encoded_df = data_encoded_df.sample(frac=1).reset_index(drop=True)

feature_cols = [col for col in data_encoded_df.columns if col not in ['Price']]
X = data_encoded_df[feature_cols].to_numpy(dtype=np.float32)
y = data_encoded_df['Price'].to_numpy(dtype=np.float32)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# train model
print("Training model...")
regr = RandomForestRegressor(
        n_estimators=200, 
        max_depth=25,
        random_state=1)

regr.fit(X_train, y_train)


# Saving Model
from skl2onnx import to_onnx

model_dir = repo_root / "models" / "car-prediction" / "python"
model_dir.mkdir(parents=True, exist_ok=True)

onx = to_onnx(regr, X[:1])
model_path = model_dir / "python_rf_carprice.onnx"
with open(model_path, "wb") as f:
    f.write(onx.SerializeToString())


# Model Evaluation
train_pred = regr.predict(X_train)
test_pred = regr.predict(X_test)

print("┌─── TRAINING SET METRICS ───┐")
print(f"  R2:   {r2_score(y_train, train_pred):.4f}")
print(f"  RMSE: {root_mean_squared_error(y_train, train_pred):.2f}")
print(f"  MAE:  {mean_absolute_error(y_train, train_pred):.2f}")
print("└─────────────────────────────┘")
print()
print("┌─── TEST SET METRICS ───┐")
print(f"  R2:   {r2_score(y_test, test_pred):.4f}")
print(f"  RMSE: {root_mean_squared_error(y_test, test_pred):.2f}")
print(f"  MAE:  {mean_absolute_error(y_test, test_pred):.2f}")
print("└─────────────────────────┘")

