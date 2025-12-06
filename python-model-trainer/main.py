import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# data cleanup
data_df = pd.read_csv('../data/train.csv')
data_df['Mileage'] = data_df['Mileage'].str.replace(' km', '').astype(int)

# filter data
print(f"Before filtering: {len(data_df)} rows")
data_df = data_df[(data_df['Price'] >= 100) & (data_df['Price'] <= 200000)]
print(f"After filtering: {len(data_df)} rows")

# one hot encoding
columns = [col for col in data_df.columns if col not in ['ID', 'Levy']]
data_encoded_df = pd.get_dummies(data_df[columns])

# shuffle
data_encoded_df = data_encoded_df.sample(frac=1).reset_index(drop=True)

feature_cols = [col for col in data_encoded_df.columns if col not in ['Price']]
X = data_encoded_df[feature_cols].to_numpy(dtype=np.float32)
y = data_encoded_df['Price'].to_numpy(dtype=np.float32)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print(f"Data shape: {X.shape}")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"y range: {y.min():.0f} to {y.max():.0f}")
print(f"y median: {np.median(y):.0f}")
print(f"y mean: {np.mean(y):.0f}")
print(f"Prices > $100k: {(y > 100000).sum()}")
print(f"Prices > $500k: {(y > 500000).sum()}")


regr = RandomForestRegressor(
        n_estimators=200,
        max_depth=25     ,
        random_state=1)

regr.fit(X_train, y_train)

train_score = regr.score(X_train, y_train)
test_score = regr.score(X_test, y_test)
val_score = regr.score(X_val, y_val)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
print(f"Val R²: {val_score:.4f}")


from skl2onnx import to_onnx
onx = to_onnx(regr, X[:1])
with open("rf_carprice.onnx", "wb") as f:
    f.write(onx.SerializeToString())



import onnxruntime as rt

sess = rt.InferenceSession("rf_carprice.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]