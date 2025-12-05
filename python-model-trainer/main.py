import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# cleanup
data_df = pd.read_csv('../data/train.csv')
data_df['Mileage'] = data_df['Mileage'].str.replace(' km', '').astype(int)
columns = [col for col in data_df.columns if col not in ['ID', 'Levy']]
data_encoded_df = pd.get_dummies(data_df[columns])

# shuffle
data_encoded_df = data_encoded_df.sample(frac=1).reset_index(drop=True)

# split data
feature_cols = [col for col in data_encoded_df.columns if col not in ['Price']]
X = data_encoded_df[feature_cols].to_numpy()
y = data_encoded_df['Price'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

clf = Ridge(alpha=1.0)
clf.fit(X_train_scaled, y_train)


train_score = clf.score(X_train_scaled, y_train)
val_score = clf.score(X_val_scaled, y_val)
test_score = clf.score(X_test_scaled, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Val R²: {val_score:.4f}")
print(f"Test R²: {test_score:.4f}")

