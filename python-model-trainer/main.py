import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data cleanup
data_df = pd.read_csv('../data/train.csv')
data_df['Mileage'] = data_df['Mileage'].str.replace(' km', '').astype(int)

# Remove outliers
print(f"Before filtering: {len(data_df)} rows")
data_df = data_df[(data_df['Price'] >= 100) & (data_df['Price'] <= 200000)]
print(f"After filtering: {len(data_df)} rows")

# one hot encoding
columns = [col for col in data_df.columns if col not in ['ID', 'Levy']]
data_encoded_df = pd.get_dummies(data_df[columns])

# shuffle
data_encoded_df = data_encoded_df.sample(frac=1).reset_index(drop=True)

feature_cols = [col for col in data_encoded_df.columns if col not in ['Price']]
X = data_encoded_df[feature_cols].to_numpy()
y = data_encoded_df['Price'].to_numpy()

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


# scale the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# scaler = StandardScaler()
# y_train = scaler.fit_transform(y_train)
# y_val = scaler.transform(y_val)
# y_test = scaler.transform(y_test)
for n in [50, 100, 200, 300, 500]:

    regr = RandomForestRegressor(
            n_estimators=n,
            max_depth=25     ,
            random_state=1)
   
    regr.fit(X_train, y_train)

    train_score = regr.score(X_train, y_train)
    test_score = regr.score(X_test, y_test)
    val_score = regr.score(X_val, y_val)

    print(f"Num Estimators: {n}")
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    print(f"Val R²: {val_score:.4f}")


# for alpha in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
#     regr = RandomForestRegressor(
#         n_estimators=100,
#         max_depth=15,
#         random_state=1)
   
#     regr.fit(X_train, y_train)

#     train_score = regr.score(X_train, y_train)
#     test_score = regr.score(X_test, y_test)
#     val_score = regr.score(X_val, y_val)
#     print(f"Alpha: {alpha:.2f}")
#     print(f"Train R²: {train_score:.4f}")
#     print(f"Test R²: {test_score:.4f}")
#     print(f"Val R²: {val_score:.4f}")

#     alpha += 0.01


