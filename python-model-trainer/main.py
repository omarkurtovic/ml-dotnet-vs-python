import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

train_df['Mileage'] = train_df['Mileage'].str.replace(' km', '').astype(int)
test_df['Mileage'] = test_df['Mileage'].str.replace(' km', '').astype(int)


feature_cols = [col for col in train_df.columns if col not in ['ID', 'Price', 'Levy']]

# za provjeru kolona
#train_df = pd.get_dummies(train_df[feature_cols])
#print(train_df.columns.tolist())
# y_train = train_df['Price']

X_train_df = pd.get_dummies(train_df[feature_cols])

X_train = X_train_df.to_numpy()
y_train = train_df['Price'].to_numpy()

print(f"Training set shape: {X_train.shape}")
print(f"Features: {X_train.shape[1]} columns")
print(f"Columns: {X_train.dtype.names}")

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)

clf.score(X_train,y_train)
clf.coef_
clf.intercept_


X_test_df = pd.get_dummies(test_df[feature_cols])
X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

prediction = clf.predict(X_test_df.iloc[0:1])
print(f"First test prediction: {prediction[0]}")