
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

repo_root = Path("..")

print("=== Sentiment Analysis Model Trainer ===")
print("=== Language: Python ===")

# load data
print("Loading data...")
dataPath = repo_root.joinpath('data/sentiment-analysis/IMDB Dataset.csv')
data_df = pd.read_csv(dataPath)
print("Sample data:")
print(data_df.head())
print(f"Number of rows: {len(data_df)}")


# data cleanup
mapping = { 'positive': True, 'negative': False }
data_df['sentiment'] = data_df['sentiment'].map(mapping)
print(data_df.head());

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data_df['review'])
y = data_df['sentiment']


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# train model
print("Training model...")
model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)


# Save Model
from skl2onnx import to_onnx

model_dir = repo_root / "models" / "sentiment-analysis" / "python"
model_dir.mkdir(parents=True, exist_ok=True)


onx = to_onnx(model, np.zeros((1, X.shape[1]), dtype=np.float32))
model_path = model_dir / "python_rf_sentiment_analysis.onnx"
with open(model_path, "wb") as f:
    f.write(onx.SerializeToString())



# Model Evaluation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("┌─── TRAINING SET METRICS ───┐")
print(f"  Accuracy:  {accuracy_score(y_train, train_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_train, train_pred):.4f}")
print(f"  AUC:       {roc_auc_score(y_train, train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, train_pred):.4f}")
print(f"  Recall:    {recall_score(y_train, train_pred):.4f}")
print("└─────────────────────────────┘")
print()
print("┌─── TEST SET METRICS ───┐")
print(f"  Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_test, test_pred):.4f}")
print(f"  AUC:       {roc_auc_score(y_test, test_pred):.4f}")
print(f"  Precision: {precision_score(y_test, test_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, test_pred):.4f}")
print("└─────────────────────────┘")



