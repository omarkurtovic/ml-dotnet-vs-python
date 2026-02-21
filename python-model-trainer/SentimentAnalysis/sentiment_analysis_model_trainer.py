
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.pipeline import Pipeline

repo_root = Path("..")
dataPath = repo_root.joinpath('data/sentiment-analysis/IMDB Dataset.csv')
data_df = pd.read_csv(dataPath)


mapping = { 'positive': True, 'negative': False }
data_df['sentiment'] = data_df['sentiment'].map(mapping)


pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3))
    ])

X, y = data_df['review'], data_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipe.fit(X_train, y_train)


# Save Model
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

model_dir = repo_root / "models" / "sentiment-analysis" / "python"
model_dir.mkdir(parents=True, exist_ok=True)

onx = to_onnx(pipe, initial_types=[('input', StringTensorType([None, 1]))])
model_path = model_dir / "python_rf_sentiment_analysis.onnx"
with open(model_path, "wb") as f:
    f.write(onx.SerializeToString())



# Model Evaluation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

train_pred = pipe.predict(X_train)
test_pred = pipe.predict(X_test)

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



