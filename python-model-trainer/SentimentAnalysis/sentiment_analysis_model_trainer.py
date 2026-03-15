from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from enum import IntEnum
from pathlib import Path
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
import pandas as pd

router = APIRouter()

repo_root = Path("..")
data_path = repo_root.joinpath('data/sentiment-analysis/IMDB Dataset.csv')

class TrainerAlgorithm(IntEnum):
    SdcaLogisticRegression  = 0
    LbfgsLogisticRegression = 1
    AveragedPerceptron      = 2
    LinearSvm               = 3
    FastTree                = 4
    FastForest              = 5


class ModelLanguage(IntEnum):
    CSharp = 0
    Python = 1


class TrainData(BaseModel):
    modelName: str
    algorithm: TrainerAlgorithm
    modelLanguage: ModelLanguage


def build_pipeline(algorithm: TrainerAlgorithm) -> Pipeline:
    classifiers = {
        TrainerAlgorithm.SdcaLogisticRegression:  SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, tol=1e-3),
        TrainerAlgorithm.LbfgsLogisticRegression: LogisticRegression(solver="lbfgs", max_iter=1000),
        TrainerAlgorithm.AveragedPerceptron:       CalibratedClassifierCV(Perceptron(max_iter=1000)),
        TrainerAlgorithm.LinearSvm:                CalibratedClassifierCV(LinearSVC(max_iter=1000)),
        TrainerAlgorithm.FastTree:                 GradientBoostingClassifier(n_estimators=100),
        TrainerAlgorithm.FastForest:               RandomForestClassifier(n_estimators=100),
    }
    return Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", classifiers[algorithm]),
    ])

@router.post("/Python/SentimentAnalysis/Train")
def train(train_data: TrainData):
    if train_data.modelLanguage != ModelLanguage.Python:
        raise HTTPException(status_code=400, detail="Only Python models are supported here.")

    if not data_path.exists():
        raise HTTPException(status_code=405, detail="Dataset not found.")

    df = pd.read_csv(data_path)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=1)

    pipe = build_pipeline(train_data.algorithm)
    pipe.fit(X_train, y_train)

    model_dir = repo_root / "models" / "sentiment-analysis" / "python"
    model_dir.mkdir(parents=True, exist_ok=True)
    onx = to_onnx(pipe, initial_types=[("input", StringTensorType([None, 1]))])
    with open(model_dir / f"{train_data.modelName}.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    train_pred = pipe.predict(X_train)
    test_pred  = pipe.predict(X_test)

    return {
        "name":                      train_data.modelName,
        "language":                  ModelLanguage.Python,
        "trainerAlgorithm":          train_data.algorithm,
        "trainingAccuracy":          accuracy_score(y_train, train_pred),
        "trainingF1Score":           f1_score(y_train, train_pred),
        "trainingAreaUnderRocCurve": roc_auc_score(y_train, train_pred),
        "trainingPositivePrecision": precision_score(y_train, train_pred),
        "trainingPositiveRecall":    recall_score(y_train, train_pred),
        "testingAccuracy":           accuracy_score(y_test, test_pred),
        "testingF1Score":            f1_score(y_test, test_pred),
        "testingAreaUnderRocCurve":  roc_auc_score(y_test, test_pred),
        "testingPositivePrecision":  precision_score(y_test, test_pred),
        "testingPositiveRecall":     recall_score(y_test, test_pred),
    }


