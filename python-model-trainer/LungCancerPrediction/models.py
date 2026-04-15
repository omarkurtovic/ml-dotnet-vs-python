
from pydantic import BaseModel
from enum import IntEnum

class ModelLanguage(IntEnum):
    CSharp = 0
    Python = 1

class LungCancerModelEpochData(BaseModel):
    epoch: int
    trainingLoss: float = None
    trainingAccuracy: float = None
    validationAccuracy: float = None
    validationLoss: float = None
    benignPrecision: float = None
    benignRecall: float = None
    benignF1Score: float = None
    malignantPrecision: float = None
    malignantRecall: float = None
    malignantF1Score: float = None
    normalPrecision: float = None
    normalRecall: float = None
    normalF1Score: float = None
    macroPrecision: float = None
    macroRecall: float = None
    macroF1Score: float = None
    weightedPrecision: float = None
    weightedRecall: float = None
    weightedF1Score: float = None

class LungCancerModel(BaseModel):
    name: str
    language: ModelLanguage
    epochData: list[LungCancerModelEpochData] = []
    trainingTimeInSeconds: int
    hardwareInfo: str


class LungCancerTrainingParams(BaseModel):
    modelName: str
    modelLanguage: ModelLanguage
    epochs: int
    withFlips: bool