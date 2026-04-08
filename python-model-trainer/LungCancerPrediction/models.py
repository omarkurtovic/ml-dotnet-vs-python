
from pydantic import BaseModel
from enum import IntEnum

class ModelLanguage(IntEnum):
    CSharp = 0
    Python = 1

class LungCancerModelEpochData(BaseModel):
    epoch: int
    trainingLoss: float
    trainingAccuracy: float
    validationAccuracy: float
    validationLoss: float
    beningPrecision: float
    beningRecall: float
    beningF1Score: float
    malignantPrecision: float
    malignantRecall: float
    malignantF1Score: float
    normalPrecision: float  
    normalRecall: float
    normalF1Score: float
    macroPrecision: float
    macroRecall: float
    macroF1Score: float
    weightedPrecision: float
    weightedRecall: float
    weightedF1Score: float

class LungCancerModel(BaseModel):
    modelName: str
    modelLanguage: ModelLanguage
    epochData: list[LungCancerModelEpochData] = []
    trainingTimeInSeconds: int


class LungCancerTrainingParams(BaseModel):
    modelName: str
    ModelLanguage: ModelLanguage
    epochs: int
    withFlips: bool
    hardwareInfo: str