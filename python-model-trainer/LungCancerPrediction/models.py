
from pydantic import BaseModel
from enum import IntEnum

class ModelLanguageDto(IntEnum):
    CSharp = 0
    Python = 1

class LCEpochDataDto(BaseModel):
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

class LCDto(BaseModel):
    name: str
    language: ModelLanguageDto
    epochData: list[LCEpochDataDto] = []
    trainingTimeInSeconds: int
    hardwareInfo: str


class LCTrainingParamsDto(BaseModel):
    name: str
    language: ModelLanguageDto
    epochs: int
    withFlips: bool


class EpochData(BaseModel):
    accuracy: float = 0.0
    loss: float = 0.0
    benignPrecision: float = 0.0
    benignRecall: float = 0.0
    benignF1Score: float = 0.0
    malignantPrecision: float = 0.0
    malignantRecall: float = 0.0
    malignantF1Score: float = 0.0
    normalPrecision: float = 0.0
    normalRecall: float = 0.0
    normalF1Score: float = 0.0
    macroPrecision: float = 0.0
    macroRecall: float = 0.0
    macroF1Score: float = 0.0
    weightedPrecision: float = 0.0
    weightedRecall: float = 0.0
    weightedF1Score: float = 0.0