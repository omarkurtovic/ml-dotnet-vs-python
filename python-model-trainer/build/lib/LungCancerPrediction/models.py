
from pydantic import BaseModel
from enum import IntEnum

class ModelLanguageDto(IntEnum):
    CSharp = 0
    Python = 1

class LCEpochDataDto(BaseModel):
    epoch: int = 0
    trainingLoss: float = 0.0
    trainingAccuracy: float = 0.0
    validationAccuracy: float = 0.0
    validationLoss: float = 0.0
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

class LCDto(BaseModel):
    name: str = ""
    language: ModelLanguageDto = ModelLanguageDto.Python
    epochData: list[LCEpochDataDto] = []
    trainingTimeInSeconds: float = 0.0
    validationTimeInSeconds: float = 0.0
    dataLoadingTimeInSeconds: float = 0.0
    hardwareInfo: str = ""


class LCTrainingParamsDto(BaseModel):
    name: str = ""
    language: ModelLanguageDto = ModelLanguageDto.Python
    epochs: int = 0
    withFlips: bool = False


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

class LCPredictionDto(BaseModel):
    benignScore: float = 0.0
    malignantScore: float = 0.0
    normalScore: float = 0.0
    predictionTimeInSeconds: float = 0.0