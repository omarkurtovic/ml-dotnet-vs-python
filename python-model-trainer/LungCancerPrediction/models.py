
from pydantic import BaseModel
from enum import IntEnum

class LCValidationScoreDto(BaseModel):
    benignProbability: float = 0.0
    malignantProbability: float = 0.0
    normalProbability: float = 0.0
    trueLabel: float = 0.0
    
class ModelLanguageDto(IntEnum):
    CSharp = 0
    Python = 1

class LCTrainingStatusDto(IntEnum):
    Training = 0
    Trained = 1
    Failed = 2

class LCTrainingProgressDto(BaseModel):
    trainingStatus: LCTrainingStatusDto = LCTrainingStatusDto.Training
    trainingTimeInSeconds: float = 0.0
    modelId: int = 0
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
    validationScores: list[LCValidationScoreDto] = []

class LCTrainingParamsDto(BaseModel):
    name: str = ""
    language: ModelLanguageDto = ModelLanguageDto.Python
    epochs: int = 0
    withAugmentation: bool = False


class SegmentEpochData(BaseModel):
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
    validationScores: list[LCValidationScoreDto] = []

class LCInferenceResultDto(BaseModel):
    benignScore: float = 0.0
    malignantScore: float = 0.0
    normalScore: float = 0.0
    predictionTimeInSeconds: float = 0.0
