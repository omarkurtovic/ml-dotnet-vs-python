from fastapi import FastAPI
from LungCancerPrediction.lung_cancer_prediction_model_trainer import router as lung_cancer_router

app = FastAPI()

app.include_router(lung_cancer_router)
