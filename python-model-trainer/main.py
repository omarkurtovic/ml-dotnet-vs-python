from fastapi import FastAPI
from LungCancerPrediction.lc_controller import router as lc_router

app = FastAPI()

app.include_router(lc_router)
