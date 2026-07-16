from fastapi import FastAPI
from LungCancerPrediction.lc_controller import router as lc_router
from LungCancerPrediction.services import PathResolver

app = FastAPI()


@app.on_event("startup")
async def initialize_storage():
    PathResolver.initialize_storage()

app.include_router(lc_router)
