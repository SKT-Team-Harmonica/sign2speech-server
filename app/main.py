from fastapi import FastAPI
from app.routers import model_serving

app = FastAPI()

app.include_router(model_serving.router)
