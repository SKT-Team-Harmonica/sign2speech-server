from fastapi import FastAPI
from app.routers import model_serving,chatgpt_router

app = FastAPI()

app.include_router(model_serving.router)
app.include_router(chatgpt_router.router)
