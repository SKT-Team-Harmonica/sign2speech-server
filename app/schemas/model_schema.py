from fastapi import UploadFile
from pydantic import BaseModel

class InputData(BaseModel):
    video: UploadFile
class PredictionResult(BaseModel):
    prediction: str