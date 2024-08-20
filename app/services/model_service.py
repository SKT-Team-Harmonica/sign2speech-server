import cv2
import os
from typing import List
from fastapi import UploadFile
from pydantic import BaseModel
from app.models.model import model

class InputData(BaseModel):
    video: UploadFile

class PredictionResult(BaseModel):
    gloss: str

def extract_frames(video_file: UploadFile, output_folder: str) -> List[str]:
    # 비디오 파일을 읽어 들입니다.
    video_path = f"{output_folder}/{video_file.filename}"

    with open(video_path, "wb") as f:
        f.write(video_file.file.read())

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 프레임을 .jpg로 저장
        frame_filename = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(frame_filename, frame)
        frames.append(frame_filename)

        count += 1

    cap.release()
    return frames


def predict(input_data: InputData, temp_folder: str = "temp_frames"):
    # 임시 폴더 생성
    os.makedirs(temp_folder, exist_ok=True)

    # MP4 파일을 프레임 단위로 .jpg 파일로 분리
    frame_files = extract_frames(input_data.video, temp_folder)

    # 모델이 기대하는 형식으로 프레임 리스트를 전달하여 예측 수행
    prediction = model.predict(frame_files)

    # 임시 파일 정리 (선택적)
    for file_path in frame_files:
        os.remove(file_path)
    os.remove(f"{temp_folder}/{input_data.video.filename}")

    # 예측 결과 반환
    return PredictionResult(gloss=prediction)
