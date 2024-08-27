# app/routers/model_serving.py
import os
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Tuple
from dotenv import load_dotenv
import openai
from app.schemas.model_schema import PredictionResult
from app.services.model_service import predict
from app.services.transformer import Resize, CenterCrop, ToTensorGen

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# 상대 경로로 img_dir 설정 (애플리케이션 루트 기준)
img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_dir")

def call_chatgpt_api(input_string: str) -> str:
    #combined_string = " ".join(input_string)

    prompt = (f"주어진 단어들을 하나의 완성된 문장으로 만들어줘, "
              f"다른 부가적인 설명 생성하지 말고, 한글 존댓말로: {input_string}")

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )

    generated_text = response.choices[0].message['content'].strip()
    return generated_text

def extract_frames_jpg(video_file: UploadFile, output_folder: str = img_dir) -> List[str]:
    # output_folder가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 비디오 파일을 저장합니다.
    video_path = os.path.join(output_folder, video_file.filename)

    # 업로드된 파일을 읽고 저장합니다.
    with open(video_path, "wb") as f:
        f.write(video_file.file.read())

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    def center_crop(img: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        h, w, _ = img.shape
        crop_h, crop_w = crop_size
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        return img[start_y:start_y + crop_h, start_x:start_x + crop_w]

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 프레임을 256x256 크기로 리사이즈
        resized_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)

        # 256x256 프레임을 중앙에서 224x224로 크롭
        cropped_frame = center_crop(resized_frame, (224, 224))

        # 크롭된 프레임을 .jpg로 저장
        frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_filename, cropped_frame)
        frames.append(frame_filename)
        count += 1

    cap.release()

    os.remove(video_path)

    return frames
@router.post("/predict", response_model=PredictionResult)
def get_prediction(video: UploadFile = File(...)):
    try:
        # 비디오에서 프레임 추출
        frame_files = extract_frames_jpg(video)

        # 모델에 프레임 파일 경로 리스트 전달하여 예측 수행
        prediction_result = predict(frame_files)
        print(prediction_result)

        # .jpg 파일들 정리
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                os.remove(frame_file)

        # 예측 결과를 ChatGPT API에 전달하여 문장 생성
        generated_sentence = call_chatgpt_api([prediction_result])

        return {"prediction": generated_sentence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/test-extract-frames/")
async def test_extract_frames(video: UploadFile = File(...)):
    frame_files = []
    try:
        frame_files = extract_frames_jpg(video)

        # 결과 반환
        return {"frame_files": frame_files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # finally:
    #     # 정리: 저장된 프레임 파일 삭제
    #     for file_path in frame_files:
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #     video_path = os.path.join(img_dir, video.filename)
    #     if os.path.exists(video_path):
    #         os.remove(video_path)

