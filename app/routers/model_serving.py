from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.model_schema import InputData, PredictionResult
from app.services.model_service import predict
import os
import cv2
from typing import List
import torch
from torchvision.io import read_video

router = APIRouter()

# 상대 경로로 img_dir 설정 (애플리케이션 루트 기준)
img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_dir")

def extract_frames_jpg(video_file: UploadFile, output_folder: str = img_dir) -> List[str]:
    # output_folder가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 비디오 파일을 읽어 들입니다.
    video_path = os.path.join(output_folder, video_file.filename)

    with open(video_path, "wb") as f:
        f.write(video_file.file.read())

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 프레임을 .jpg로 저장
        frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frames.append(frame_filename)
        count += 1

    cap.release()
    return frames

def extract_frames_tensor(video_file: UploadFile, output_folder: str = img_dir) -> torch.Tensor:
    # output_folder가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 비디오 파일을 읽어 들입니다.
    video_path = os.path.join(output_folder, video_file.filename)

    with open(video_path, "wb") as f:
        f.write(video_file.file.read())

    # torchvision.io.read_video를 사용하여 비디오를 읽고 프레임을 Tensor로 반환
    video_frames, _, _ = read_video(video_path, pts_unit='sec')

    # 데이터 타입을 명시적으로 변환 (float32)
    video_frames = video_frames.float() / 255.0  # uint8을 float32로 변환하며 [0, 1] 범위로 스케일링

    # 비디오 파일 삭제 (선택적)
    os.remove(video_path)

    return video_frames

@router.post("/predict", response_model=PredictionResult)
def get_prediction(video: UploadFile = File(...)):
    # InputData 객체를 생성
    input_data = InputData(video=video)

    # 예측 수행
    prediction_result = predict(input_data)

    # 예측 결과 반환
    return prediction_result

@router.post("/test-extract-frames/")
async def test_extract_frames(video: UploadFile = File(...)):
    frame_files = []
    try:
        frame_files = extract_frames_jpg(video)

        # 결과 반환
        return {"frame_files": frame_files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 정리: 저장된 프레임 파일 삭제
        for file_path in frame_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        video_path = os.path.join(img_dir, video.filename)
        if os.path.exists(video_path):
            os.remove(video_path)

@router.post("/test-extract-frames-tensor/")
async def test_extract_frames(video: UploadFile = File(...)):
    try:
        # extract_frames 함수에서 img_dir을 사용
        video_tensor = extract_frames_tensor(video)

        # Tensor의 크기를 반환 (예: 프레임 수, 높이, 너비, 채널 수)
        return {"video_tensor_shape": video_tensor.shape}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
