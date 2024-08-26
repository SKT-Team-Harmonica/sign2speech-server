import torch
from typing import List
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# 모델 로드 함수
def load_model(model_path: str, model_class: nn.Module) -> nn.Module:
    try:
        model = model_class()  # 모델 클래스의 인스턴스를 생성합니다.
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # state_dict를 로드
        model.eval()  # 모델을 평가 모드로 전환
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# 모델 로드
class SignModel(nn.Module):
    # 여기에 SignModel의 정의를 추가해야 합니다.
    pass  # SignModel 클래스를 실제로 정의하세요.

model_class = SignModel  # 로드하려는 모델 클래스를 지정합니다.
model = load_model("/Users/bes/PycharmProjects/sign2speech_fastapi/app/models/data.pth", model_class)

# 인퍼런스 함수
def predict(frame_files: List[str]) -> str:
    if model is None:
        raise ValueError("Model is not loaded.")

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델이 기대하는 입력 크기에 맞춰 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames_tensor = []

    for frame_file in frame_files:
        image = Image.open(frame_file).convert('RGB')
        tensor = transform(image)
        frames_tensor.append(tensor)

    # 배치 차원 추가 및 텐서화
    frames_tensor = torch.stack(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)

    # 모델 예측 수행
    with torch.no_grad():
        outputs = model(frames_tensor)

    # 예측 결과를 문자열로 변환 (예시로 가장 높은 확률의 클래스를 반환)
    # outputs: (1, T//4, len(vocab))
    outputs = outputs.mean(dim=1)  # (1, len(vocab)) - sequence 차원을 평균 내어 제거
    _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스를 선택
    return str(predicted.item())
