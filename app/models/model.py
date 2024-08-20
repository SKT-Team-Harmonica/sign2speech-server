# todo 학습 완료된 모델 로드 (확장자에 맞게 수정해야 함.)

import pickle

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model("models/your_model.pkl")
except FileNotFoundError:
    print("Warning: Model file not found. Ensure the path is correct.")
    model = None  # 모델이 없는 상태로 진행 (주의: 실제 예측에서는 오류 발생)
