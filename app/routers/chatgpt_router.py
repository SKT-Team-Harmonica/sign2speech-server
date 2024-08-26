from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수를 로드
load_dotenv()

# OpenAI API Key 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# APIRouter 객체 생성
router = APIRouter()

app = FastAPI()

# 단일 문자열 입력을 위한 데이터 모델
class InputData(BaseModel):
    string: str

@router.post("/translate")
async def generate_sentence(input_data: InputData):
    try:
        combined_string = input_data.string

        # ChatGPT에게 전달할 프롬프트
        prompt = f"언어를 감지해서 한국어 문장으로 만들어줘. 다른 부가 설명 필요없이 번역된 문장만 출력.: {combined_string}"

        # ChatGPT API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        # 생성된 텍스트 추출
        generated_text = response.choices[0].message['content'].strip()

        return {"generated_sentence": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
