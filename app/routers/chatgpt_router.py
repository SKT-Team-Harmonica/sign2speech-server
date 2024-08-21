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

class InputData(BaseModel):
    strings: list[str]

@router.post("/generate-sentence/")
async def generate_sentence(input_data: InputData):
    try:
        # 리스트의 문자열을 하나의 문자열로 결합
        combined_string = " ".join(input_data.strings)

        # ChatGPT에게 전달할 프롬프트
        prompt = f"각 글로스들을 줄테니, 완성된 하나의 문장으로 만들어줘. 다른 부가적인 설명 문장 없이 한글 존댓말로: {combined_string}"

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

