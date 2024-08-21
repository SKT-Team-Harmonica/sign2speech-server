# 베이스 이미지로 공식 Python 3.9-slim 이미지 사용
FROM python:3.9-slim

# 기본 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 작업 디렉토리 설정
WORKDIR /app

# 로컬의 requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt .

# 필요한 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 모든 파일을 작업 디렉토리로 복사
COPY . .

# 환경 변수 설정 (출력 버퍼링 비활성화)
ENV PYTHONUNBUFFERED=1

# 8000번 포트 외부 노출
EXPOSE 8000

# FastAPI 애플리케이션 실행 (포트 8000)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
