# Python 3.12 베이스 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 빌드 인수 정의
ARG OPENAI_API_KEY

# 환경 변수 설정
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# requirements.txt 복사 및 종속성 설치
COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# FastAPI 서버 실행
CMD ["uvicorn", "api.endpoint.mediation_service:app", "--host", "0.0.0.0", "--port", "8000"]