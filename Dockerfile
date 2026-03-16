FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 복사 후 설치 (레이어 캐시 활용 - requirements.txt가 안 바뀌면 재설치 안 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY . .

# Streamlit 기본 포트
EXPOSE 8102

# 컨테이너 실행 시 Streamlit 앱 구동
# --server.port: 포트 지정
# --server.address: 외부 접속 허용 (0.0.0.0)
# --server.headless: 브라우저 자동 실행 비활성화 (서버 환경)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8102", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
