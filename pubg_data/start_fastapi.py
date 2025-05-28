
import subprocess
import sys
import os

print("🚀 FastAPI 서버 시작...")
print(f"📁 작업 디렉토리: {os.getcwd()}")

# FastAPI 실행
try:
    result = subprocess.run([
        sys.executable, "-m", "uvicorn",
        "fastapi_app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], capture_output=False, text=True)
except KeyboardInterrupt:
    print("\n⏹️ 서버 중지됨")
except Exception as e:
    print(f"❌ 서버 실행 오류: {e}")
