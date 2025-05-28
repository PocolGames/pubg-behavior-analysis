#!/usr/bin/env python3
"""
PUBG Player Behavior Analysis - FastAPI Server Launcher
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """필요한 패키지 설치 확인"""
    required_packages = [
        'fastapi',
        'uvicorn[standard]',
        'pydantic',
        'numpy'
    ]
    
    print("📦 필요한 패키지 확인 중...")
    
    for package in required_packages:
        try:
            __import__(package.split('[')[0])
            print(f"✅ {package} - 설치됨")
        except ImportError:
            print(f"❌ {package} - 설치 필요")
            print(f"📥 {package} 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 설치 완료")

def start_server():
    """FastAPI 서버 시작"""
    print("\n🚀 PUBG Player Classification API 서버 시작...")
    print("📍 서버 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🎮 대시보드: http://localhost:8000/static/index.html")
    print("\n서버를 중지하려면 Ctrl+C를 누르세요.\n")
    
    try:
        # uvicorn으로 FastAPI 서버 실행
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "fastapi_app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n⏹️ 서버가 중지되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🎮 PUBG Player Behavior Analysis API Server")
    print("=" * 60)
    
    # 의존성 확인
    check_dependencies()
    
    # 서버 시작
    start_server()
