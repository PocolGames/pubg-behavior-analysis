# PUBG Player Behavior Analysis Dashboard

딥러닝 개발자를 위한 PUBG 플레이어 행동 분석 및 모델 성능 대시보드

## 🎯 프로젝트 개요

이 프로젝트는 PUBG 게임 데이터를 활용하여 플레이어의 행동 패턴을 분석하고, 8가지 플레이어 유형으로 분류하는 머신러닝 모델의 성능을 시각화하는 대시보드입니다.

### 주요 기능

- **모델 성능 분석**: 99.25% 정확도를 달성한 분류 모델의 성능 메트릭 표시
- **8가지 플레이어 유형 분류**: Survivor, Explorer, Aggressive 등 다양한 플레이어 유형
- **인터랙티브 차트**: Chart.js를 활용한 동적 데이터 시각화
- **실시간 API 연동**: FastAPI 백엔드와 연동된 실시간 데이터 처리

## 🏗️ 프로젝트 구조

```
pubg-behavior-analysis/
├── frontend/                 # 프론트엔드 (HTML, CSS, JS)
│   ├── css/
│   │   ├── dashboard.css    # 메인 대시보드 스타일
│   │   └── components.css   # 컴포넌트 스타일
│   ├── js/
│   │   ├── api.js          # API 통신 관련
│   │   ├── charts.js       # 차트 생성 및 관리
│   │   └── dashboard.js    # 메인 대시보드 로직
│   ├── assets/             # 정적 자원
│   └── index.html          # 메인 대시보드 페이지
├── backend/                 # 백엔드 (FastAPI)
│   ├── fastapi_app.py      # FastAPI 애플리케이션
│   └── run_server.py       # 서버 실행 스크립트
└── README.md               # 프로젝트 문서
```

## 🚀 실행 방법

### 1. 백엔드 서버 실행

```bash
cd backend
python run_server.py
```

또는 직접 실행:

```bash
cd backend
python -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 대시보드 접속

브라우저에서 다음 주소로 접속:

- **대시보드**: http://localhost:8000/static/index.html
- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/health

## 📊 대시보드 구성

### 1. 전체 모델 성능 메트릭
- 모델 정확도: 99.25%
- F1 스코어: 98.67%
- 플레이어 유형: 8개
- 사용된 특성: 30개

### 2. 플레이어 유형별 분류 정확도
- 각 플레이어 유형별 분류 성능을 바차트로 표시

### 3. Confusion Matrix
- 실제 vs 예측 결과를 히트맵으로 시각화

### 4. Feature Importance
- 모델에서 중요하게 사용된 특성들의 중요도 순위

### 5. 플레이어 유형 분포
- 데이터셋에서 각 플레이어 유형의 분포를 도넛차트로 표시

## 🎮 플레이어 유형

1. **Survivor (생존형)**: 치료 아이템을 많이 사용하며 생존에 집중
2. **Explorer (탐험형)**: 맵을 많이 돌아다니며 탐험하는 플레이어
3. **Aggressive (공격형)**: 높은 킬 수와 데미지를 기록하는 공격적인 플레이어

## 🛠️ 기술 스택

### Frontend
- **HTML5**: 시맨틱 마크업
- **CSS3**: 다크 테마, PC 최적화 디자인
- **JavaScript (ES6+)**: 모듈화된 구조
- **Chart.js**: 인터랙티브 차트 라이브러리

### Backend
- **FastAPI**: 고성능 Python 웹 프레임워크
- **Pydantic**: 데이터 검증 및 직렬화
- **Uvicorn**: ASGI 서버
- **NumPy**: 수치 계산

## 📋 API 엔드포인트

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| GET | `/` | API 기본 정보 |
| GET | `/health` | 서버 상태 확인 |
| GET | `/model/info` | 모델 기본 정보 |
| GET | `/model/performance` | 모델 성능 메트릭 |
| POST | `/predict` | 플레이어 유형 예측 |

## 🔧 개발 환경 설정

### 필요한 패키지

```bash
pip install fastapi uvicorn[standard] pydantic numpy
```

### 개발 시 유의사항

- 각 파일은 300줄 이내로 작성
- CSS와 JavaScript는 별도 파일로 분리
- 다크 테마 기본 적용
- PC 환경 최적화 (반응형 제외)

## 📈 성능 지표

- **모델 정확도**: 99.25%
- **F1 스코어 (Macro)**: 98.67%
- **특성 중요도 최고**: has_kills (0.3232)
- **처리 시간**: 평균 < 10ms

## 🎯 향후 개발 계획

1. **실시간 모니터링 강화**
2. **A/B 테스트 결과 시각화**
3. **배치 분석 도구 추가**
4. **모델 성능 히스토리 추적**

## 📝 라이센스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 👥 기여하기

개선사항이나 버그 리포트는 GitHub Issues를 통해 제출해 주세요.

---

**Created by**: PUBG Analytics Team  
**Last Updated**: 2024-12-28
