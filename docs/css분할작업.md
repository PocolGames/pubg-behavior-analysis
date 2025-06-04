## 📋 CSS 분할 작업 목차

### 현재 상태 분석
1. **main.css** (약 290줄) - 기본 스타일, 레이아웃, 컴포넌트
2. **dashboard.css** (약 360줄) - 대시보드 전용 스타일
3. **components.css** - 컴포넌트 스타일
4. **cluster-analysis.css** - 클러스터 분석 페이지 스타일

### 제안하는 새로운 CSS 구조
1. **base/** - 기본 스타일
   - `reset.css` - 리셋 및 기본 설정
   - `typography.css` - 폰트, 텍스트 스타일
   - `variables.css` - CSS 변수 정의

2. **layout/** - 레이아웃 관련
   - `grid.css` - 그리드 시스템
   - `header.css` - 헤더 스타일
   - `footer.css` - 푸터 스타일

3. **components/** - 재사용 컴포넌트
   - `buttons.css` - 버튼 스타일
   - `cards.css` - 카드 컴포넌트
   - `forms.css` - 폼 관련
   - `animations.css` - 애니메이션

4. **pages/** - 페이지별 스타일
   - `dashboard.css` - 대시보드
   - `cluster-analysis.css` - 클러스터 분석
   - `model-performance.css` - 모델 성능

5. **utilities/** - 유틸리티
   - `helpers.css` - 헬퍼 클래스
   - `responsive.css` - 반응형

**dashboard.css 파일부터 분할을 시작하겠습니다. 이 파일이 가장 크고 여러 기능이 섞여있어 분할이 필요합니다.**

어떤 방식으로 진행하시겠습니까?
1. dashboard.css를 먼저 기능별로 분할
2. 전체 CSS 구조를 새롭게 재편성
3. 특정 파일만 선택해서 분할

어떤 방향으로 진행하시길 원하시나요?