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

---

## base 분리

✅ 완료된 작업:
css/base/variables.css - CSS 변수 정의 (색상, 폰트, 스페이싱 등)
css/base/reset.css - CSS 리셋 및 기본 설정
css/base/typography.css - 폰트 및 텍스트 스타일

---

## 현재 상황 설명:

### 🔄 **분할한 것 (base 스타일들)**
- CSS Reset, 기본 설정 → `base/reset.css`
- CSS 변수, 색상 정의 → `base/variables.css` 
- 폰트, 타이포그래피 → `base/typography.css`

### 🎯 **main.css에 남아있는 것들 (여전히 필요)**
- 버튼 스타일 (`.btn`, `.btn-primary` 등)
- 헤더 스타일 (`.main-header`, `.nav-link` 등)
- 히어로 섹션 (`.hero-section`, `.hero-title` 등)
- 카드 스타일 (`.card-base` 등)
- 푸터 스타일 (`.main-footer` 등)
- 유틸리티 클래스들

### 📋 **앞으로 할 작업**
다음 단계에서 `main.css`의 나머지 스타일들도 분할할 예정입니다:
- `layout/header.css`, `layout/footer.css`
- `components/buttons.css`, `components/cards.css`
- `utilities/helpers.css`

다음 단계로 `layout/` 폴더를 만들어서 헤더/푸터를 분리하시겠습니까?

---

## Layout 분리리

✅ 완료된 작업: layout/header.css 생성
✅ 완료된 작업: layout/footer.css 생성
✅ 완료된 작업: layout/grid.css 생성

---

## Components 분리

✅ 완료된 작업: components/buttons.css 생성
✅ 완료된 작업: components/cards.css 생성
✅ 완료된 작업: components/forms.css 생성
✅ 완료된 작업: components/animations.css 생성

다음 작업: `utilities/` 폴더를 생성하고 `utilities/helpers.css` 파일을 만들어 유틸리티 클래스들을 분리하겠습니다.

---

## utilities 분리

✅ 완료된 작업: utilities/helpers.css 생성
✅ 완료된 작업: utilities/responsive.css 생성

---

네, `main.css` 파일을 정리하고 분리된 CSS 파일들을 import하는 구조로 변경하는 방법을 알려드리겠습니다.

## 📋 main.css 수정 방법

### 1. 현재 main.css에서 분리된 스타일들 제거

다음 스타일들을 `main.css`에서 **삭제**하세요:

**제거할 스타일들:**
```css
/* 헤더 관련 (layout/header.css로 이동됨) */
.main-header { ... }
.header-content { ... }
.logo-section { ... }
.logo-icon { ... }
.site-title { ... }
.main-nav .nav-list { ... }
.nav-link { ... }

/* 푸터 관련 (layout/footer.css로 이동됨) */
.main-footer { ... }
.footer-content { ... }
.footer-info { ... }
.footer-stats { ... }
.footer-stat { ... }

/* 히어로/그리드 관련 (layout/grid.css로 이동됨) */
.hero-section { ... }
.hero-title { ... }
.hero-description { ... }
.hero-stats { ... }
.stat-item { ... }
.stat-number { ... }
.stat-label { ... }
.main-dashboard { ... }
.section-title { ... }

/* 버튼 관련 (components/buttons.css로 이동됨) */
.btn { ... }
.btn-primary { ... }
.btn-secondary { ... }
.btn-success { ... }

/* 카드 관련 (components/cards.css로 이동됨) */
.card-base { ... }
.card-header { ... }

/* 유틸리티 클래스들 (utilities/helpers.css로 이동됨) */
.mb-1, .mb-2, .mb-3 { ... }
.mt-1, .mt-2, .mt-3 { ... }
.d-flex, .align-items-center { ... }
/* 기타 모든 유틸리티 클래스들 */

/* 반응형 스타일들 (utilities/responsive.css로 이동됨) */
@media (max-width: 768px) { ... }
@media (max-width: 480px) { ... }
```

### 2. main.css 상단에 import 구문 추가

`main.css` 파일의 **맨 상단**에 다음 import 구문들을 추가하세요:

```css
/* ===== PUBG 플레이어 행동 분석 - Main CSS ===== */

/* Base Styles */
@import 'base/reset.css';
@import 'base/variables.css';
@import 'base/typography.css';

/* Layout Styles */
@import 'layout/header.css';
@import 'layout/footer.css';
@import 'layout/grid.css';

/* Component Styles */
@import 'components/buttons.css';
@import 'components/cards.css';
@import 'components/forms.css';
@import 'components/animations.css';

/* Utility Styles */
@import 'utilities/helpers.css';
@import 'utilities/responsive.css';

/* Page-specific styles remain here */
```

### 3. main.css에 남겨둘 내용

다음과 같은 스타일들만 `main.css`에 남겨두세요:

```css
/* 페이지별 특별한 스타일들 */
/* 다른 CSS 파일로 분류하기 어려운 특수한 스타일들 */
/* 프로젝트 특화 스타일들 */
```

### 4. HTML 파일들의 CSS 링크 확인

각 HTML 파일에서 CSS 링크가 올바른지 확인하세요:

```html
<!-- index.html 등에서 -->
<link rel="stylesheet" href="css/main.css">
```

### 5. 수정 후 확인 사항

1. **브라우저에서 확인** - 모든 스타일이 정상적으로 적용되는지 확인
2. **개발자 도구에서 확인** - CSS 오류가 없는지 확인
3. **반응형 동작 확인** - 다양한 화면 크기에서 정상 동작하는지 확인

### 6. 선택적 최적화

페이지별로 필요한 CSS만 로드하고 싶다면:

```html
<!-- 기본 스타일만 -->
<link rel="stylesheet" href="css/base/variables.css">
<link rel="stylesheet" href="css/base/reset.css">
<link rel="stylesheet" href="css/layout/header.css">
<link rel="stylesheet" href="css/components/buttons.css">
<!-- 필요한 것만 선택적으로 로드 -->
```

이렇게 수정하시면 CSS가 모듈화되어 유지보수가 훨씬 쉬워집니다! 수정 중에 문제가 생기면 언제든 알려주세요.