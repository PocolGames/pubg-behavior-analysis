/**
 * PUBG 플레이어 행동 분석 - 네비게이션 컴포넌트
 * 모든 페이지에서 공통으로 사용되는 네비게이션 바를 동적으로 생성합니다.
 */

class NavigationComponent {
    constructor() {
        this.currentPath = window.location.pathname;
        this.basePath = this.getBasePath();
        this.navigationData = this.getNavigationData();
    }

    /**
     * 현재 페이지의 기준 경로를 계산합니다.
     * @returns {string} 기준 경로
     */
    getBasePath() {
        const path = this.currentPath;
        
        // 루트 디렉토리인 경우
        if (path.endsWith('index.html') || path.endsWith('/')) {
            return './';
        }
        
        // pages 폴더 내의 파일인 경우
        if (path.includes('/pages/')) {
            return '../';
        }
        
        // 기본값
        return './';
    }

    /**
     * 네비게이션 메뉴 데이터를 반환합니다.
     * @returns {Array} 네비게이션 메뉴 배열
     */
    getNavigationData() {
        return [
            {
                name: '대시보드',
                href: 'index.html',
                icon: 'fas fa-tachometer-alt',
                id: 'dashboard'
            },
            {
                name: '클러스터 분석',
                href: 'pages/cluster-analysis.html',
                icon: 'fas fa-project-diagram',
                id: 'cluster-analysis'
            },
            {
                name: '모델 성능',
                href: 'pages/model-performance.html',
                icon: 'fas fa-chart-line',
                id: 'model-performance'
            },
            {
                name: '플레이어 예측',
                href: 'pages/player-prediction.html',
                icon: 'fas fa-user-check',
                id: 'player-prediction'
            }
        ];
    }

    /**
     * 현재 활성 페이지를 확인합니다.
     * @param {string} href - 확인할 페이지 경로
     * @returns {boolean} 활성 페이지 여부
     */
    isActivePage(href) {
        const currentFile = this.currentPath.split('/').pop() || 'index.html';
        const targetFile = href.split('/').pop();
        
        // 메인 페이지인 경우
        if (currentFile === 'index.html' && targetFile === 'index.html') {
            return true;
        }
        
        // 다른 페이지들인 경우
        return currentFile === targetFile;
    }

    /**
     * 상대 경로를 조정합니다.
     * @param {string} href - 원본 경로
     * @returns {string} 조정된 경로
     */
    adjustPath(href) {
        if (href === 'index.html') {
            return this.basePath + 'index.html';
        }
        
        if (href.startsWith('pages/')) {
            return this.basePath + href;
        }
        
        return href;
    }

    /**
     * 네비게이션 HTML을 생성합니다.
     * @returns {string} 네비게이션 HTML 문자열
     */
    generateNavigationHTML() {
        const logoPath = this.adjustPath('index.html');
        
        let navLinksHTML = '';
        this.navigationData.forEach(item => {
            const isActive = this.isActivePage(item.href);
            const adjustedPath = this.adjustPath(item.href);
            const activeClass = isActive ? ' active' : '';
            
            navLinksHTML += `
                <a href="${adjustedPath}" class="nav-link${activeClass}" data-page="${item.id}">
                    ${item.name}
                </a>
            `;
        });

        return `
            <header class="header">
                <div class="container">
                    <div class="nav-brand">
                        <a href="${logoPath}">
                            <i class="fas fa-crosshairs logo-icon"></i>
                            <span>PUBG 플레이어 행동 분석</span>
                        </a>
                    </div>
                    
                    <nav class="nav-menu">
                        ${navLinksHTML}
                    </nav>
                    
                    <button class="nav-toggle" id="navToggle">
                        <span></span>
                        <span></span>
                        <span></span>
                    </button>
                </div>
            </header>
        `;
    }

    /**
     * 모바일 네비게이션 토글 이벤트를 설정합니다.
     */
    setupMobileToggle() {
        const navToggle = document.getElementById('navToggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
                navToggle.classList.toggle('active');
            });

            // 메뉴 항목 클릭 시 모바일 메뉴 닫기
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                link.addEventListener('click', () => {
                    navMenu.classList.remove('active');
                    navToggle.classList.remove('active');
                });
            });

            // 윈도우 리사이즈 시 모바일 메뉴 상태 초기화
            window.addEventListener('resize', () => {
                if (window.innerWidth > 768) {
                    navMenu.classList.remove('active');
                    navToggle.classList.remove('active');
                }
            });
        }
    }

    /**
     * 네비게이션을 페이지에 렌더링합니다.
     * @param {string} targetSelector - 네비게이션을 삽입할 요소의 선택자
     */
    render(targetSelector = 'body') {
        const targetElement = document.querySelector(targetSelector);
        
        if (!targetElement) {
            console.error(`Navigation: 대상 요소 '${targetSelector}'를 찾을 수 없습니다.`);
            return;
        }

        // 기존 헤더가 있다면 제거
        const existingHeader = document.querySelector('.header');
        if (existingHeader) {
            existingHeader.remove();
        }
        
        // 추가: 하드코딩된 네비게이션 요소들 제거
        // .nav-brand가 있는 .container 찾아서 제거
        const navBrandElements = document.querySelectorAll('.nav-brand');
        navBrandElements.forEach(navBrand => {
            const container = navBrand.closest('.container');
            if (container) {
                container.remove();
            }
        });
        
        // 혹시 남아있는 nav-menu와 nav-toggle 요소들도 제거
        const remainingNavMenus = document.querySelectorAll('.nav-menu');
        const remainingNavToggles = document.querySelectorAll('.nav-toggle');
        
        remainingNavMenus.forEach(menu => {
            const container = menu.closest('.container');
            if (container) container.remove();
        });
        
        remainingNavToggles.forEach(toggle => {
            const container = toggle.closest('.container');
            if (container) container.remove();
        });

        // 새로운 네비게이션 HTML 생성
        const navigationHTML = this.generateNavigationHTML();
        
        // body의 첫 번째 자식으로 삽입
        if (targetSelector === 'body') {
            targetElement.insertAdjacentHTML('afterbegin', navigationHTML);
        } else {
            targetElement.innerHTML = navigationHTML;
        }

        // 이벤트 리스너 설정
        this.setupMobileToggle();
        
        console.log('Navigation component loaded successfully');
    }

    /**
     * 네비게이션 정보를 반환합니다 (디버깅용)
     * @returns {Object} 네비게이션 정보
     */
    getInfo() {
        return {
            currentPath: this.currentPath,
            basePath: this.basePath,
            navigationData: this.navigationData
        };
    }
}

/**
 * 전역 함수: 네비게이션 컴포넌트 초기화
 * 모든 페이지에서 이 함수를 호출하여 네비게이션을 생성합니다.
 */
function initNavigation() {
    // DOM이 로드된 후 실행
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            const navigation = new NavigationComponent();
            navigation.render();
        });
    } else {
        const navigation = new NavigationComponent();
        navigation.render();
    }
}

/**
 * 수동으로 네비게이션을 초기화하는 함수 (선택적 사용)
 * @param {string} targetSelector - 대상 선택자
 */
function renderNavigation(targetSelector = 'body') {
    const navigation = new NavigationComponent();
    navigation.render(targetSelector);
    return navigation;
}

// 전역 객체에 함수 노출
window.initNavigation = initNavigation;
window.renderNavigation = renderNavigation;
window.NavigationComponent = NavigationComponent;
