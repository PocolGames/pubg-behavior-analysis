/**
 * PUBG 플레이어 행동 분석 - 네비게이션 컴포넌트
 * 모든 페이지에서 공통으로 사용되는 네비게이션 바를 동적으로 생성합니다.
 * 접근성(ARIA), 키보드 네비게이션, 드롭다운 메뉴를 지원합니다.
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
                id: 'cluster-analysis',
                dropdown: [
                    {
                        name: '플레이어 유형 분석',
                        href: 'pages/cluster-analysis.html#overview',
                        icon: 'fas fa-users'
                    },
                    {
                        name: '클러스터 비교',
                        href: 'pages/cluster-analysis.html#comparison',
                        icon: 'fas fa-balance-scale'
                    },
                    {
                        name: '특성 분석',
                        href: 'pages/cluster-analysis.html#features',
                        icon: 'fas fa-chart-bar'
                    }
                ]
            },
            {
                name: '모델 성능',
                href: 'pages/model-performance.html',
                icon: 'fas fa-chart-line',
                id: 'model-performance',
                dropdown: [
                    {
                        name: '모델 개요',
                        href: 'pages/model-performance.html#overview',
                        icon: 'fas fa-brain'
                    },
                    {
                        name: '성능 지표',
                        href: 'pages/model-performance.html#metrics',
                        icon: 'fas fa-tachometer-alt'
                    },
                    {
                        name: '특성 중요도',
                        href: 'pages/model-performance.html#importance',
                        icon: 'fas fa-star'
                    }
                ]
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
        this.navigationData.forEach((item, index) => {
            const isActive = this.isActivePage(item.href);
            const adjustedPath = this.adjustPath(item.href);
            const activeClass = isActive ? ' active' : '';
            
            if (item.dropdown && item.dropdown.length > 0) {
                // 드롭다운 메뉴가 있는 경우
                let dropdownHTML = '';
                item.dropdown.forEach((dropdownItem, dropdownIndex) => {
                    const dropdownPath = this.adjustPath(dropdownItem.href);
                    dropdownHTML += `
                        <a href="${dropdownPath}" 
                           class="dropdown-item" 
                           role="menuitem"
                           aria-label="${dropdownItem.name} 페이지로 이동"
                           tabindex="-1"
                           id="dropdown-${item.id}-${dropdownIndex}">
                            <i class="${dropdownItem.icon}" aria-hidden="true"></i>
                            ${dropdownItem.name}
                        </a>
                    `;
                });
                
                navLinksHTML += `
                    <div class="nav-item-dropdown" data-page="${item.id}" role="none">
                        <a href="${adjustedPath}" 
                           class="nav-link dropdown-toggle${activeClass}" 
                           data-page="${item.id}"
                           role="button"
                           aria-expanded="false"
                           aria-haspopup="true"
                           aria-controls="dropdown-menu-${item.id}"
                           aria-label="${item.name} 메뉴 열기"
                           id="dropdown-toggle-${item.id}"
                           tabindex="0">
                            ${item.name}
                            <i class="fas fa-chevron-down dropdown-arrow" aria-hidden="true"></i>
                        </a>
                        <div class="dropdown-menu" 
                             role="menu"
                             aria-labelledby="dropdown-toggle-${item.id}"
                             id="dropdown-menu-${item.id}">
                            ${dropdownHTML}
                        </div>
                    </div>
                `;
            } else {
                // 일반 메뉴
                navLinksHTML += `
                    <a href="${adjustedPath}" 
                       class="nav-link${activeClass}" 
                       data-page="${item.id}"
                       role="menuitem"
                       aria-label="${item.name} 페이지로 이동"
                       tabindex="0"
                       ${isActive ? 'aria-current="page"' : ''}>
                        ${item.name}
                    </a>
                `;
            }
        });

        return `
            <header class="header" role="banner">
                <div class="container">
                    <div class="nav-brand">
                        <a href="${logoPath}" 
                           aria-label="PUBG 플레이어 행동 분석 홈페이지로 이동"
                           tabindex="0">
                            <i class="fas fa-crosshairs logo-icon" aria-hidden="true"></i>
                            <span>PUBG 플레이어 행동 분석</span>
                        </a>
                    </div>
                    
                    <nav class="nav-menu" 
                         role="navigation" 
                         aria-label="주 메뉴">
                        ${navLinksHTML}
                    </nav>
                    
                    <button class="nav-toggle" 
                            id="navToggle"
                            type="button"
                            aria-label="메뉴 열기"
                            aria-expanded="false"
                            aria-controls="nav-menu"
                            tabindex="0">
                        <span aria-hidden="true"></span>
                        <span aria-hidden="true"></span>
                        <span aria-hidden="true"></span>
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
            // 클릭 이벤트
            navToggle.addEventListener('click', () => {
                const isExpanded = navMenu.classList.contains('active');
                navMenu.classList.toggle('active');
                navToggle.classList.toggle('active');
                
                // ARIA 속성 업데이트
                navToggle.setAttribute('aria-expanded', !isExpanded);
                navToggle.setAttribute('aria-label', 
                    !isExpanded ? '메뉴 닫기' : '메뉴 열기'
                );
            });
        }
    }

    /**
     * 드롭다운 메뉴를 설정합니다.
     */
    setupDropdownMenus() {
        const dropdownToggles = document.querySelectorAll('.dropdown-toggle');
        
        dropdownToggles.forEach(toggle => {
            const menu = toggle.nextElementSibling;
            
            // 클릭 이벤트
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                const isExpanded = toggle.getAttribute('aria-expanded') === 'true';
                
                // 다른 드롭다운 닫기
                this.closeAllDropdowns();
                
                // 현재 드롭다운 토글
                if (!isExpanded) {
                    toggle.setAttribute('aria-expanded', 'true');
                    menu.classList.add('show');
                }
            });
            
            // 호버 이벤트
            const dropdown = toggle.closest('.nav-item-dropdown');
            dropdown.addEventListener('mouseenter', () => {
                toggle.setAttribute('aria-expanded', 'true');
                menu.classList.add('show');
            });
            
            dropdown.addEventListener('mouseleave', () => {
                toggle.setAttribute('aria-expanded', 'false');
                menu.classList.remove('show');
            });
        });
        
        // 외부 클릭으로 드롭다운 닫기
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-item-dropdown')) {
                this.closeAllDropdowns();
            }
        });
    }

    /**
     * 모든 드롭다운을 닫습니다.
     */
    closeAllDropdowns() {
        const dropdowns = document.querySelectorAll('.dropdown-menu');
        const toggles = document.querySelectorAll('.dropdown-toggle');
        
        dropdowns.forEach(dropdown => dropdown.classList.remove('show'));
        toggles.forEach(toggle => toggle.setAttribute('aria-expanded', 'false'));
    }

    /**
     * 키보드 네비게이션을 설정합니다.
     */
    setupKeyboardNavigation() {
        // 스킵 링크 추가 (접근성)
        this.addSkipLink();
        
        // 포커스 관리
        this.setupFocusManagement();
        
        // ESC 키로 모든 메뉴 닫기
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // 모바일 메뉴 닫기
                const navMenu = document.querySelector('.nav-menu');
                const navToggle = document.getElementById('navToggle');
                
                if (navMenu && navMenu.classList.contains('active')) {
                    navMenu.classList.remove('active');
                    navToggle.classList.remove('active');
                    navToggle.setAttribute('aria-expanded', 'false');
                    navToggle.setAttribute('aria-label', '메뉴 열기');
                    navToggle.focus();
                }
                
                // 모든 드롭다운 닫기
                this.closeAllDropdowns();
            }
        });
    }

    /**
     * 스킵 링크를 추가합니다.
     */
    addSkipLink() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-link';
        skipLink.textContent = '본문으로 바로가기';
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
            z-index: 10000;
            border-radius: 4px;
            transition: top 0.3s;
        `;
        
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });
        
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });
        
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    /**
     * 포커스 관리를 설정합니다.
     */
    setupFocusManagement() {
        // 포커스 가능한 요소들
        const focusableElements = [
            'a[href]',
            'button:not([disabled])',
            'input:not([disabled])',
            'select:not([disabled])',
            'textarea:not([disabled])',
            '[tabindex]:not([tabindex="-1"])'
        ].join(', ');

        // 네비게이션 내에서 Tab 순환
        const nav = document.querySelector('.nav-menu');
        if (nav) {
            nav.addEventListener('keydown', (e) => {
                if (e.key === 'Tab') {
                    const focusable = nav.querySelectorAll(focusableElements);
                    const firstFocusable = focusable[0];
                    const lastFocusable = focusable[focusable.length - 1];

                    if (e.shiftKey) {
                        // Shift + Tab
                        if (document.activeElement === firstFocusable) {
                            e.preventDefault();
                            lastFocusable.focus();
                        }
                    } else {
                        // Tab
                        if (document.activeElement === lastFocusable) {
                            e.preventDefault();
                            firstFocusable.focus();
                        }
                    }
                }
            });
        }
    }

    /**
     * 접근성 진단 도구
     */
    checkAccessibility() {
        const issues = [];
        
        // ARIA 레이블 확인 - 더 정확한 검사
        const elementsNeedingLabels = document.querySelectorAll('button:not([aria-label]):not([aria-labelledby]), [role="button"]:not([aria-label]):not([aria-labelledby])');
        elementsNeedingLabels.forEach(element => {
            // 텍스트 내용이 있는지 확인
            const hasTextContent = element.textContent && element.textContent.trim().length > 0;
            if (!hasTextContent) {
                issues.push(`요소에 접근성 레이블이 없습니다: ${element.tagName} ${element.className}`);
            }
        });
        
        // 링크 요소 확인
        const linksNeedingLabels = document.querySelectorAll('a:not([aria-label]):not([aria-labelledby])');
        linksNeedingLabels.forEach(element => {
            const hasTextContent = element.textContent && element.textContent.trim().length > 0;
            const hasTitle = element.getAttribute('title');
            if (!hasTextContent && !hasTitle) {
                issues.push(`링크에 접근성 레이블이 없습니다: ${element.href}`);
            }
        });
        
        // 포커스 가능한 요소 확인
        const interactive = document.querySelectorAll('a, button, input, select, textarea, [tabindex]');
        interactive.forEach(element => {
            const tabindex = element.getAttribute('tabindex');
            if (tabindex && parseInt(tabindex) > 0) {
                issues.push(`양수 tabindex 사용을 피하세요: ${element.tagName}`);
            }
        });
        
        // 이미지 alt 텍스트 확인
        const images = document.querySelectorAll('img:not([alt])');
        images.forEach(img => {
            issues.push(`이미지에 alt 텍스트가 없습니다: ${img.src}`);
        });
        
        if (issues.length === 0) {
            console.log('✅ 접근성 검사 통과!');
        } else {
            console.warn('⚠️ 접근성 문제 발견:');
            issues.forEach(issue => console.warn(issue));
        }
        
        return issues;
    }

    /**
     * 네비게이션을 초기화합니다.
     */
    init() {
        const navContainer = document.getElementById('navigation-container') || 
                           document.querySelector('body');
        
        if (navContainer) {
            // 기존 네비게이션이 있다면 제거
            const existingNav = navContainer.querySelector('.header');
            if (existingNav) {
                existingNav.remove();
            }
            
            // 새 네비게이션 추가
            const navHTML = this.generateNavigationHTML();
            
            if (navContainer.id === 'navigation-container') {
                navContainer.innerHTML = navHTML;
            } else {
                navContainer.insertAdjacentHTML('afterbegin', navHTML);
            }
            
            // 이벤트 설정
            this.setupMobileToggle();
            this.setupDropdownMenus();
            this.setupKeyboardNavigation();
            
            // 접근성 검사 (개발 모드에서만)
            if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                this.checkAccessibility();
            }
            
            console.log('✅ Navigation component loaded with accessibility features');
        } else {
            console.error('❌ Navigation container not found');
        }
    }
}

// 전역 함수로 내보내기
function initNavigation() {
    const nav = new NavigationComponent();
    nav.init();
}
