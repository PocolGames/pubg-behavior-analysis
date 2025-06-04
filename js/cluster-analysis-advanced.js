/**
 * PUBG Cluster Analysis JavaScript - 완성된 기능들
 * 툴팁, 로딩, 데이터 검증, 브라우저 호환성 등
 */

// ===============================
// 툴팁 기능
// ===============================
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
        element.addEventListener('focus', showTooltip);
        element.addEventListener('blur', hideTooltip);
    });
}

function showTooltip(e) {
    const element = e.currentTarget;
    const tooltipText = element.getAttribute('data-tooltip');
    
    if (!tooltipText) return;
    
    // 기존 툴팁 제거
    hideTooltip();
    
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = tooltipText;
    tooltip.id = 'active-tooltip';
    
    // CSS 스타일 추가
    Object.assign(tooltip.style, {
        position: 'absolute',
        backgroundColor: '#333',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '4px',
        fontSize: '12px',
        whiteSpace: 'nowrap',
        zIndex: '10000',
        opacity: '0',
        transition: 'opacity 0.3s ease',
        pointerEvents: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
    });
    
    document.body.appendChild(tooltip);
    
    // 위치 계산
    const rect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
    let top = rect.top - tooltipRect.height - 10;
    
    // 화면 경계 체크
    if (left < 10) left = 10;
    if (left + tooltipRect.width > window.innerWidth - 10) {
        left = window.innerWidth - tooltipRect.width - 10;
    }
    if (top < 10) {
        top = rect.bottom + 10;
    }
    
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    tooltip.style.opacity = '1';
}

function hideTooltip() {
    const tooltip = document.getElementById('active-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// ===============================
// 로딩 진행 상태
// ===============================
function showLoadingProgress() {
    // 로딩 오버레이 생성
    const loader = document.createElement('div');
    loader.className = 'page-loader';
    loader.innerHTML = `
        <div class="loader-content">
            <div class="loader-spinner"></div>
            <div class="progress-text">클러스터 데이터 로딩...</div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
    `;
    
    // CSS 스타일 추가
    Object.assign(loader.style, {
        position: 'fixed',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: '9999',
        transition: 'opacity 0.3s ease'
    });
    
    document.body.appendChild(loader);
    
    // 진행 단계
    const progressSteps = [
        { text: '클러스터 데이터 로딩...', duration: 500 },
        { text: '차트 초기화...', duration: 800 },
        { text: '인터랙티브 요소 설정...', duration: 400 },
        { text: '완료!', duration: 300 }
    ];
    
    let currentStep = 0;
    
    function updateProgress() {
        if (currentStep >= progressSteps.length) {
            // 로딩 완료
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 300);
            return;
        }
        
        const step = progressSteps[currentStep];
        const progressText = loader.querySelector('.progress-text');
        const progressBar = loader.querySelector('.progress-fill');
        
        if (progressText) progressText.textContent = step.text;
        if (progressBar) {
            const percentage = ((currentStep + 1) / progressSteps.length) * 100;
            progressBar.style.width = percentage + '%';
        }
        
        currentStep++;
        setTimeout(updateProgress, step.duration);
    }
    
    updateProgress();
}

// ===============================
// 데이터 검증 및 오류 처리
// ===============================
function validateClusterData() {
    try {
        // 필수 데이터 구조 검증
        if (!CLUSTER_DATA || !CLUSTER_DATA.clusters) {
            throw new Error('클러스터 데이터가 없습니다.');
        }
        
        if (!Array.isArray(CLUSTER_DATA.clusters)) {
            throw new Error('클러스터 데이터 형식이 올바르지 않습니다.');
        }
        
        // 각 클러스터 데이터 검증
        CLUSTER_DATA.clusters.forEach((cluster, index) => {
            const required = ['id', 'name', 'type', 'count', 'percentage'];
            required.forEach(field => {
                if (cluster[field] === undefined || cluster[field] === null) {
                    throw new Error(`클러스터 ${index}의 ${field} 필드가 누락되었습니다.`);
                }
            });
        });
        
        console.log('✅ 클러스터 데이터 검증 완료');
        return true;
        
    } catch (error) {
        console.error('❌ 데이터 검증 실패:', error.message);
        showErrorMessage(error.message);
        return false;
    }
}

// ===============================
// 브라우저 호환성 체크
// ===============================
function checkBrowserCompatibility() {
    const features = {
        'ES6 지원': () => {
            try {
                new Function('(a = 0) => a');
                return true;
            } catch (e) {
                return false;
            }
        },
        'Canvas 지원': () => {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext && canvas.getContext('2d'));
        },
        'Intersection Observer': () => {
            return 'IntersectionObserver' in window;
        },
        'CSS Grid': () => {
            return window.CSS && CSS.supports('display', 'grid');
        }
    };
    
    const incompatible = [];
    
    Object.entries(features).forEach(([name, test]) => {
        if (!test()) {
            incompatible.push(name);
        }
    });
    
    if (incompatible.length > 0) {
        console.warn('⚠️ 일부 기능이 지원되지 않을 수 있습니다:', incompatible);
        showNotification(
            '일부 고급 기능이 현재 브라우저에서 제한될 수 있습니다. 최신 브라우저 사용을 권장합니다.',
            'warning'
        );
    }
}

// ===============================
// 차트 테마 및 커스터마이징
// ===============================
function applyChartThemes() {
    // Chart.js 기본 설정
    if (typeof Chart !== 'undefined') {
        Chart.defaults.font.family = "'Noto Sans KR', sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.color = '#333';
        Chart.defaults.borderColor = '#e0e0e0';
        Chart.defaults.backgroundColor = 'rgba(54, 162, 235, 0.2)';
        
        // 반응형 설정
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
        
        console.log('✅ 차트 테마 설정 완료');
    }
}

// ===============================
// 키보드 네비게이션
// ===============================
function setupKeyboardNavigation() {
    let focusableElements = [];
    
    function updateFocusableElements() {
        focusableElements = Array.from(document.querySelectorAll(
            'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )).filter(el => {
            return el.offsetParent !== null && !el.disabled;
        });
    }
    
    updateFocusableElements();
    
    document.addEventListener('keydown', function(e) {
        const currentIndex = focusableElements.indexOf(document.activeElement);
        
        switch(e.key) {
            case 'Tab':
                if (e.shiftKey) {
                    // Shift + Tab: 이전 요소
                    if (currentIndex <= 0) {
                        e.preventDefault();
                        focusableElements[focusableElements.length - 1].focus();
                    }
                } else {
                    // Tab: 다음 요소
                    if (currentIndex >= focusableElements.length - 1) {
                        e.preventDefault();
                        focusableElements[0].focus();
                    }
                }
                break;
                
            case 'Escape':
                closeModal();
                break;
                
            case 'Enter':
            case ' ':
                if (document.activeElement.classList.contains('cluster-detail-btn')) {
                    e.preventDefault();
                    document.activeElement.click();
                }
                break;
        }
    });
    
    // DOM 변경 시 포커스 가능한 요소 업데이트
    const observer = new MutationObserver(updateFocusableElements);
    observer.observe(document.body, { childList: true, subtree: true });
}

// ===============================
// 에러 복구 및 재시도
// ===============================
function setupErrorRecovery() {
    window.addEventListener('error', function(e) {
        console.error('❌ JavaScript 오류:', e.error);
        
        // 차트 관련 오류인 경우 재시도
        if (e.error && e.error.message && e.error.message.includes('Chart')) {
            setTimeout(() => {
                try {
                    initializeCharts();
                    showNotification('차트가 복구되었습니다.', 'success');
                } catch (retryError) {
                    showErrorMessage('차트 복구에 실패했습니다.');
                }
            }, 1000);
        }
    });
    
    // Promise 오류 처리
    window.addEventListener('unhandledrejection', function(e) {
        console.error('❌ Promise 오류:', e.reason);
        e.preventDefault();
        showErrorMessage('일시적인 오류가 발생했습니다. 페이지를 새로고침해주세요.');
    });
}

// ===============================
// 메인 초기화 함수 (기존 파일에 추가용)
// ===============================
function initializeAdvancedFeatures() {
    // 차트 테마 적용
    applyChartThemes();
    
    // 툴팁 초기화
    initializeTooltips();
    
    // 키보드 네비게이션 설정
    setupKeyboardNavigation();
    
    // 에러 복구 설정
    setupErrorRecovery();
    
    // 브라우저 호환성 체크
    checkBrowserCompatibility();
    
    // 로딩 프로그레스 (필요시)
    if (document.querySelector('.page-loader')) {
        showLoadingProgress();
    }
    
    console.log('✅ 고급 기능 초기화 완료');
}

// ===============================
// CSS 스타일 동적 추가
// ===============================
function addDynamicStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .custom-tooltip {
            font-family: 'Noto Sans KR', sans-serif;
            font-size: 12px;
            line-height: 1.4;
        }
        
        .page-loader {
            font-family: 'Noto Sans KR', sans-serif;
        }
        
        .loader-content {
            text-align: center;
            max-width: 300px;
        }
        
        .loader-spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2196F3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-text {
            margin-bottom: 15px;
            color: #333;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: #e0e0e0;
            border-radius: 2px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2196F3, #4CAF50);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        [data-tooltip] {
            position: relative;
            cursor: help;
        }
        
        .notification {
            max-width: 400px;
            min-width: 250px;
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .notification-close {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        
        .notification-close:hover {
            opacity: 1;
        }
    `;
    
    document.head.appendChild(style);
}

// 페이지 로드 시 스타일 추가
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addDynamicStyles);
} else {
    addDynamicStyles();
}

console.log('✅ 클러스터 분석 고급 기능 모듈 로드 완료');