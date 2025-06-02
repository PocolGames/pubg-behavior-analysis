/**
 * PUBG 플레이어 행동 분석 웹사이트 - 메인 JavaScript
 * 공통 기능, 네비게이션, 유틸리티 함수들
 */

// ==================== 전역 변수 ====================
const App = {
    currentPage: '',
    isMobile: false,
    isTablet: false,
    loadingStates: new Map(),
    animations: new Map()
};

// ==================== DOM 로드 완료 시 초기화 ====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎮 PUBG Analysis Dashboard 초기화...');
    
    // 기본 초기화
    App.init();
    
    // 페이지별 초기화
    App.initPage();
    
    console.log('✅ 초기화 완료');
});

// ==================== 메인 앱 객체 ====================
App.init = function() {
    // 반응형 체크
    this.checkResponsive();
    
    // 네비게이션 초기화
    this.initNavigation();
    
    // 스크롤 이벤트
    this.initScrollEffects();
    
    // 공통 이벤트 리스너
    this.initEventListeners();
    
    // 다크 모드 체크
    this.initTheme();
    
    // 페이지 로딩 완료
    this.hidePageLoader();
};

// ==================== 반응형 디자인 체크 ====================
App.checkResponsive = function() {
    const updateResponsive = () => {
        this.isMobile = window.innerWidth <= 768;
        this.isTablet = window.innerWidth <= 1024 && window.innerWidth > 768;
        
        document.body.classList.toggle('mobile', this.isMobile);
        document.body.classList.toggle('tablet', this.isTablet);
    };
    
    updateResponsive();
    window.addEventListener('resize', updateResponsive);
};

// ==================== 네비게이션 초기화 ====================
App.initNavigation = function() {
    // 모바일 메뉴 토글
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navMenu = document.querySelector('.nav-menu');
    
    if (mobileMenuBtn && navMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            mobileMenuBtn.classList.toggle('active');
        });
    }
    
    // 현재 페이지 하이라이트
    this.highlightCurrentPage();
    
    // 부드러운 앵커 스크롤
    this.initSmoothScroll();
};

// ==================== 현재 페이지 하이라이트 ====================
App.highlightCurrentPage = function() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-menu a');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (currentPath.includes(href) || 
            (currentPath === '/' && href === 'index.html')) {
            link.classList.add('active');
        }
    });
};

// ==================== 부드러운 스크롤 ====================
App.initSmoothScroll = function() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
};

// ==================== 스크롤 효과 ====================
App.initScrollEffects = function() {
    // 헤더 스크롤 효과
    const header = document.querySelector('.header');
    if (header) {
        let lastScrollY = window.scrollY;
        
        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;
            
            if (currentScrollY > 100) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
            
            // 스크롤 방향에 따른 헤더 숨김/표시
            if (currentScrollY > lastScrollY && currentScrollY > 200) {
                header.classList.add('hidden');
            } else {
                header.classList.remove('hidden');
            }
            
            lastScrollY = currentScrollY;
        });
    }
    
    // 스크롤 애니메이션
    this.initScrollAnimations();
};

// ==================== 스크롤 애니메이션 ====================
App.initScrollAnimations = function() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, observerOptions);
    
    // 애니메이션 대상 요소들
    document.querySelectorAll('.fade-in, .slide-up, .card').forEach(el => {
        observer.observe(el);
    });
};

// ==================== 공통 이벤트 리스너 ====================
App.initEventListeners = function() {
    // 모든 버튼에 리플 효과
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', this.createRippleEffect);
    });
    
    // 카드 호버 효과
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mouseenter', this.cardHoverEnter);
        card.addEventListener('mouseleave', this.cardHoverLeave);
    });
    
    // 모달 이벤트
    this.initModalEvents();
    
    // 알림 시스템
    this.initNotificationSystem();
};

// ==================== 리플 효과 ====================
App.createRippleEffect = function(e) {
    const button = e.currentTarget;
    const ripple = document.createElement('span');
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');
    
    button.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
};

// ==================== 카드 호버 효과 ====================
App.cardHoverEnter = function(e) {
    const card = e.currentTarget;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    card.style.setProperty('--mouse-x', x + 'px');
    card.style.setProperty('--mouse-y', y + 'px');
};

App.cardHoverLeave = function(e) {
    const card = e.currentTarget;
    card.style.removeProperty('--mouse-x');
    card.style.removeProperty('--mouse-y');
};

// ==================== 테마 관리 ====================
App.initTheme = function() {
    // 시스템 다크 모드 감지
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
    
    // 저장된 테마 또는 시스템 테마 적용
    const savedTheme = localStorage.getItem('theme') || (prefersDark.matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // 테마 변경 감지
    prefersDark.addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        }
    });
};

// ==================== 모달 시스템 ====================
App.initModalEvents = function() {
    // 모달 열기
    document.querySelectorAll('[data-modal]').forEach(trigger => {
        trigger.addEventListener('click', (e) => {
            e.preventDefault();
            const modalId = trigger.getAttribute('data-modal');
            this.openModal(modalId);
        });
    });
    
    // 모달 닫기
    document.querySelectorAll('.modal-close, .modal-backdrop').forEach(close => {
        close.addEventListener('click', (e) => {
            if (e.target === close) {
                this.closeModal();
            }
        });
    });
    
    // ESC 키로 모달 닫기
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            this.closeModal();
        }
    });
};

App.openModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('show');
        document.body.classList.add('modal-open');
        
        // 포커스 트랩
        const focusableElements = modal.querySelectorAll('button, input, textarea, select, a[href]');
        if (focusableElements.length > 0) {
            focusableElements[0].focus();
        }
    }
};

App.closeModal = function() {
    const activeModal = document.querySelector('.modal.show');
    if (activeModal) {
        activeModal.classList.remove('show');
        document.body.classList.remove('modal-open');
    }
};

// ==================== 알림 시스템 ====================
App.initNotificationSystem = function() {
    this.createNotificationContainer();
};

App.createNotificationContainer = function() {
    if (!document.querySelector('.notification-container')) {
        const container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
};

App.showNotification = function(message, type = 'info', duration = 5000) {
    const container = document.querySelector('.notification-container');
    const notification = document.createElement('div');
    
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close">×</button>
    `;
    
    container.appendChild(notification);
    
    // 애니메이션
    setTimeout(() => notification.classList.add('show'), 10);
    
    // 자동 제거
    if (duration > 0) {
        setTimeout(() => {
            this.removeNotification(notification);
        }, duration);
    }
    
    // 수동 제거
    notification.querySelector('.notification-close').addEventListener('click', () => {
        this.removeNotification(notification);
    });
    
    return notification;
};

App.getNotificationIcon = function(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || icons.info;
};

App.removeNotification = function(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 300);
};

// ==================== 페이지 로더 ====================
App.showPageLoader = function() {
    let loader = document.querySelector('.page-loader');
    if (!loader) {
        loader = document.createElement('div');
        loader.className = 'page-loader';
        loader.innerHTML = `
            <div class="loader-content">
                <div class="loader-spinner"></div>
                <p>로딩 중...</p>
            </div>
        `;
        document.body.appendChild(loader);
    }
    loader.classList.add('show');
};

App.hidePageLoader = function() {
    const loader = document.querySelector('.page-loader');
    if (loader) {
        loader.classList.remove('show');
        setTimeout(() => {
            if (loader.parentNode) {
                loader.parentNode.removeChild(loader);
            }
        }, 300);
    }
};

// ==================== 페이지별 초기화 ====================
App.initPage = function() {
    const path = window.location.pathname;
    
    if (path.includes('cluster-analysis')) {
        this.currentPage = 'cluster-analysis';
    } else if (path.includes('model-performance')) {
        this.currentPage = 'model-performance';
    } else if (path.includes('player-prediction')) {
        this.currentPage = 'player-prediction';
    } else {
        this.currentPage = 'home';
    }
    
    console.log(`📄 현재 페이지: ${this.currentPage}`);
};

// ==================== 유틸리티 함수들 ====================
App.utils = {
    // 숫자 포맷팅
    formatNumber: function(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    },
    
    // 퍼센트 포맷팅
    formatPercent: function(num, decimals = 1) {
        return (num * 100).toFixed(decimals) + '%';
    },
    
    // 디바운스
    debounce: function(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    },
    
    // 쓰로틀
    throttle: function(func, delay) {
        let lastCall = 0;
        return function (...args) {
            const now = new Date().getTime();
            if (now - lastCall < delay) {
                return;
            }
            lastCall = now;
            return func.apply(this, args);
        };
    },
    
    // 랜덤 ID 생성
    generateId: function() {
        return 'id_' + Math.random().toString(36).substr(2, 9);
    }
};

// ==================== CSS 애니메이션 추가 ====================
const additionalStyles = `
.ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0);
    animation: ripple-animation 0.6s linear;
    pointer-events: none;
}

@keyframes ripple-animation {
    to {
        transform: scale(4);
        opacity: 0;
    }
}

.header.scrolled {
    backdrop-filter: blur(10px);
    background: rgba(23, 25, 35, 0.95);
}

.header.hidden {
    transform: translateY(-100%);
}

.notification-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    pointer-events: none;
}

.notification {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transform: translateX(400px);
    transition: all 0.3s ease;
    pointer-events: all;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.notification.show {
    transform: translateX(0);
}

.page-loader {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(23, 25, 35, 0.9);
    backdrop-filter: blur(5px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
}

.page-loader.show {
    opacity: 1;
    visibility: visible;
}

.loader-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255, 107, 53, 0.3);
    border-top: 3px solid #ff6b35;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
`;

// 스타일 추가
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// ==================== 전역 접근 ====================
window.App = App;