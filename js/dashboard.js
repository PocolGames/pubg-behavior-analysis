/**
 * PUBG 플레이어 행동 분석 웹사이트 - 대시보드 JavaScript
 * 메인 대시보드 페이지 전용 기능들
 */

// ==================== 대시보드 객체 ====================
const Dashboard = {
    initialized: false,
    animationRunning: false,
    charts: new Map(),
    counters: new Map()
};

// ==================== DOM 로드 완료 시 초기화 ====================
document.addEventListener('DOMContentLoaded', function() {
    // 메인 페이지에서만 실행
    if (window.location.pathname === '/' || window.location.pathname.includes('index.html')) {
        console.log('📊 Dashboard 모듈 초기화...');
        Dashboard.init();
    }
});

// ==================== 대시보드 초기화 ====================
Dashboard.init = function() {
    if (this.initialized) return;
    
    try {
        // 통계 카운터 애니메이션
        this.initStatsCounters();
        
        // 카드 인터랙션 효과
        this.initCardInteractions();
        
        // 호버 효과
        this.initHoverEffects();
        
        // 진행률 바 애니메이션
        this.initProgressBars();
        
        // 툴팁 시스템
        this.initTooltips();
        
        // 스크롤 애니메이션
        this.initScrollAnimations();
        
        this.initialized = true;
        console.log('✅ Dashboard 초기화 완료');
        
    } catch (error) {
        console.error('❌ Dashboard 초기화 오류:', error);
    }
};

// ==================== 통계 카운터 애니메이션 ====================
Dashboard.initStatsCounters = function() {
    const counters = document.querySelectorAll('.stat-number, .type-percentage, .metric-value');
    
    const animateCounter = (element) => {
        const target = this.parseNumber(element.textContent);
        const duration = 2000; // 2초
        const startTime = Date.now();
        const isPercentage = element.textContent.includes('%');
        const hasComma = element.textContent.includes(',');
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // 이징 함수 (ease-out)
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = target * easeOut;
            
            // 숫자 포맷팅
            let displayValue;
            if (isPercentage) {
                displayValue = current.toFixed(1) + '%';
            } else if (hasComma) {
                displayValue = Math.floor(current).toLocaleString();
            } else if (target > 1000) {
                displayValue = Math.floor(current).toLocaleString();
            } else {
                displayValue = current.toFixed(1);
            }
            
            element.textContent = displayValue;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                // 최종값 설정
                if (isPercentage) {
                    element.textContent = target.toFixed(1) + '%';
                } else if (hasComma || target > 1000) {
                    element.textContent = Math.floor(target).toLocaleString();
                } else {
                    element.textContent = target.toString();
                }
            }
        };
        
        // Intersection Observer로 뷰포트 진입시 애니메이션 시작
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !entry.target.hasAttribute('data-animated')) {
                    entry.target.setAttribute('data-animated', 'true');
                    setTimeout(() => animate(), Math.random() * 500); // 랜덤 지연으로 시차 효과
                }
            });
        }, { threshold: 0.5 });
        
        observer.observe(element);
    };
    
    counters.forEach(counter => {
        animateCounter(counter);
    });
};

// ==================== 숫자 파싱 헬퍼 함수 ====================
Dashboard.parseNumber = function(text) {
    // 퍼센트 제거
    let cleanText = text.replace('%', '');
    
    // 쉼표 제거
    cleanText = cleanText.replace(/,/g, '');
    
    // K, M 단위 처리
    if (cleanText.includes('K')) {
        return parseFloat(cleanText.replace('K', '')) * 1000;
    } else if (cleanText.includes('M')) {
        return parseFloat(cleanText.replace('M', '')) * 1000000;
    }
    
    return parseFloat(cleanText) || 0;
};

// ==================== 카드 인터랙션 효과 ====================
Dashboard.initCardInteractions = function() {
    const cards = document.querySelectorAll('.player-type-card, .analysis-card, .insight-card');
    
    cards.forEach(card => {
        // 3D 틸트 효과
        card.addEventListener('mousemove', (e) => {
            if (App.isMobile) return; // 모바일에서는 비활성화
            
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / centerY * -10; // 최대 10도
            const rotateY = (x - centerX) / centerX * 10;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
            card.style.transition = 'transform 0.1s ease';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)';
            card.style.transition = 'transform 0.3s ease';
        });
        
        // 클릭 애니메이션
        card.addEventListener('click', (e) => {
            // 리플 효과는 main.js에서 처리됨
            card.style.transform = 'scale(0.98)';
            setTimeout(() => {
                card.style.transform = '';
            }, 150);
        });
    });
};

// ==================== 호버 효과 ====================
Dashboard.initHoverEffects = function() {
    // 플레이어 타입 카드 호버
    const typeCards = document.querySelectorAll('.player-type-card');
    
    typeCards.forEach(card => {
        const icon = card.querySelector('.type-icon');
        
        card.addEventListener('mouseenter', () => {
            if (icon) {
                icon.style.transform = 'scale(1.2) rotate(10deg)';
                icon.style.transition = 'transform 0.3s ease';
            }
            
            // 배경 그라디언트 애니메이션
            card.style.backgroundSize = '110% 110%';
        });
        
        card.addEventListener('mouseleave', () => {
            if (icon) {
                icon.style.transform = 'scale(1) rotate(0deg)';
            }
            
            card.style.backgroundSize = '100% 100%';
        });
    });
    
    // 버튼 호버 효과
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            btn.style.transform = 'translateY(-2px)';
            btn.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.3)';
        });
        
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'translateY(0)';
            btn.style.boxShadow = '';
        });
    });
};

// ==================== 진행률 바 애니메이션 ====================
Dashboard.initProgressBars = function() {
    // 특성 중요도 진행률 바
    const rankItems = document.querySelectorAll('.rank-item');
    
    rankItems.forEach((item, index) => {
        const impact = item.querySelector('.rank-impact');
        if (impact) {
            const percentage = this.parseNumber(impact.textContent);
            
            // 진행률 바 생성
            const progressBar = document.createElement('div');
            progressBar.className = 'rank-progress';
            progressBar.innerHTML = `<div class="rank-progress-fill"></div>`;
            
            item.appendChild(progressBar);
            
            // 애니메이션
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const fill = progressBar.querySelector('.rank-progress-fill');
                        setTimeout(() => {
                            fill.style.width = percentage + '%';
                        }, index * 200); // 순차적 애니메이션
                    }
                });
            }, { threshold: 0.5 });
            
            observer.observe(item);
        }
    });
};

// ==================== 툴팁 시스템 ====================
Dashboard.initTooltips = function() {
    // 툴팁이 필요한 요소들
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        let tooltip = null;
        
        element.addEventListener('mouseenter', (e) => {
            const content = element.getAttribute('data-tooltip');
            if (!content) return;
            
            tooltip = document.createElement('div');
            tooltip.className = 'custom-tooltip';
            tooltip.textContent = content;
            document.body.appendChild(tooltip);
            
            // 위치 계산
            const rect = element.getBoundingClientRect();
            tooltip.style.left = rect.left + rect.width / 2 + 'px';
            tooltip.style.top = rect.top - 10 + 'px';
            
            // 애니메이션
            setTimeout(() => tooltip.classList.add('show'), 10);
        });
        
        element.addEventListener('mouseleave', () => {
            if (tooltip) {
                tooltip.classList.remove('show');
                setTimeout(() => {
                    if (tooltip && tooltip.parentNode) {
                        tooltip.parentNode.removeChild(tooltip);
                    }
                }, 200);
            }
        });
    });
};

// ==================== 스크롤 애니메이션 ====================
Dashboard.initScrollAnimations = function() {
    const animationElements = document.querySelectorAll('.hero-stats, .player-types-grid, .insights-grid');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                
                // 자식 요소들 순차 애니메이션
                const children = entry.target.children;
                Array.from(children).forEach((child, index) => {
                    setTimeout(() => {
                        child.classList.add('animate-child');
                    }, index * 100);
                });
            }
        });
    }, { 
        threshold: 0.2,
        rootMargin: '0px 0px -50px 0px'
    });
    
    animationElements.forEach(el => observer.observe(el));
};

// ==================== 반응형 대시보드 ====================
Dashboard.handleResize = function() {
    // 차트 크기 조정
    this.charts.forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
    
    // 카드 레이아웃 재계산
    if (App.isMobile) {
        document.querySelectorAll('.player-type-card').forEach(card => {
            card.style.transform = ''; // 모바일에서는 3D 효과 제거
        });
    }
};

// ==================== 성능 최적화 ====================
Dashboard.optimize = function() {
    // 불필요한 애니메이션 정리
    if (this.animationRunning && !this.isVisible()) {
        this.pauseAnimations();
    }
};

Dashboard.isVisible = function() {
    return document.visibilityState === 'visible';
};

Dashboard.pauseAnimations = function() {
    document.querySelectorAll('.animated').forEach(el => {
        el.style.animationPlayState = 'paused';
    });
};

Dashboard.resumeAnimations = function() {
    document.querySelectorAll('.animated').forEach(el => {
        el.style.animationPlayState = 'running';
    });
};

// ==================== 이벤트 리스너 ====================
window.addEventListener('resize', Dashboard.handleResize.bind(Dashboard));
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        Dashboard.resumeAnimations();
    } else {
        Dashboard.pauseAnimations();
    }
});

// ==================== CSS 스타일 추가 ====================
const dashboardStyles = `
/* 진행률 바 스타일 */
.rank-progress {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 8px;
}

.rank-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff6b35, #ff8f65);
    width: 0%;
    transition: width 1.5s ease-out;
    border-radius: 2px;
}

/* 툴팁 스타일 */
.custom-tooltip {
    position: absolute;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    z-index: 10000;
    transform: translateX(-50%) translateY(-100%);
    opacity: 0;
    transition: opacity 0.2s ease;
    white-space: nowrap;
}

.custom-tooltip.show {
    opacity: 1;
}

.custom-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: rgba(0, 0, 0, 0.9);
}

/* 카드 애니메이션 */
.player-type-card, .analysis-card, .insight-card {
    transition: all 0.3s ease;
    background-size: 100% 100%;
    background-transition: background-size 0.3s ease;
}

/* 스크롤 애니메이션 */
.animate-in {
    animation: fadeInUp 0.6s ease-out;
}

.animate-child {
    animation: slideInUp 0.4s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 호버 효과 강화 */
.btn {
    transition: all 0.3s ease;
}

.type-icon {
    transition: all 0.3s ease;
}

/* 모바일 최적화 */
@media (max-width: 768px) {
    .player-type-card, .analysis-card, .insight-card {
        transform: none !important;
    }
    
    .custom-tooltip {
        font-size: 14px;
        padding: 10px 14px;
    }
}

/* 성능 최적화 */
.player-type-card, .analysis-card, .insight-card {
    will-change: transform;
    backface-visibility: hidden;
}

/* 접근성 개선 */
@media (prefers-reduced-motion: reduce) {
    .rank-progress-fill,
    .animate-in,
    .animate-child,
    .player-type-card,
    .analysis-card,
    .insight-card {
        animation: none !important;
        transition: none !important;
    }
}
`;

// 스타일 추가
const dashboardStyleSheet = document.createElement('style');
dashboardStyleSheet.textContent = dashboardStyles;
document.head.appendChild(dashboardStyleSheet);

// ==================== 전역 접근 ====================
window.Dashboard = Dashboard;
