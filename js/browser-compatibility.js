/**
 * 브라우저 호환성 체크 및 폴백 처리
 * PUBG 네비게이션 컴포넌트의 크로스 브라우저 지원
 */

window.BrowserCompatibilityChecker = {
    
    /**
     * 브라우저 정보 및 지원 기능 확인
     */
    checkBrowserSupport() {
        console.group('🌐 Browser Compatibility Check');
        
        // 브라우저 정보
        const userAgent = navigator.userAgent;
        let browserName = 'Unknown';
        let browserVersion = 'Unknown';
        
        // 브라우저 감지
        if (userAgent.includes('Chrome') && !userAgent.includes('Edg')) {
            browserName = 'Chrome';
            browserVersion = userAgent.match(/Chrome\/([0-9.]+)/)?.[1] || 'Unknown';
        } else if (userAgent.includes('Firefox')) {
            browserName = 'Firefox';
            browserVersion = userAgent.match(/Firefox\/([0-9.]+)/)?.[1] || 'Unknown';
        } else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
            browserName = 'Safari';
            browserVersion = userAgent.match(/Version\/([0-9.]+)/)?.[1] || 'Unknown';
        } else if (userAgent.includes('Edg')) {
            browserName = 'Edge';
            browserVersion = userAgent.match(/Edg\/([0-9.]+)/)?.[1] || 'Unknown';
        }
        
        console.log(`브라우저: ${browserName} ${browserVersion}`);
        console.log(`User Agent: ${userAgent}`);
        
        // 필수 기능 지원 확인
        const features = {
            'CSS Variables': CSS.supports('color', 'var(--color)'),
            'CSS Grid': CSS.supports('display', 'grid'),
            'CSS Flexbox': CSS.supports('display', 'flex'),
            'CSS Backdrop Filter': CSS.supports('backdrop-filter', 'blur(10px)'),
            'CSS Custom Properties': !!window.CSS?.supports,
            'ES6 Classes': typeof class {} === 'function',
            'Arrow Functions': (() => true)(),
            'Template Literals': typeof `template` === 'string',
            'Fetch API': typeof fetch !== 'undefined',
            'Local Storage': typeof localStorage !== 'undefined',
            'Session Storage': typeof sessionStorage !== 'undefined'
        };
        
        console.log('\n지원 기능:');
        Object.entries(features).forEach(([feature, supported]) => {
            console.log(`  ${supported ? '✅' : '❌'} ${feature}`);
        });
        
        // 호환성 점수 계산
        const supportedCount = Object.values(features).filter(Boolean).length;
        const totalCount = Object.keys(features).length;
        const compatibilityScore = Math.round((supportedCount / totalCount) * 100);
        
        console.log(`\n호환성 점수: ${compatibilityScore}% (${supportedCount}/${totalCount})`);
        
        if (compatibilityScore >= 90) {
            console.log('🎉 완벽한 호환성');
        } else if (compatibilityScore >= 70) {
            console.log('⚠️ 기본 기능은 작동하지만 일부 고급 기능에 제한이 있을 수 있습니다');
        } else {
            console.log('❌ 호환성 문제가 있을 수 있습니다');
        }
        
        console.groupEnd();
        
        return {
            browserName,
            browserVersion,
            features,
            compatibilityScore
        };
    },
    
    /**
     * CSS 기능 폴백 적용
     */
    applyFallbacks() {
        console.group('🔧 CSS Fallbacks');
        
        // backdrop-filter 지원 확인 및 폴백
        if (!CSS.supports('backdrop-filter', 'blur(10px)')) {
            console.log('⚠️ backdrop-filter 미지원 - 폴백 적용');
            
            // 폴백 스타일 동적 추가
            const style = document.createElement('style');
            style.textContent = `
                .header {
                    background: rgba(0, 0, 0, 0.95) !important;
                    backdrop-filter: none !important;
                }
                .nav-menu {
                    background: rgba(0, 0, 0, 0.95) !important;
                    backdrop-filter: none !important;
                }
            `;
            document.head.appendChild(style);
        }
        
        // CSS Grid 폴백
        if (!CSS.supports('display', 'grid')) {
            console.log('⚠️ CSS Grid 미지원 - Flexbox 폴백 적용');
            
            const style = document.createElement('style');
            style.textContent = `
                .grid {
                    display: flex !important;
                    flex-wrap: wrap !important;
                }
                .grid > * {
                    flex: 1 1 300px !important;
                }
            `;
            document.head.appendChild(style);
        }
        
        // CSS Variables 폴백
        if (!CSS.supports('color', 'var(--color)')) {
            console.log('⚠️ CSS Variables 미지원 - 하드코딩된 값 사용');
            
            const style = document.createElement('style');
            style.textContent = `
                .header {
                    background: rgba(255, 255, 255, 0.05) !important;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
                }
                .nav-link {
                    color: #e0e0e0 !important;
                }
                .nav-link:hover,
                .nav-link.active {
                    color: #ff6b35 !important;
                    background: rgba(255, 107, 53, 0.1) !important;
                }
            `;
            document.head.appendChild(style);
        }
        
        console.groupEnd();
    },
    
    /**
     * 모바일 환경 감지 및 최적화
     */
    optimizeForMobile() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        
        console.group('📱 Mobile Optimization');
        console.log(`모바일 디바이스: ${isMobile ? '✅' : '❌'}`);
        console.log(`터치 지원: ${isTouch ? '✅' : '❌'}`);
        
        if (isMobile || isTouch) {
            // 터치 친화적인 스타일 적용
            const style = document.createElement('style');
            style.textContent = `
                .nav-link {
                    min-height: 44px !important;
                    display: flex !important;
                    align-items: center !important;
                }
                .nav-toggle {
                    min-width: 44px !important;
                    min-height: 44px !important;
                }
                .test-btn {
                    min-height: 44px !important;
                    padding: 12px 20px !important;
                }
            `;
            document.head.appendChild(style);
            
            // 터치 이벤트 최적화
            document.addEventListener('touchstart', function() {}, { passive: true });
            
            console.log('✅ 모바일 최적화 적용됨');
        }
        
        console.groupEnd();
        
        return { isMobile, isTouch };
    },
    
    /**
     * 성능 모니터링
     */
    monitorPerformance() {
        console.group('⚡ Performance Monitor');
        
        // DOM 로딩 시간
        const navigationStart = performance.timing.navigationStart;
        const domLoaded = performance.timing.domContentLoadedEventEnd;
        const pageLoaded = performance.timing.loadEventEnd;
        
        const domLoadTime = domLoaded - navigationStart;
        const pageLoadTime = pageLoaded - navigationStart;
        
        console.log(`DOM 로딩 시간: ${domLoadTime}ms`);
        console.log(`페이지 로딩 시간: ${pageLoadTime}ms`);
        
        // 메모리 사용량 (지원되는 브라우저에서만)
        if (performance.memory) {
            const memory = performance.memory;
            console.log('메모리 사용량:');
            console.log(`  사용 중: ${Math.round(memory.usedJSHeapSize / 1024 / 1024)}MB`);
            console.log(`  할당됨: ${Math.round(memory.totalJSHeapSize / 1024 / 1024)}MB`);
            console.log(`  한계: ${Math.round(memory.jsHeapSizeLimit / 1024 / 1024)}MB`);
        }
        
        // 렌더링 성능 체크
        let frameCount = 0;
        let lastTime = performance.now();
        
        function measureFPS() {
            const currentTime = performance.now();
            frameCount++;
            
            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                console.log(`FPS: ${fps}`);
                
                frameCount = 0;
                lastTime = currentTime;
                
                // 성능 경고
                if (fps < 30) {
                    console.warn('⚠️ 낮은 FPS 감지 - 성능 최적화가 필요할 수 있습니다');
                }
            }
            
            requestAnimationFrame(measureFPS);
        }
        
        // 3초간 FPS 측정
        setTimeout(() => {
            requestAnimationFrame(measureFPS);
        }, 1000);
        
        console.groupEnd();
    },
    
    /**
     * 전체 호환성 체크 실행
     */
    runFullCompatibilityCheck() {
        console.clear();
        console.log('🔧 브라우저 호환성 및 성능 체크 시작');
        console.log('=' .repeat(50));
        
        const browserInfo = this.checkBrowserSupport();
        this.applyFallbacks();
        const mobileInfo = this.optimizeForMobile();
        this.monitorPerformance();
        
        console.log('=' .repeat(50));
        console.log('✅ 호환성 체크 완료!');
        
        return {
            browser: browserInfo,
            mobile: mobileInfo,
            timestamp: new Date().toISOString()
        };
    }
};

// 전역 함수로 등록
window.checkBrowserCompat = () => BrowserCompatibilityChecker.runFullCompatibilityCheck();

// 페이지 로드 시 자동 실행
window.addEventListener('load', function() {
    console.log('🔧 브라우저 호환성 체커가 로드되었습니다.');
    console.log('💡 사용법: checkBrowserCompat() - 전체 호환성 체크');
    
    // 자동으로 폴백 적용
    BrowserCompatibilityChecker.applyFallbacks();
    BrowserCompatibilityChecker.optimizeForMobile();
});
