/**
 * 네비게이션 디버깅 도구
 * 브라우저 콘솔에서 네비게이션 상태를 확인할 수 있는 유틸리티 함수들
 */

window.NavigationDebugger = {
    
    /**
     * 네비게이션 전체 상태 확인
     */
    checkNavigationStatus() {
        console.group('🔍 Navigation Status Check');
        
        // 1. NavigationComponent 클래스 존재 여부
        console.log('1. NavigationComponent 클래스:', typeof NavigationComponent !== 'undefined' ? '✅ 존재' : '❌ 없음');
        
        // 2. initNavigation 함수 존재 여부
        console.log('2. initNavigation 함수:', typeof initNavigation !== 'undefined' ? '✅ 존재' : '❌ 없음');
        
        // 3. 네비게이션 요소 존재 여부
        const header = document.querySelector('.header');
        const navMenu = document.querySelector('.nav-menu');
        const navToggle = document.querySelector('.nav-toggle');
        
        console.log('3. Header 요소:', header ? '✅ 존재' : '❌ 없음');
        console.log('4. Nav Menu 요소:', navMenu ? '✅ 존재' : '❌ 없음');
        console.log('5. Nav Toggle 요소:', navToggle ? '✅ 존재' : '❌ 없음');
        
        // 4. 현재 페이지 정보
        if (typeof NavigationComponent !== 'undefined') {
            const nav = new NavigationComponent();
            const info = nav.getInfo();
            console.log('6. 현재 페이지 정보:', info);
        }
        
        // 5. CSS 변수 확인
        const rootStyles = getComputedStyle(document.documentElement);
        const primaryColor = rootStyles.getPropertyValue('--color-primary');
        const headerHeight = rootStyles.getPropertyValue('--header-height');
        
        console.log('7. CSS 변수 확인:');
        console.log('   - Primary Color:', primaryColor || '❌ 없음');
        console.log('   - Header Height:', headerHeight || '❌ 없음');
        
        console.groupEnd();
    },
    
    /**
     * 네비게이션 링크 상태 확인
     */
    checkNavigationLinks() {
        console.group('🔗 Navigation Links Check');
        
        const navLinks = document.querySelectorAll('.nav-link');
        console.log(`발견된 네비게이션 링크: ${navLinks.length}개`);
        
        navLinks.forEach((link, index) => {
            const href = link.getAttribute('href');
            const isActive = link.classList.contains('active');
            const text = link.textContent.trim();
            
            console.log(`${index + 1}. "${text}"`);
            console.log(`   - href: ${href}`);
            console.log(`   - active: ${isActive ? '✅' : '❌'}`);
            console.log(`   - classes: ${link.className}`);
        });
        
        console.groupEnd();
    },
    
    /**
     * 반응형 상태 확인
     */
    checkResponsiveState() {
        console.group('📱 Responsive State Check');
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        console.log(`화면 크기: ${width}x${height}px`);
        
        // 반응형 브레이크포인트 확인
        if (width <= 480) {
            console.log('📱 상태: Extra Small (XS)');
        } else if (width <= 768) {
            console.log('📱 상태: Mobile');
        } else if (width <= 992) {
            console.log('💻 상태: Tablet');
        } else {
            console.log('🖥️ 상태: Desktop');
        }
        
        // 모바일 메뉴 상태
        const navMenu = document.querySelector('.nav-menu');
        const navToggle = document.querySelector('.nav-toggle');
        
        if (navMenu && navToggle) {
            const menuIsActive = navMenu.classList.contains('active');
            const toggleIsActive = navToggle.classList.contains('active');
            
            console.log('모바일 메뉴 상태:');
            console.log(`  - Menu active: ${menuIsActive ? '✅' : '❌'}`);
            console.log(`  - Toggle active: ${toggleIsActive ? '✅' : '❌'}`);
            
            // CSS 미디어 쿼리 상태
            const navMenuStyle = getComputedStyle(navMenu);
            const navToggleStyle = getComputedStyle(navToggle);
            
            console.log('CSS 상태:');
            console.log(`  - Menu display: ${navMenuStyle.display}`);
            console.log(`  - Toggle display: ${navToggleStyle.display}`);
        }
        
        console.groupEnd();
    },
    
    /**
     * CSS 로딩 상태 확인
     */
    checkCSSLoading() {
        console.group('🎨 CSS Loading Check');
        
        const links = document.querySelectorAll('link[rel="stylesheet"]');
        console.log(`로드된 CSS 파일: ${links.length}개`);
        
        links.forEach((link, index) => {
            const href = link.getAttribute('href');
            const isLoaded = link.sheet !== null;
            
            console.log(`${index + 1}. ${href}`);
            console.log(`   - 로드 상태: ${isLoaded ? '✅' : '❌'}`);
        });
        
        console.groupEnd();
    },
    
    /**
     * 이벤트 리스너 테스트
     */
    testEventListeners() {
        console.group('🎯 Event Listeners Test');
        
        const navToggle = document.querySelector('.nav-toggle');
        const navLinks = document.querySelectorAll('.nav-link');
        
        if (navToggle) {
            console.log('모바일 토글 버튼 테스트...');
            navToggle.click();
            
            setTimeout(() => {
                const navMenu = document.querySelector('.nav-menu');
                const isActive = navMenu?.classList.contains('active');
                console.log(`토글 결과: ${isActive ? '✅ 메뉴 열림' : '❌ 메뉴 닫힘'}`);
                
                // 다시 닫기
                navToggle.click();
            }, 100);
        }
        
        console.log(`네비게이션 링크: ${navLinks.length}개 발견`);
        
        console.groupEnd();
    },
    
    /**
     * 전체 진단 실행
     */
    runFullDiagnostic() {
        console.clear();
        console.log('🚀 PUBG 네비게이션 전체 진단 시작');
        console.log('=' .repeat(50));
        
        this.checkNavigationStatus();
        this.checkNavigationLinks();
        this.checkResponsiveState();
        this.checkCSSLoading();
        
        console.log('=' .repeat(50));
        console.log('✅ 진단 완료! 문제가 발견되면 위의 결과를 확인하세요.');
        
        // 요약 정보
        const header = document.querySelector('.header');
        const navLinks = document.querySelectorAll('.nav-link');
        const activeLink = document.querySelector('.nav-link.active');
        
        console.log('\n📋 요약:');
        console.log(`- 네비게이션: ${header ? '정상' : '문제 있음'}`);
        console.log(`- 링크 수: ${navLinks.length}개`);
        console.log(`- 활성 링크: ${activeLink ? activeLink.textContent.trim() : '없음'}`);
        console.log(`- 화면 크기: ${window.innerWidth}x${window.innerHeight}px`);
    }
};

// 콘솔에서 쉽게 사용할 수 있도록 전역 함수로 등록
window.checkNav = () => NavigationDebugger.runFullDiagnostic();
window.testNav = () => NavigationDebugger.testEventListeners();

// 페이지 로드 완료 후 자동 실행 (선택적)
window.addEventListener('load', function() {
    console.log('🔧 네비게이션 디버깅 도구가 로드되었습니다.');
    console.log('💡 사용법:');
    console.log('  - checkNav(): 전체 진단 실행');
    console.log('  - testNav(): 이벤트 리스너 테스트');
    console.log('  - NavigationDebugger.checkNavigationStatus(): 상태 확인');
    console.log('  - NavigationDebugger.checkResponsiveState(): 반응형 확인');
});
