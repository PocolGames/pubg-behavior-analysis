/**
 * PUBG 플레이어 행동 분석 - Tabs 유틸리티
 * 탭 기능을 위한 JavaScript 모듈
 */

class TabManager {
    constructor(tabContainer = '.tabs') {
        this.tabContainer = typeof tabContainer === 'string' 
            ? document.querySelector(tabContainer) 
            : tabContainer;
        
        this.tabLinks = null;
        this.tabContents = null;
        this.activeTab = null;
        
        this.init();
    }

    /**
     * 탭 시스템 초기화
     */
    init() {
        if (!this.tabContainer) {
            console.warn('⚠️ 탭 컨테이너를 찾을 수 없습니다.');
            return;
        }

        this.tabLinks = this.tabContainer.querySelectorAll('.tab-link');
        this.tabContents = document.querySelectorAll('.tab-content');

        if (this.tabLinks.length === 0 || this.tabContents.length === 0) {
            console.warn('⚠️ 탭 링크 또는 탭 콘텐츠를 찾을 수 없습니다.');
            return;
        }

        this.setupEventListeners();
        this.setInitialTab();
        
        console.log('✅ 탭 시스템 초기화 완료');
    }

    /**
     * 이벤트 리스너 설정
     */
    setupEventListeners() {
        this.tabLinks.forEach((link, index) => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                this.showTab(targetId);
            });

            // 키보드 접근성
            link.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const targetId = link.getAttribute('href').substring(1);
                    this.showTab(targetId);
                }
            });
        });
    }

    /**
     * 초기 탭 설정
     */
    setInitialTab() {
        // URL 해시가 있으면 해당 탭 활성화
        const hash = window.location.hash.substring(1);
        if (hash && document.getElementById(hash)) {
            this.showTab(hash);
        } else {
            // 첫 번째 탭 활성화
            const firstTab = this.tabLinks[0];
            if (firstTab) {
                const targetId = firstTab.getAttribute('href').substring(1);
                this.showTab(targetId);
            }
        }
    }

    /**
     * 특정 탭 표시
     * @param {string} tabId - 표시할 탭의 ID
     */
    showTab(tabId) {
        // 모든 탭 링크에서 active 클래스 제거
        this.tabLinks.forEach(link => {
            link.classList.remove('active');
            link.setAttribute('aria-selected', 'false');
        });

        // 모든 탭 콘텐츠 숨기기
        this.tabContents.forEach(content => {
            content.classList.remove('active');
            content.setAttribute('aria-hidden', 'true');
        });

        // 선택된 탭 링크 활성화
        const activeLink = this.tabContainer.querySelector(`[href="#${tabId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
            activeLink.setAttribute('aria-selected', 'true');
        }

        // 선택된 탭 콘텐츠 표시
        const activeContent = document.getElementById(tabId);
        if (activeContent) {
            activeContent.classList.add('active');
            activeContent.setAttribute('aria-hidden', 'false');
            this.activeTab = tabId;

            // 콘텐츠 로딩 완료 이벤트 발생
            this.triggerTabChangeEvent(tabId);
        }

        // URL 해시 업데이트 (히스토리에 추가하지 않고)
        if (history.replaceState) {
            history.replaceState(null, null, `#${tabId}`);
        }
    }

    /**
     * 탭 변경 이벤트 발생
     * @param {string} tabId - 변경된 탭의 ID
     */
    triggerTabChangeEvent(tabId) {
        const event = new CustomEvent('tabchange', {
            detail: {
                tabId: tabId,
                tabElement: document.getElementById(tabId)
            }
        });
        
        document.dispatchEvent(event);
    }

    /**
     * 다음 탭으로 이동
     */
    nextTab() {
        const currentIndex = Array.from(this.tabLinks).findIndex(link => 
            link.classList.contains('active')
        );
        
        const nextIndex = (currentIndex + 1) % this.tabLinks.length;
        const nextLink = this.tabLinks[nextIndex];
        
        if (nextLink) {
            const targetId = nextLink.getAttribute('href').substring(1);
            this.showTab(targetId);
        }
    }

    /**
     * 이전 탭으로 이동
     */
    previousTab() {
        const currentIndex = Array.from(this.tabLinks).findIndex(link => 
            link.classList.contains('active')
        );
        
        const prevIndex = currentIndex === 0 ? this.tabLinks.length - 1 : currentIndex - 1;
        const prevLink = this.tabLinks[prevIndex];
        
        if (prevLink) {
            const targetId = prevLink.getAttribute('href').substring(1);
            this.showTab(targetId);
        }
    }

    /**
     * 현재 활성 탭 ID 반환
     * @returns {string} 현재 활성 탭의 ID
     */
    getActiveTab() {
        return this.activeTab;
    }

    /**
     * 메모리 정리
     */
    destroy() {
        if (this.tabLinks) {
            this.tabLinks.forEach(link => {
                link.removeEventListener('click', this.handleTabClick);
                link.removeEventListener('keydown', this.handleTabKeydown);
            });
        }
        
        this.tabContainer = null;
        this.tabLinks = null;
        this.tabContents = null;
        this.activeTab = null;
    }
}

// 전역 탭 매니저 인스턴스
let globalTabManager = null;

// 초기화 함수
function initializeTabs(tabContainer = '.tabs') {
    if (globalTabManager) {
        globalTabManager.destroy();
    }
    
    globalTabManager = new TabManager(tabContainer);
    
    // 탭 변경 이벤트 리스너 예제
    document.addEventListener('tabchange', (e) => {
        console.log(`탭 변경됨: ${e.detail.tabId}`);
        
        // 탭별 특별한 로직이 필요한 경우 여기에 추가
        if (e.detail.tabId === 'characteristics') {
            // 특성 탭이 활성화될 때의 로직
            console.log('클러스터 특성 탭 활성화됨');
        } else if (e.detail.tabId === 'comparison') {
            // 비교 탭이 활성화될 때의 로직
            console.log('클러스터 비교 탭 활성화됨');
        } else if (e.detail.tabId === 'statistics') {
            // 통계 탭이 활성화될 때의 로직
            console.log('상세 통계 탭 활성화됨');
        }
    });
    
    return globalTabManager;
}

// DOM 로드 시 자동 초기화
document.addEventListener('DOMContentLoaded', function() {
    // 탭 컨테이너가 존재하는 경우에만 초기화
    if (document.querySelector('.tabs')) {
        initializeTabs();
        console.log('✅ 탭 시스템 자동 초기화 완료');
    }
});

// 전역 접근을 위한 export
if (typeof window !== 'undefined') {
    window.TabManager = TabManager;
    window.initializeTabs = initializeTabs;
}
