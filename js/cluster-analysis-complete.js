/**
 * PUBG Cluster Analysis Page JavaScript - Complete Module
 * 이벤트 리스너, 탭 기능, 모달 처리 등 완성 모듈
 */

// ===============================
// 데이터 검증 함수
// ===============================
function validateClusterData() {
    if (!CLUSTER_DATA || !CLUSTER_DATA.clusters || !Array.isArray(CLUSTER_DATA.clusters)) {
        console.error('❌ 클러스터 데이터가 올바르지 않습니다.');
        return false;
    }
    
    if (CLUSTER_DATA.clusters.length === 0) {
        console.error('❌ 클러스터 데이터가 비어있습니다.');
        return false;
    }
    
    console.log('✅ 클러스터 데이터 검증 완료:', CLUSTER_DATA.clusters.length + '개 클러스터');
    return true;
}

// ===============================
// 이벤트 리스너 설정
// ===============================
function setupEventListeners() {
    console.log('🔧 이벤트 리스너 설정 시작...');
    
    // 클러스터 필터 체크박스
    setupClusterFilters();
    
    // 필터 액션 버튼들
    setupFilterActions();
    
    // 클러스터 상세보기 버튼들
    setupDetailButtons();
    
    // 모달 관련 이벤트
    setupModalEvents();
    
    // 내보내기 버튼들
    setupExportButtons();
    
    // 탭 내비게이션
    setupTabNavigation();
    
    console.log('✅ 모든 이벤트 리스너 설정 완료');
}

function setupClusterFilters() {
    const filterCheckboxes = document.querySelectorAll('.cluster-filter');
    
    filterCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const clusterId = parseInt(this.value);
            const isChecked = this.checked;
            
            console.log(`클러스터 ${clusterId} 필터: ${isChecked ? '활성화' : '비활성화'}`);
            
            // 실시간 차트 업데이트
            updateFilteredVisualization();
            
            // 체크박스 상태에 따른 UI 업데이트
            updateClusterCardVisibility(clusterId, isChecked);
        });
    });
}

function setupFilterActions() {
    // 전체 선택 버튼
    const selectAllBtn = document.getElementById('selectAllClusters');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', function() {
            const checkboxes = document.querySelectorAll('.cluster-filter');
            checkboxes.forEach(checkbox => {
                checkbox.checked = true;
                checkbox.dispatchEvent(new Event('change'));
            });
            showNotification('모든 클러스터가 선택되었습니다.', 'info');
        });
    }
    
    // 전체 해제 버튼
    const clearAllBtn = document.getElementById('clearAllClusters');
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', function() {
            const checkboxes = document.querySelectorAll('.cluster-filter');
            checkboxes.forEach(checkbox => {
                checkbox.checked = false;
                checkbox.dispatchEvent(new Event('change'));
            });
            showNotification('모든 클러스터 선택이 해제되었습니다.', 'info');
        });
    }
    
    // 선택된 클러스터 비교 버튼
    const compareBtn = document.getElementById('compareSelected');
    if (compareBtn) {
        compareBtn.addEventListener('click', function() {
            const selectedClusters = getSelectedClusters();
            if (selectedClusters.length < 2) {
                showNotification('비교하려면 최소 2개 클러스터를 선택하세요.', 'warning');
                return;
            }
            
            // 비교 탭으로 이동
            switchToTab('comparison');
            updateComparisonChart(selectedClusters);
            showNotification(`${selectedClusters.length}개 클러스터 비교가 업데이트되었습니다.`, 'success');
        });
    }
}

function setupDetailButtons() {
    document.addEventListener('click', function(e) {
        if (e.target.closest('.cluster-detail-btn')) {
            const btn = e.target.closest('.cluster-detail-btn');
            const clusterId = btn.getAttribute('data-cluster');
            
            console.log('클러스터 상세보기:', clusterId);
            showClusterDetailModal(clusterId);
        }
    });
}

function setupModalEvents() {
    const modal = document.getElementById('clusterDetailModal');
    const closeBtn = document.getElementById('closeModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    
    // 모달 닫기 버튼들
    [closeBtn, closeModalBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', function() {
                hideClusterDetailModal();
            });
        }
    });
    
    // 모달 배경 클릭으로 닫기
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                hideClusterDetailModal();
            }
        });
    }
    
    // ESC 키로 모달 닫기
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
            hideClusterDetailModal();
        }
    });
}

function setupExportButtons() {
    const exportCSVBtn = document.getElementById('exportCSV');
    if (exportCSVBtn) {
        exportCSVBtn.addEventListener('click', function() {
            console.log('CSV 내보내기 시작...');
            exportToCSV();
        });
    }
    
    const exportJSONBtn = document.getElementById('exportJSON');
    if (exportJSONBtn) {
        exportJSONBtn.addEventListener('click', function() {
            console.log('JSON 내보내기 시작...');
            exportToJSON();
        });
    }
}

function setupTabNavigation() {
    const tabLinks = document.querySelectorAll('.tab-link');
    
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetTab = this.getAttribute('href').substring(1);
            switchToTab(targetTab);
        });
    });
}

// ===============================
// 탭 기능 구현
// ===============================
function initializeTabs() {
    const firstTab = document.querySelector('.tab-link.active');
    if (firstTab) {
        const targetTab = firstTab.getAttribute('href').substring(1);
        switchToTab(targetTab);
    }
}

function switchToTab(tabId) {
    console.log('탭 전환:', tabId);
    
    // 모든 탭 링크에서 active 클래스 제거
    document.querySelectorAll('.tab-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // 모든 탭 콘텐츠 숨기기
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // 선택된 탭 활성화
    const targetTabLink = document.querySelector(`[href="#${tabId}"]`);
    const targetTabContent = document.getElementById(tabId);
    
    if (targetTabLink) {
        targetTabLink.classList.add('active');
    }
    
    if (targetTabContent) {
        targetTabContent.classList.add('active');
        
        // 탭별 특별 처리
        handleTabSpecificActions(tabId);
    }
}

function handleTabSpecificActions(tabId) {
    switch(tabId) {
        case 'characteristics':
            // 레이더 차트 리사이즈
            setTimeout(() => {
                Object.values(radarCharts).forEach(chart => {
                    if (chart) chart.resize();
                });
            }, 100);
            break;
            
        case 'comparison':
            // 비교 차트 업데이트
            setTimeout(() => {
                const selectedClusters = getSelectedClusters();
                if (selectedClusters.length > 0) {
                    updateComparisonChart(selectedClusters);
                }
                if (comparisonChart) comparisonChart.resize();
                if (correlationChart) correlationChart.resize();
            }, 100);
            break;
            
        case 'statistics':
            // 통계 테이블 새로고침
            generateStatisticsTable();
            break;
    }
}

// ===============================
// 클러스터 필터링 관련 함수들
// ===============================
function getSelectedClusters() {
    const selectedCheckboxes = document.querySelectorAll('.cluster-filter:checked');
    return Array.from(selectedCheckboxes).map(cb => parseInt(cb.value));
}

function updateFilteredVisualization() {
    const selectedClusters = getSelectedClusters();
    
    // 분포 차트 업데이트
    updateDistributionChart(selectedClusters);
    
    // 현재 비교 탭이 활성화되어 있다면 비교 차트도 업데이트
    if (document.getElementById('comparison').classList.contains('active')) {
        updateComparisonChart(selectedClusters);
    }
}

function updateDistributionChart(selectedClusters) {
    if (!clusterDistributionChart) return;
    
    const filteredData = CLUSTER_DATA.clusters.filter(cluster => 
        selectedClusters.includes(cluster.id)
    );
    
    clusterDistributionChart.data.labels = filteredData.map(cluster => cluster.name);
    clusterDistributionChart.data.datasets[0].data = filteredData.map(cluster => cluster.percentage);
    clusterDistributionChart.data.datasets[0].backgroundColor = filteredData.map(cluster => cluster.color);
    
    clusterDistributionChart.update();
}

function updateClusterCardVisibility(clusterId, isVisible) {
    const card = document.querySelector(`[data-cluster="${clusterId}"]`);
    if (card) {
        card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        if (isVisible) {
            card.style.opacity = '1';
            card.style.transform = 'scale(1)';
            card.style.pointerEvents = 'auto';
        } else {
            card.style.opacity = '0.3';
            card.style.transform = 'scale(0.95)';
            card.style.pointerEvents = 'none';
        }
    }
}

// ===============================
// 모달 기능 구현
// ===============================
function showClusterDetailModal(clusterId) {
    const modal = document.getElementById('clusterDetailModal');
    const modalTitle = document.getElementById('modalClusterTitle');
    const modalContent = document.getElementById('modalClusterContent');
    
    if (!modal || !modalContent) {
        console.error('모달 요소를 찾을 수 없습니다.');
        return;
    }
    
    let cluster;
    
    // Explorer 통합 모달 처리
    if (clusterId === 'explorer') {
        cluster = createExplorerCombinedData();
    } else {
        cluster = CLUSTER_DATA.clusters.find(c => c.id === parseInt(clusterId));
    }
    
    if (!cluster) {
        console.error('클러스터를 찾을 수 없습니다:', clusterId);
        showNotification('클러스터 정보를 찾을 수 없습니다.', 'error');
        return;
    }
    
    // 모달 내용 업데이트
    if (modalTitle) {
        modalTitle.textContent = cluster.name + ' - 상세 정보';
    }
    
    modalContent.innerHTML = createDetailModalContent(cluster);
    
    // 모달 표시
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    
    // 모달 내용 애니메이션
    setTimeout(() => {
        const modalDialog = modal.querySelector('.modal');
        if (modalDialog) {
            modalDialog.style.transform = 'scale(1)';
            modalDialog.style.opacity = '1';
        }
    }, 10);
}

function hideClusterDetailModal() {
    const modal = document.getElementById('clusterDetailModal');
    if (!modal) return;
    
    const modalDialog = modal.querySelector('.modal');
    if (modalDialog) {
        modalDialog.style.transform = 'scale(0.9)';
        modalDialog.style.opacity = '0';
    }
    
    setTimeout(() => {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }, 300);
}

function createExplorerCombinedData() {
    const explorerClusters = CLUSTER_DATA.clusters.filter(c => c.type === 'explorer');
    
    return {
        id: 'explorer-combined',
        name: 'Explorer (통합)',
        type: 'explorer',
        count: explorerClusters.reduce((sum, c) => sum + c.count, 0),
        percentage: explorerClusters.reduce((sum, c) => sum + c.percentage, 0),
        color: '#FF9800',
        icon: 'fas fa-route',
        description: '맵 탐색과 이동을 중시하는 5가지 유형의 플레이어들',
        subtypes: explorerClusters,
        characteristics: [
            '전체 플레이어의 약 50% 차지',
            '이동 거리 최대화 추구',
            '맵 탐색 및 포지셔닝 중시',
            '장거리 교전 선호',
            '다양한 하위 유형 존재'
        ]
    };
}

// ===============================
// 고급 기능 초기화 (다른 파일과 연동)
// ===============================
function initializeAdvancedFeatures() {
    console.log('🚀 고급 기능 초기화...');
    
    // 다른 JavaScript 파일의 고급 기능이 있다면 초기화
    if (typeof initializeClusterAdvancedAnalysis === 'function') {
        initializeClusterAdvancedAnalysis();
    }
    
    if (typeof setupClusterInteractivity === 'function') {
        setupClusterInteractivity();
    }
    
    // 툴팁 초기화
    initializeTooltips();
    
    // 키보드 단축키 설정
    setupKeyboardShortcuts();
}

function initializeTooltips() {
    // 간단한 툴팁 구현
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function(e) {
            showTooltip(e.target, e.target.getAttribute('data-tooltip'));
        });
        
        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        z-index: 10000;
        pointer-events: none;
        white-space: nowrap;
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.custom-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + 숫자 키로 탭 전환
        if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '3') {
            e.preventDefault();
            
            const tabMap = {
                '1': 'characteristics',
                '2': 'comparison', 
                '3': 'statistics'
            };
            
            const targetTab = tabMap[e.key];
            if (targetTab) {
                switchToTab(targetTab);
                showNotification(`${targetTab} 탭으로 이동했습니다.`, 'info');
            }
        }
        
        // Ctrl/Cmd + A로 모든 클러스터 선택
        if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
            e.preventDefault();
            document.getElementById('selectAllClusters')?.click();
        }
        
        // Ctrl/Cmd + D로 모든 클러스터 선택 해제
        if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
            e.preventDefault();
            document.getElementById('clearAllClusters')?.click();
        }
    });
}

console.log('✅ 클러스터 분석 완성 모듈 로드 완료');
