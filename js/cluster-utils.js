/**
 * PUBG Cluster Analysis Utility Functions
 * 클러스터 분석에 필요한 유틸리티 함수들
 */

// ===============================
// 데이터 검증 함수들
// ===============================

/**
 * 클러스터 데이터 검증
 */
function validateClusterData() {
    try {
        // CLUSTER_DATA 객체 존재 확인
        if (typeof CLUSTER_DATA === 'undefined') {
            console.error('❌ CLUSTER_DATA가 정의되지 않았습니다.');
            return false;
        }

        // clusters 배열 확인
        if (!CLUSTER_DATA.clusters || !Array.isArray(CLUSTER_DATA.clusters)) {
            console.error('❌ clusters 배열이 유효하지 않습니다.');
            return false;
        }

        // 각 클러스터 데이터 검증
        for (let i = 0; i < CLUSTER_DATA.clusters.length; i++) {
            const cluster = CLUSTER_DATA.clusters[i];
            
            if (!cluster.id && cluster.id !== 0) {
                console.error(`❌ 클러스터 ${i}의 id가 없습니다.`);
                return false;
            }
            
            if (!cluster.name) {
                console.error(`❌ 클러스터 ${i}의 name이 없습니다.`);
                return false;
            }
            
            if (!cluster.features || typeof cluster.features !== 'object') {
                console.error(`❌ 클러스터 ${i}의 features가 유효하지 않습니다.`);
                return false;
            }
        }

        console.log('✅ 클러스터 데이터 검증 완료');
        return true;
    } catch (error) {
        console.error('❌ 데이터 검증 중 오류:', error);
        return false;
    }
}

// ===============================
// 메시지 표시 함수들
// ===============================

/**
 * 에러 메시지 표시
 */
function showErrorMessage(message, type = 'error') {
    console.error('🚨 Error:', message);
    
    // 에러 컨테이너 찾기 또는 생성
    let errorContainer = document.getElementById('error-container');
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'error-container';
        errorContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 300px;
        `;
        document.body.appendChild(errorContainer);
    }

    // 에러 메시지 요소 생성
    const errorDiv = document.createElement('div');
    errorDiv.className = `alert alert-${type}`;
    errorDiv.style.cssText = `
        background-color: ${type === 'error' ? '#f8d7da' : '#d4edda'};
        border: 1px solid ${type === 'error' ? '#f5c6cb' : '#c3e6cb'};
        color: ${type === 'error' ? '#721c24' : '#155724'};
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: slideInRight 0.3s ease-out;
    `;
    errorDiv.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" 
                    style="background: none; border: none; font-size: 18px; cursor: pointer; margin-left: 10px;">×</button>
        </div>
    `;

    errorContainer.appendChild(errorDiv);

    // 5초 후 자동 제거
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 5000);
}

/**
 * 성공 메시지 표시
 */
function showSuccessMessage(message) {
    showErrorMessage(message, 'success');
}

// ===============================
// 초기화 함수들
// ===============================

/**
 * 차트 초기화
 */
function initializeCharts() {
    console.log('🎨 차트 초기화 시작');
    
    try {
        // Chart.js 로드 확인
        if (typeof Chart === 'undefined') {
            throw new Error('Chart.js가 로드되지 않았습니다.');
        }

        // 각 차트 생성
        createClusterDistributionChart();
        createRadarCharts();
        createComparisonChart();
        createCorrelationChart();
        
        console.log('✅ 모든 차트 초기화 완료');
    } catch (error) {
        console.error('❌ 차트 초기화 실패:', error);
        showErrorMessage('차트를 로딩하는 중 오류가 발생했습니다.');
    }
}

/**
 * 이벤트 리스너 설정
 */
function setupEventListeners() {
    console.log('🎧 이벤트 리스너 설정');
    
    // 클러스터 카드 클릭 이벤트
    document.querySelectorAll('.cluster-card').forEach(card => {
        card.addEventListener('click', function() {
            const clusterId = this.dataset.clusterId;
            if (clusterId) {
                highlightCluster(clusterId);
            }
        });
    });

    // 필터 버튼 이벤트
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const filterType = this.dataset.filter;
            if (filterType) {
                applyFilter(filterType);
            }
        });
    });

    // 윈도우 리사이즈 이벤트
    window.addEventListener('resize', debounce(resizeCharts, 300));
}

/**
 * 탭 기능 초기화
 */
function initializeTabs() {
    console.log('📑 탭 기능 초기화');
    
    // 탭 버튼 이벤트 리스너
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.dataset.tab;
            if (tabId) {
                switchTab(tabId);
            }
        });
    });
}

/**
 * 통계 테이블 생성
 */
function generateStatisticsTable() {
    console.log('📊 통계 테이블 생성');
    
    const tableContainer = document.getElementById('statistics-table');
    if (!tableContainer) return;

    const table = document.createElement('table');
    table.className = 'table table-striped';
    
    // 테이블 헤더
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>클러스터</th>
            <th>플레이어 수</th>
            <th>비율</th>
            <th>주요 특징</th>
        </tr>
    `;
    table.appendChild(thead);

    // 테이블 본문
    const tbody = document.createElement('tbody');
    CLUSTER_DATA.clusters.forEach(cluster => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <span class="cluster-badge" style="background-color: ${cluster.color};">
                    <i class="${cluster.icon}"></i> ${cluster.name}
                </span>
            </td>
            <td>${cluster.count.toLocaleString()}명</td>
            <td>${cluster.percentage}%</td>
            <td>${cluster.description}</td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    tableContainer.appendChild(table);
}

/**
 * 애니메이션 시작
 */
function startAnimations() {
    console.log('✨ 애니메이션 시작');
    
    // 카드 애니메이션
    const cards = document.querySelectorAll('.cluster-card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'all 0.5s ease-out';
            
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 50);
        }, index * 100);
    });
}

// ===============================
// 유틸리티 함수들
// ===============================

/**
 * 디바운스 유틸리티
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * 차트 리사이즈
 */
function resizeCharts() {
    console.log('📏 차트 리사이즈');
    
    // 모든 차트 리사이즈
    if (typeof clusterDistributionChart !== 'undefined' && clusterDistributionChart) {
        clusterDistributionChart.resize();
    }
    if (typeof comparisonChart !== 'undefined' && comparisonChart) {
        comparisonChart.resize();
    }
    if (typeof correlationChart !== 'undefined' && correlationChart) {
        correlationChart.resize();
    }
    
    if (typeof radarCharts !== 'undefined' && radarCharts) {
        Object.values(radarCharts).forEach(chart => {
            if (chart) chart.resize();
        });
    }
}

/**
 * 클러스터 하이라이트
 */
function highlightCluster(clusterId) {
    console.log('🎯 클러스터 하이라이트:', clusterId);
    
    // 모든 카드에서 active 클래스 제거
    document.querySelectorAll('.cluster-card').forEach(card => {
        card.classList.remove('active');
    });
    
    // 선택된 카드에 active 클래스 추가
    const selectedCard = document.querySelector(`[data-cluster-id="${clusterId}"]`);
    if (selectedCard) {
        selectedCard.classList.add('active');
    }
}

/**
 * 필터 적용
 */
function applyFilter(filterType) {
    console.log('🔍 필터 적용:', filterType);
    
    const cards = document.querySelectorAll('.cluster-card');
    
    cards.forEach(card => {
        const clusterType = card.dataset.clusterType;
        
        if (filterType === 'all' || clusterType === filterType) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

/**
 * 탭 전환
 */
function switchTab(tabId) {
    console.log('📑 탭 전환:', tabId);
    
    // 모든 탭 비활성화
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    
    // 선택된 탭 활성화
    const selectedButton = document.querySelector(`[data-tab="${tabId}"]`);
    const selectedContent = document.getElementById(tabId);
    
    if (selectedButton && selectedContent) {
        selectedButton.classList.add('active');
        selectedContent.style.display = 'block';
    }
}

/**
 * 고급 기능 초기화 (조건부)
 */
function initializeAdvancedFeatures() {
    console.log('🚀 고급 기능 초기화');
    
    // 고급 차트 기능이 있다면 초기화
    if (typeof initializeAdvancedCharts === 'function') {
        initializeAdvancedCharts();
    }
    
    // 고급 인터랙션 기능이 있다면 초기화
    if (typeof initializeAdvancedInteractions === 'function') {
        initializeAdvancedInteractions();
    }
    
    // 데이터 내보내기 기능이 있다면 초기화
    if (typeof initializeDataExport === 'function') {
        initializeDataExport();
    }
}

// ===============================
// CSS 스타일 추가
// ===============================

// 동적으로 필요한 CSS 스타일 추가
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .cluster-card.active {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #007bff;
    }
    
    .cluster-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 4px;
        color: white;
        font-size: 12px;
        font-weight: 500;
    }
    
    .cluster-badge i {
        margin-right: 5px;
    }
`;
document.head.appendChild(style);

console.log('✅ 클러스터 분석 유틸리티 함수들 로드 완료');
