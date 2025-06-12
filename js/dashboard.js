/**
 * PUBG 플레이어 행동 분석 - 대시보드 스크립트
 * 차트 생성 및 플레이어 유형 정보 표시
 */

// 전역 변수
let playerTypeChart = null;
let featuresChart = null;

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

/**
 * 대시보드 초기화
 */
function initializeDashboard() {
    console.log('대시보드 초기화 시작');
    
    try {
        // 클러스터 데이터 확인
        if (typeof window.CLUSTER_DATA === 'undefined') {
            console.error('클러스터 데이터가 로드되지 않았습니다.');
            showErrorMessage('데이터 로딩 중 오류가 발생했습니다.');
            return;
        }
        
        // 차트 생성
        createPlayerTypeChart();
        createFeaturesChart();
        
        // 플레이어 유형 그리드 생성
        createPlayerTypesGrid();
        
        // 범례 생성
        createPlayerTypeLegend();
        
        console.log('대시보드 초기화 완료');
        
    } catch (error) {
        console.error('대시보드 초기화 중 오류:', error);
        showErrorMessage('대시보드 로딩 중 오류가 발생했습니다.');
    }
}

/**
 * 플레이어 유형 분포 차트 생성
 */
function createPlayerTypeChart() {
    const ctx = document.getElementById('playerTypeChart');
    if (!ctx) {
        console.error('playerTypeChart 캔버스를 찾을 수 없습니다.');
        return;
    }
    
    // 플레이어 유형별 데이터 준비
    const playerTypes = window.CLUSTER_DATA.PLAYER_TYPES;
    const labels = [];
    const data = [];
    const colors = [];
    
    // 클러스터별 비율 (실제 분석 결과 기반)
    const clusterRatios = {
        0: 18.2, // Survivor
        1: 31.2, // Survivor  
        2: 13.4, // Explorer
        3: 19.9, // Explorer
        4: 5.4,  // Explorer
        5: 5.1,  // Explorer
        6: 6.7,  // Explorer
        7: 0.1   // Aggressive
    };
    
    // 유형별 색상 정의
    const typeColors = {
        'Survivor': ['#28a745', '#20c997'],
        'Explorer': ['#17a2b8', '#6f42c1', '#fd7e14', '#ffc107', '#6c757d'],
        'Aggressive': ['#dc3545']
    };
    
    let colorIndex = {};
    for (let type in typeColors) {
        colorIndex[type] = 0;
    }
    
    for (let clusterId in playerTypes) {
        const cluster = playerTypes[clusterId];
        labels.push(`${cluster.name} ${clusterId}`);
        data.push(clusterRatios[clusterId] || 5);
        
        const playerType = cluster.name;
        if (typeColors[playerType]) {
            colors.push(typeColors[playerType][colorIndex[playerType] % typeColors[playerType].length]);
            colorIndex[playerType]++;
        } else {
            colors.push('#6c757d');
        }
    }
    
    // 차트 생성
    playerTypeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderColor: '#ffffff',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false // 사용자 정의 범례 사용
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return `${label}: ${value.toFixed(1)}%`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 1000
            }
        }
    });
}

/**
 * 주요 분석 특성 차트 생성
 */
function createFeaturesChart() {
    const ctx = document.getElementById('featuresChart');
    if (!ctx) {
        console.error('featuresChart 캔버스를 찾을 수 없습니다.');
        return;
    }
    
    // 특성 중요도 데이터 (실제 분석 결과 기반)
    const featureImportance = [
        { name: 'has_kills', importance: 32.3, color: '#dc3545' },
        { name: 'walkDistance_log', importance: 15.4, color: '#28a745' },
        { name: 'weaponsAcquired', importance: 11.2, color: '#ffc107' },
        { name: 'walkDistance', importance: 9.8, color: '#17a2b8' },
        { name: 'damageDealt_log', importance: 8.7, color: '#fd7e14' },
        { name: 'heal_boost_ratio', importance: 7.3, color: '#6f42c1' },
        { name: 'killPlace', importance: 6.9, color: '#20c997' },
        { name: 'total_distance', importance: 4.8, color: '#6c757d' }
    ];
    
    const labels = featureImportance.map(item => item.name.replace('_', ' '));
    const data = featureImportance.map(item => item.importance);
    const colors = featureImportance.map(item => item.color);
    
    // 차트 생성
    featuresChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '중요도 (%)',
                data: data,
                backgroundColor: colors.map(color => color + '80'), // 투명도 추가
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'y', // 수평 막대 차트
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `중요도: ${context.parsed.x.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 35,
                    grid: {
                        color: '#f8f9fa'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * 플레이어 유형 그리드 생성
 */
function createPlayerTypesGrid() {
    const container = document.getElementById('playerTypesGrid');
    if (!container) {
        console.error('playerTypesGrid 컨테이너를 찾을 수 없습니다.');
        return;
    }
    
    const playerTypes = window.CLUSTER_DATA.PLAYER_TYPES;
    let gridHTML = '';
    
    // 클러스터별 비율
    const clusterRatios = {
        0: 18.2, 1: 31.2, 2: 13.4, 3: 19.9,
        4: 5.4, 5: 5.1, 6: 6.7, 7: 0.1
    };
    
    // 유형별 아이콘 정의
    const typeIcons = {
        'Survivor': 'fas fa-heart',
        'Explorer': 'fas fa-compass',
        'Aggressive': 'fas fa-fire'
    };
    
    // 유형별 색상 정의
    const typeColorClasses = {
        'Survivor': 'success',
        'Explorer': 'info', 
        'Aggressive': 'danger'
    };
    
    for (let clusterId in playerTypes) {
        const cluster = playerTypes[clusterId];
        const ratio = clusterRatios[clusterId] || 5;
        const icon = typeIcons[cluster.name] || 'fas fa-user';
        const colorClass = typeColorClasses[cluster.name] || 'secondary';
        
        gridHTML += `
            <div class="col-lg-6 col-xl-4 mb-4">
                <div class="card border-0 shadow-sm h-100 player-type-card" data-cluster="${clusterId}">
                    <div class="card-header bg-${colorClass} text-white">
                        <div class="d-flex justify-content-between align-items-center">
                            <h5 class="card-title mb-0">
                                <i class="${icon} me-2"></i>
                                ${cluster.name}
                            </h5>
                            <span class="badge bg-light text-${colorClass}">
                                클러스터 ${clusterId}
                            </span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-6">
                                <div class="text-center">
                                    <div class="h4 text-${colorClass} mb-1">${ratio.toFixed(1)}%</div>
                                    <small class="text-muted">전체 플레이어</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="text-center">
                                    <div class="h4 text-${colorClass} mb-1">${Math.round(ratio * 8000)}</div>
                                    <small class="text-muted">예상 인원</small>
                                </div>
                            </div>
                        </div>
                        
                        <p class="card-text text-muted mb-3">
                            ${cluster.description}
                        </p>
                        
                        <div class="mb-3">
                            <h6 class="text-${colorClass} mb-2">
                                <i class="fas fa-star me-1"></i>주요 특징
                            </h6>
                            <div class="characteristics">
                                ${cluster.characteristics.map(char => 
                                    `<span class="badge bg-light text-${colorClass} me-1 mb-1">${char}</span>`
                                ).join('')}
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <h6 class="text-${colorClass} mb-2">
                                <i class="fas fa-gamepad me-1"></i>플레이 스타일
                            </h6>
                            <small class="text-muted">${cluster.playstyle}</small>
                        </div>
                        
                        <div>
                            <h6 class="text-${colorClass} mb-2">
                                <i class="fas fa-lightbulb me-1"></i>개선 팁
                            </h6>
                            <small class="text-muted">${cluster.tips}</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = gridHTML;
    
    // 카드 호버 효과 추가
    addCardHoverEffects();
}

/**
 * 플레이어 유형 범례 생성
 */
function createPlayerTypeLegend() {
    const container = document.getElementById('playerTypeLegend');
    if (!container) {
        console.error('playerTypeLegend 컨테이너를 찾을 수 없습니다.');
        return;
    }
    
    const playerTypes = window.CLUSTER_DATA.PLAYER_TYPES;
    let legendHTML = '';
    
    // 클러스터별 비율
    const clusterRatios = {
        0: 18.2, 1: 31.2, 2: 13.4, 3: 19.9,
        4: 5.4, 5: 5.1, 6: 6.7, 7: 0.1
    };
    
    // 유형별 색상
    const typeColors = {
        'Survivor': '#28a745',
        'Explorer': '#17a2b8',
        'Aggressive': '#dc3545'
    };
    
    for (let clusterId in playerTypes) {
        const cluster = playerTypes[clusterId];
        const ratio = clusterRatios[clusterId] || 5;
        const color = typeColors[cluster.name] || '#6c757d';
        
        legendHTML += `
            <div class="legend-item d-flex align-items-center mb-2" data-cluster="${clusterId}">
                <div class="legend-color me-3" style="width: 15px; height: 15px; background-color: ${color}; border-radius: 3px;"></div>
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="fw-medium">${cluster.name} ${clusterId}</span>
                        <span class="text-muted">${ratio.toFixed(1)}%</span>
                    </div>
                    <small class="text-muted">${cluster.description}</small>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = legendHTML;
    
    // 범례 클릭 이벤트 추가
    addLegendClickEvents();
}

/**
 * 카드 호버 효과 추가
 */
function addCardHoverEffects() {
    const cards = document.querySelectorAll('.player-type-card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

/**
 * 범례 클릭 이벤트 추가
 */
function addLegendClickEvents() {
    const legendItems = document.querySelectorAll('.legend-item');
    
    legendItems.forEach(item => {
        item.addEventListener('click', function() {
            const clusterId = this.dataset.cluster;
            highlightPlayerType(clusterId);
        });
        
        // 호버 효과
        item.style.cursor = 'pointer';
        item.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f8f9fa';
            this.style.borderRadius = '4px';
            this.style.padding = '4px';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.backgroundColor = 'transparent';
            this.style.padding = '0';
        });
    });
}

/**
 * 플레이어 유형 하이라이트
 */
function highlightPlayerType(clusterId) {
    // 모든 카드에서 하이라이트 제거
    const allCards = document.querySelectorAll('.player-type-card');
    allCards.forEach(card => {
        card.classList.remove('border-primary');
        card.style.boxShadow = '';
    });
    
    // 선택된 카드 하이라이트
    const targetCard = document.querySelector(`[data-cluster="${clusterId}"]`);
    if (targetCard) {
        targetCard.classList.add('border-primary');
        targetCard.style.boxShadow = '0 0.5rem 1rem rgba(0, 123, 255, 0.25)';
        
        // 카드로 스크롤
        targetCard.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }
}

/**
 * 에러 메시지 표시
 */
function showErrorMessage(message) {
    // 간단한 알림 표시
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.style.maxWidth = '400px';
    
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 5초 후 자동 제거
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

/**
 * 차트 크기 조정 (반응형)
 */
function resizeCharts() {
    if (playerTypeChart) {
        playerTypeChart.resize();
    }
    if (featuresChart) {
        featuresChart.resize();
    }
}

// 윈도우 크기 변경 시 차트 크기 조정
window.addEventListener('resize', debounce(resizeCharts, 300));

/**
 * 디바운스 함수
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
