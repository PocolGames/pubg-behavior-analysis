/**
 * PUBG Cluster Analysis Page JavaScript - Part 3
 * 모달 콘텐츠, 테이블, 내보내기 및 애니메이션 기능
 */

function createDetailModalContent(cluster) {
    let content = `
        <div class="modal-cluster-detail">
            <div class="cluster-summary">
                <div class="cluster-icon-large">
                    <i class="${cluster.icon || 'fas fa-users'}" style="color: ${cluster.color}"></i>
                </div>
                <div class="cluster-info-detail">
                    <h5>${cluster.name}</h5>
                    <p class="cluster-description">${cluster.description}</p>
                    <div class="cluster-stats-detail">
                        <div class="stat-item">
                            <span class="stat-label">플레이어 수</span>
                            <span class="stat-value">${cluster.count.toLocaleString()}명</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">전체 비율</span>
                            <span class="stat-value">${cluster.percentage}%</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="cluster-characteristics">
                <h6>핵심 특징</h6>
                <ul class="characteristics-list">
                    ${cluster.characteristics.map(char => `<li>${char}</li>`).join('')}
                </ul>
            </div>`;

    // Explorer 통합 데이터인 경우 하위 유형 표시
    if (cluster.subtypes) {
        content += `
            <div class="cluster-subtypes">
                <h6>하위 유형 (${cluster.subtypes.length}가지)</h6>
                <div class="subtypes-grid">
                    ${cluster.subtypes.map(subtype => `
                        <div class="subtype-card">
                            <div class="subtype-header">
                                <i class="${subtype.icon}" style="color: ${subtype.color}"></i>
                                <span class="subtype-name">${subtype.name}</span>
                            </div>
                            <div class="subtype-stats">
                                <span>${subtype.percentage}%</span>
                                <span>${subtype.count.toLocaleString()}명</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>`;
    }

    // 일반 클러스터인 경우 특성 수치 표시
    if (cluster.features) {
        content += `
            <div class="cluster-features-detail">
                <h6>특성 수치 (평균 대비 배수)</h6>
                <div class="features-grid">
                    ${Object.entries(cluster.features).map(([feature, value]) => `
                        <div class="feature-item">
                            <span class="feature-name">${feature.replace(/_/g, ' ')}</span>
                            <span class="feature-value">${value.toFixed(2)}×</span>
                        </div>
                    `).join('')}
                </div>
            </div>`;
    }

    content += `
            <div class="cluster-insights">
                <h6>비즈니스 인사이트</h6>
                <div class="insights-list">
                    ${generateClusterInsights(cluster)}
                </div>
            </div>
        </div>`;

    return content;
}

function generateClusterInsights(cluster) {
    const insights = {
        'survivor': [
            '치료 아이템 할인 이벤트 효과가 높을 것으로 예상',
            '팀 플레이 관련 컨텐츠 선호도 높음',
            '신규 플레이어 온보딩에 적합한 멘토 역할 가능'
        ],
        'explorer': [
            '맵 확장 컨텐츠에 가장 민감하게 반응할 유형',
            '이동 수단 관련 아이템 수요 높음',
            '장거리 무기 밸런싱에 큰 영향을 받음'
        ],
        'aggressive': [
            '경쟁 모드나 랭킹 시스템에 가장 적극적으로 참여',
            '새로운 무기나 전투 메커니즘에 빠르게 적응',
            '게임 밸런스 변화에 가장 민감하게 반응'
        ]
    };

    const clusterType = cluster.type || (cluster.name.toLowerCase().includes('survivor') ? 'survivor' : 
                        cluster.name.toLowerCase().includes('explorer') ? 'explorer' : 'aggressive');
    
    const clusterInsights = insights[clusterType] || insights['explorer'];
    
    return clusterInsights.map(insight => `
        <div class="insight-item">
            <i class="fas fa-lightbulb"></i>
            <span>${insight}</span>
        </div>
    `).join('');
}

// ===============================
// 통계 테이블 생성
// ===============================
function generateStatisticsTable() {
    const tableBody = document.getElementById('statisticsTableBody');
    if (!tableBody) return;

    // 모의 상세 통계 데이터 생성
    const statisticsData = CLUSTER_DATA.clusters.map(cluster => ({
        cluster: cluster.name,
        playerCount: cluster.count,
        percentage: cluster.percentage,
        avgKills: generateMockStat(cluster.type, 'kills'),
        avgDamage: generateMockStat(cluster.type, 'damage'),
        avgDistance: generateMockStat(cluster.type, 'distance'),
        avgSurvivalTime: generateMockStat(cluster.type, 'survival')
    }));

    tableBody.innerHTML = statisticsData.map(row => `
        <tr>
            <td class="cluster-name-cell">
                <span class="cluster-type-indicator ${getClusterTypeClass(row.cluster)}"></span>
                ${row.cluster}
            </td>
            <td>${row.playerCount.toLocaleString()}</td>
            <td>${row.percentage.toFixed(1)}%</td>
            <td>${row.avgKills.toFixed(1)}</td>
            <td>${row.avgDamage.toFixed(0)}</td>
            <td>${row.avgDistance.toFixed(0)}m</td>
            <td>${row.avgSurvivalTime.toFixed(1)}분</td>
        </tr>
    `).join('');
}

function generateMockStat(clusterType, statType) {
    const baseStats = {
        survivor: { kills: 1.2, damage: 120, distance: 800, survival: 18.5 },
        explorer: { kills: 1.8, damage: 180, distance: 2200, survival: 22.3 },
        aggressive: { kills: 6.2, damage: 480, distance: 1200, survival: 12.8 }
    };

    const base = baseStats[clusterType] || baseStats.explorer;
    const variation = (Math.random() - 0.5) * 0.3; // ±15% 변동
    
    return base[statType] * (1 + variation);
}

function getClusterTypeClass(clusterName) {
    if (clusterName.includes('Survivor')) return 'survivor-indicator';
    if (clusterName.includes('Explorer')) return 'explorer-indicator';
    if (clusterName.includes('Aggressive')) return 'aggressive-indicator';
    return 'default-indicator';
}

// ===============================
// 데이터 내보내기 기능
// ===============================
function exportToCSV() {
    const data = CLUSTER_DATA.clusters.map(cluster => ({
        '클러스터 ID': cluster.id,
        '클러스터 명': cluster.name,
        '유형': cluster.type,
        '플레이어 수': cluster.count,
        '비율(%)': cluster.percentage,
        '설명': cluster.description
    }));

    const csvContent = convertToCSV(data);
    downloadFile(csvContent, 'pubg-cluster-analysis.csv', 'text/csv');
    
    showNotification('CSV 파일이 다운로드되었습니다.', 'success');
}

function exportToJSON() {
    const jsonContent = JSON.stringify(CLUSTER_DATA, null, 2);
    downloadFile(jsonContent, 'pubg-cluster-analysis.json', 'application/json');
    
    showNotification('JSON 파일이 다운로드되었습니다.', 'success');
}

function convertToCSV(data) {
    if (!data.length) return '';

    const headers = Object.keys(data[0]);
    const csvRows = [
        headers.join(','),
        ...data.map(row => headers.map(header => `"${row[header]}"`).join(','))
    ];

    return csvRows.join('\n');
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// ===============================
// 비교 차트 업데이트
// ===============================
function updateComparisonChart(selectedClusters) {
    if (!comparisonChart) return;

    const filteredClusters = CLUSTER_DATA.clusters.filter(cluster => 
        selectedClusters.includes(cluster.id)
    );

    const features = ['kills', 'walkDistance', 'heal_boost_ratio', 'assists'];
    const datasets = features.map((feature, index) => ({
        label: feature.replace(/_/g, ' '),
        data: filteredClusters.map(cluster => {
            const value = cluster.features[feature] || 1;
            return Math.log10(value + 1);
        }),
        backgroundColor: `hsl(${index * 60}, 70%, 60%)`,
        borderWidth: 1
    }));

    comparisonChart.data.labels = filteredClusters.map(c => c.name.split(' ')[0]);
    comparisonChart.data.datasets = datasets;
    comparisonChart.update();
}

// ===============================
// 애니메이션 및 UI 효과
// ===============================
function startAnimations() {
    // 카운터 애니메이션
    animateCounters();
    
    // 카드 페이드인 애니메이션
    animateCards();
    
    // 차트 순차 로딩 애니메이션
    staggerChartAnimations();
}

function animateCounters() {
    const counters = document.querySelectorAll('.stat-number, .cluster-percentage, .cluster-count');
    
    counters.forEach(counter => {
        const target = parseFloat(counter.textContent.replace(/[^\d.]/g, ''));
        if (isNaN(target)) return;

        let current = 0;
        const increment = target / 60; // 1초 동안 애니메이션
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            
            // 원본 텍스트 형식 유지
            if (counter.classList.contains('cluster-percentage')) {
                counter.textContent = current.toFixed(1) + '%';
            } else if (counter.classList.contains('cluster-count')) {
                counter.textContent = Math.round(current).toLocaleString() + '명';
            } else {
                counter.textContent = current >= 1000 ? 
                    Math.round(current).toLocaleString() : 
                    current.toFixed(3);
            }
        }, 16); // ~60fps
    });
}

function animateCards() {
    const cards = document.querySelectorAll('.detailed-cluster-card');
    
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

function staggerChartAnimations() {
    // 차트들을 순차적으로 애니메이션
    const chartContainers = document.querySelectorAll('.chart-container');
    
    chartContainers.forEach((container, index) => {
        container.style.opacity = '0';
        container.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            container.style.transition = 'all 0.8s ease';
            container.style.opacity = '1';
            container.style.transform = 'scale(1)';
        }, index * 300);
    });
}

// ===============================
// 유틸리티 함수들
// ===============================
function showNotification(message, type = 'info') {
    // 기존 알림 제거
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    // 새 알림 생성
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;

    // 스타일 적용
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 20px',
        borderRadius: '8px',
        backgroundColor: getNotificationColor(type),
        color: 'white',
        zIndex: '10000',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        transform: 'translateX(100%)',
        transition: 'transform 0.3s ease'
    });

    document.body.appendChild(notification);

    // 애니메이션
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);

    // 닫기 버튼
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    });

    // 자동 제거
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        warning: 'exclamation-triangle',
        error: 'times-circle',
        info: 'info-circle'
    };
    return icons[type] || icons.info;
}

function getNotificationColor(type) {
    const colors = {
        success: '#4CAF50',
        warning: '#FF9800',
        error: '#F44336',
        info: '#2196F3'
    };
    return colors[type] || colors.info;
}

function showErrorMessage(message) {
    console.error('❌ 오류:', message);
    showNotification(message, 'error');
}

// ===============================
// 반응형 처리
// ===============================
function handleResize() {
    // 차트 반응형 조정
    Object.values(radarCharts).forEach(chart => {
        if (chart) chart.resize();
    });
    
    if (clusterDistributionChart) clusterDistributionChart.resize();
    if (comparisonChart) comparisonChart.resize();
    if (correlationChart) correlationChart.resize();
}

// 리사이즈 이벤트 리스너
window.addEventListener('resize', debounce(handleResize, 250));

// 디바운스 유틸리티
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

// ===============================
// 페이지 언로드 시 정리
// ===============================
window.addEventListener('beforeunload', function() {
    // 차트 인스턴스 정리
    Object.values(radarCharts).forEach(chart => {
        if (chart) chart.destroy();
    });
    
    if (clusterDistributionChart) clusterDistributionChart.destroy();
    if (comparisonChart) comparisonChart.destroy();
    if (correlationChart) correlationChart.destroy();
});

console.log('✅ 클러스터 분석 JavaScript 모듈 로드 완료');
