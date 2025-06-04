/**
 * PUBG Cluster Analysis - Final Completion Module
 * 클러스터 분석 JavaScript의 최종 완성 모듈
 * 통계 테이블, 내보내기, 애니메이션, 알림 등 구현
 */

// ===============================
// 통계 테이블 생성
// ===============================
function generateStatisticsTable() {
    const tableContainer = document.getElementById('statisticsTable');
    if (!tableContainer) {
        console.warn('통계 테이블 컨테이너를 찾을 수 없습니다.');
        return;
    }

    const table = createStatisticsTableHTML();
    tableContainer.innerHTML = table;
    
    // 테이블 정렬 기능 추가
    addTableSortingFeature();
    
    console.log('✅ 통계 테이블 생성 완료');
}

function createStatisticsTableHTML() {
    const headers = [
        '클러스터', '플레이어 수', '비율', '주요 특성', 
        '킬 평균', '이동 거리', '생존력', '공격성'
    ];
    
    let tableHTML = `
        <div class="table-responsive">
            <table class="statistics-table">
                <thead>
                    <tr>
                        ${headers.map(header => `<th data-sort="${header.toLowerCase()}">${header} <i class="fas fa-sort"></i></th>`).join('')}
                    </tr>
                </thead>
                <tbody>
    `;
    
    CLUSTER_DATA.clusters.forEach(cluster => {
        const mainFeature = Object.keys(cluster.features)[0];
        const killAvg = cluster.features.kills || 1.0;
        const moveDistance = cluster.features.walkDistance || cluster.features.walkDistance_log || 1.0;
        const survival = cluster.features.heal_boost_ratio || 1.0;
        const aggression = cluster.type === 'aggressive' ? 'Very High' : 
                         cluster.type === 'explorer' ? 'Medium' : 'Low';
        
        tableHTML += `
            <tr data-cluster="${cluster.id}" class="table-row-hover">
                <td>
                    <div class="cluster-cell">
                        <div class="cluster-color" style="background-color: ${cluster.color}"></div>
                        <span class="cluster-name">${cluster.name}</span>
                    </div>
                </td>
                <td class="number-cell">${cluster.count.toLocaleString()}</td>
                <td class="percentage-cell">${cluster.percentage}%</td>
                <td class="feature-cell">${mainFeature.replace(/_/g, ' ')}</td>
                <td class="number-cell">${killAvg.toFixed(1)}</td>
                <td class="number-cell">${moveDistance.toFixed(1)}</td>
                <td class="number-cell">${survival.toFixed(1)}</td>
                <td class="aggression-cell">
                    <span class="aggression-${aggression.toLowerCase().replace(' ', '-')}">${aggression}</span>
                </td>
            </tr>
        `;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    return tableHTML;
}

function addTableSortingFeature() {
    const headers = document.querySelectorAll('.statistics-table th[data-sort]');
    
    headers.forEach(header => {
        header.addEventListener('click', function() {
            const sortBy = this.getAttribute('data-sort');
            const tbody = this.closest('table').querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // 정렬 방향 결정
            const isAscending = !this.classList.contains('sort-desc');
            
            // 모든 헤더에서 정렬 클래스 제거
            headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
            
            // 현재 헤더에 정렬 클래스 추가
            this.classList.add(isAscending ? 'sort-asc' : 'sort-desc');
            
            // 행 정렬
            rows.sort((a, b) => {
                const aValue = getSortValue(a, sortBy);
                const bValue = getSortValue(b, sortBy);
                
                if (typeof aValue === 'number' && typeof bValue === 'number') {
                    return isAscending ? aValue - bValue : bValue - aValue;
                } else {
                    return isAscending ? 
                        aValue.toString().localeCompare(bValue.toString()) :
                        bValue.toString().localeCompare(aValue.toString());
                }
            });
            
            // 정렬된 행들을 다시 추가
            rows.forEach(row => tbody.appendChild(row));
        });
    });
}

function getSortValue(row, sortBy) {
    const clusterId = parseInt(row.getAttribute('data-cluster'));
    const cluster = CLUSTER_DATA.clusters.find(c => c.id === clusterId);
    
    switch(sortBy) {
        case '클러스터':
            return cluster.name;
        case '플레이어 수':
            return cluster.count;
        case '비율':
            return cluster.percentage;
        case '킬 평균':
            return cluster.features.kills || 1.0;
        case '이동 거리':
            return cluster.features.walkDistance || cluster.features.walkDistance_log || 1.0;
        case '생존력':
            return cluster.features.heal_boost_ratio || 1.0;
        default:
            return 0;
    }
}

// ===============================
// 애니메이션 및 UI 효과
// ===============================
function startAnimations() {
    // 카드 애니메이션
    animateClusterCards();
    
    // 카운터 애니메이션
    animateCounters();
    
    // 차트 등장 애니메이션
    setTimeout(() => {
        animateCharts();
    }, 500);
    
    console.log('✅ 애니메이션 시작');
}

function animateClusterCards() {
    const cards = document.querySelectorAll('.cluster-card');
    
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease-out';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

function animateCounters() {
    const counters = document.querySelectorAll('.counter-number');
    
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target') || counter.textContent.replace(/,/g, ''));
        const duration = 2000;
        const stepTime = 50;
        const steps = duration / stepTime;
        const increment = target / steps;
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            counter.textContent = Math.floor(current).toLocaleString();
        }, stepTime);
    });
}

function animateCharts() {
    // 차트들이 존재할 때만 애니메이션 실행
    if (clusterDistributionChart) {
        clusterDistributionChart.update('active');
    }
    
    Object.values(radarCharts).forEach(chart => {
        if (chart) {
            chart.update('active');
        }
    });
    
    if (comparisonChart) {
        comparisonChart.update('active');
    }
}

// ===============================
// 데이터 내보내기 기능
// ===============================
function exportToCSV() {
    try {
        const csvContent = generateCSVContent();
        downloadFile(csvContent, 'pubg-cluster-analysis.csv', 'text/csv');
        showNotification('CSV 파일이 다운로드되었습니다.', 'success');
    } catch (error) {
        console.error('CSV 내보내기 오류:', error);
        showNotification('CSV 내보내기 중 오류가 발생했습니다.', 'error');
    }
}

function exportToJSON() {
    try {
        const jsonContent = JSON.stringify(CLUSTER_DATA, null, 2);
        downloadFile(jsonContent, 'pubg-cluster-analysis.json', 'application/json');
        showNotification('JSON 파일이 다운로드되었습니다.', 'success');
    } catch (error) {
        console.error('JSON 내보내기 오류:', error);
        showNotification('JSON 내보내기 중 오류가 발생했습니다.', 'error');
    }
}

function generateCSVContent() {
    const headers = ['ID', 'Name', 'Type', 'Count', 'Percentage', 'Color', 'Description'];
    const featureHeaders = ['kills', 'walkDistance', 'heal_boost_ratio', 'assists', 'damage_per_kill'];
    
    let csv = [...headers, ...featureHeaders].join(',') + '\\n';
    
    CLUSTER_DATA.clusters.forEach(cluster => {
        const basicData = [
            cluster.id,
            `"${cluster.name}"`,
            cluster.type,
            cluster.count,
            cluster.percentage,
            cluster.color,
            `"${cluster.description}"`
        ];
        
        const featureData = featureHeaders.map(feature => 
            cluster.features[feature] || ''
        );
        
        csv += [...basicData, ...featureData].join(',') + '\\n';
    });
    
    return csv;
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
// 비교 차트 업데이트 함수
// ===============================
function updateComparisonChart(selectedClusters) {
    if (!comparisonChart || selectedClusters.length === 0) return;
    
    const filteredData = CLUSTER_DATA.clusters.filter(cluster => 
        selectedClusters.includes(cluster.id)
    );
    
    const features = ['kills', 'walkDistance', 'heal_boost_ratio', 'assists'];
    
    comparisonChart.data.labels = filteredData.map(c => c.name.split(' ')[0]);
    comparisonChart.data.datasets = features.map((feature, index) => ({
        label: feature.replace(/_/g, ' '),
        data: filteredData.map(cluster => {
            const value = cluster.features[feature] || 1;
            return Math.log10(value + 1);
        }),
        backgroundColor: `hsl(${index * 60}, 70%, 60%)`,
        borderWidth: 1
    }));
    
    comparisonChart.update();
}

// ===============================
// 모달 내용 생성 함수
// ===============================
function createDetailModalContent(cluster) {
    return `
        <div class="modal-cluster-header">
            <div class="cluster-icon" style="background-color: ${cluster.color}">
                <i class="${cluster.icon}"></i>
            </div>
            <div class="cluster-info">
                <h3>${cluster.name}</h3>
                <p class="cluster-type">${cluster.type.charAt(0).toUpperCase() + cluster.type.slice(1)} Type</p>
                <p class="cluster-stats">${cluster.count.toLocaleString()}명 (${cluster.percentage}%)</p>
            </div>
        </div>
        
        <div class="modal-cluster-description">
            <p>${cluster.description}</p>
        </div>
        
        <div class="modal-cluster-characteristics">
            <h4>주요 특성</h4>
            <ul>
                ${cluster.characteristics.map(char => `<li>${char}</li>`).join('')}
            </ul>
        </div>
        
        <div class="modal-cluster-features">
            <h4>특성 수치</h4>
            <div class="feature-grid">
                ${Object.entries(cluster.features).map(([key, value]) => `
                    <div class="feature-item">
                        <span class="feature-name">${key.replace(/_/g, ' ')}</span>
                        <span class="feature-value">${value.toFixed(2)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        
        ${cluster.subtypes ? createSubtypesContent(cluster.subtypes) : ''}
    `;
}

function createSubtypesContent(subtypes) {
    return `
        <div class="modal-cluster-subtypes">
            <h4>하위 유형들</h4>
            <div class="subtypes-grid">
                ${subtypes.map(subtype => `
                    <div class="subtype-card">
                        <div class="subtype-header">
                            <div class="subtype-color" style="background-color: ${subtype.color}"></div>
                            <span class="subtype-name">${subtype.name}</span>
                        </div>
                        <div class="subtype-stats">
                            <span>${subtype.count.toLocaleString()}명 (${subtype.percentage}%)</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// ===============================
// 알림 시스템
// ===============================
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // 알림 컨테이너가 없으면 생성
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // 애니메이션으로 표시
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // 5초 후 자동 제거
    setTimeout(() => {
        removeNotification(notification);
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function removeNotification(notification) {
    notification.classList.add('hide');
    setTimeout(() => {
        if (notification.parentElement) {
            notification.parentElement.removeChild(notification);
        }
    }, 300);
}

function showErrorMessage(message) {
    const errorContainer = document.createElement('div');
    errorContainer.className = 'error-message';
    errorContainer.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <h3>오류 발생</h3>
            <p>${message}</p>
            <button onclick="location.reload()" class="btn btn-primary">
                페이지 새로고침
            </button>
        </div>
    `;
    
    document.body.appendChild(errorContainer);
}

// ===============================
// 유틸리티 함수들
// ===============================
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

function throttle(func, limit) {
    let lastFunc;
    let lastRan;
    return function(...args) {
        if (!lastRan) {
            func.apply(this, args);
            lastRan = Date.now();
        } else {
            clearTimeout(lastFunc);
            lastFunc = setTimeout(() => {
                if ((Date.now() - lastRan) >= limit) {
                    func.apply(this, args);
                    lastRan = Date.now();
                }
            }, limit - (Date.now() - lastRan));
        }
    }
}

// 성능 최적화: 리사이즈 이벤트 처리
window.addEventListener('resize', debounce(() => {
    // 차트 리사이즈
    if (clusterDistributionChart) clusterDistributionChart.resize();
    if (comparisonChart) comparisonChart.resize();
    if (correlationChart) correlationChart.resize();
    
    Object.values(radarCharts).forEach(chart => {
        if (chart) chart.resize();
    });
}, 250));

// ===============================
// 접근성 및 키보드 내비게이션
// ===============================
function enhanceAccessibility() {
    // 포커스 트랩 for 모달
    const modal = document.getElementById('clusterDetailModal');
    if (modal) {
        modal.addEventListener('keydown', handleModalKeydown);
    }
    
    // 차트에 대한 대안 텍스트 제공
    addChartAriaLabels();
    
    // 테이블 접근성 향상
    enhanceTableAccessibility();
}

function handleModalKeydown(e) {
    if (e.key === 'Tab') {
        const focusableElements = this.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        
        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    }
}

function addChartAriaLabels() {
    const charts = document.querySelectorAll('canvas');
    charts.forEach((chart, index) => {
        chart.setAttribute('role', 'img');
        chart.setAttribute('aria-label', `클러스터 분석 차트 ${index + 1}`);
    });
}

function enhanceTableAccessibility() {
    const table = document.querySelector('.statistics-table');
    if (table) {
        table.setAttribute('role', 'table');
        table.setAttribute('aria-label', '클러스터 통계 테이블');
        
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.setAttribute('scope', 'col');
            header.setAttribute('aria-sort', 'none');
        });
    }
}

// 페이지 로드 완료 후 접근성 향상 적용
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(enhanceAccessibility, 1000);
});

console.log('✅ 클러스터 분석 최종 완성 모듈 로드 완료');
