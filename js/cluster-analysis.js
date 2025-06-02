/**
 * PUBG 플레이어 행동 분석 웹사이트 - 클러스터 분석 페이지
 * 클러스터 분석 전용 로직 및 인터랙션
 */

// ==================== 클러스터 분석 객체 ====================
const ClusterAnalysis = {
    data: {
        clusters: {},
        selectedClusters: new Set(),
        currentTab: 'features',
        charts: new Map()
    },
    
    // 실제 PUBG 클러스터 데이터 (분석 결과 기반)
    clusterData: {
        0: {
            name: "Survivor (보수적)",
            type: "Survivor",
            count: 14527,
            percentage: 18.2,
            color: "#56ab2f",
            characteristics: {
                kills: 15.2,
                damage: 25.3,
                walkDistance: 35.8,
                survival: 85.6,
                assists: 75.4,
                weapons: 42.1,
                boosts: 78.9,
                heals: 89.3
            },
            topFeatures: ["heal_boost_ratio", "assists", "has_swimDistance"],
            description: "생존을 우선시하는 신중한 플레이어"
        },
        1: {
            name: "Survivor (적극적)",
            type: "Survivor", 
            count: 24981,
            percentage: 31.2,
            color: "#4CAF50",
            characteristics: {
                kills: 28.5,
                damage: 42.7,
                walkDistance: 45.2,
                survival: 82.1,
                assists: 68.9,
                weapons: 55.8,
                boosts: 75.3,
                heals: 85.7
            },
            topFeatures: ["heal_boost_ratio", "assists", "damage_per_kill"],
            description: "적극적이면서도 생존력이 높은 플레이어"
        },
        2: {
            name: "Explorer (활발한)",
            type: "Explorer",
            count: 10756,
            percentage: 13.4,
            color: "#667eea",
            characteristics: {
                kills: 35.8,
                damage: 48.2,
                walkDistance: 89.4,
                survival: 65.3,
                assists: 45.7,
                weapons: 62.1,
                boosts: 58.9,
                heals: 68.2
            },
            topFeatures: ["walkDistance_log", "walkDistance", "revives"],
            description: "높은 이동성과 탐험 성향의 플레이어"
        },
        3: {
            name: "Explorer (균형형)",
            type: "Explorer",
            count: 15898,
            percentage: 19.9,
            color: "#5472d3",
            characteristics: {
                kills: 42.3,
                damage: 55.1,
                walkDistance: 78.6,
                survival: 68.9,
                assists: 52.4,
                weapons: 68.7,
                boosts: 62.1,
                heals: 65.8
            },
            topFeatures: ["walkDistance_log", "longestKill", "has_kills"],
            description: "이동과 전투의 균형을 이루는 플레이어"
        },
        4: {
            name: "Explorer (극한)",
            type: "Explorer",
            count: 4312,
            percentage: 5.4,
            color: "#3f51b5",
            characteristics: {
                kills: 38.9,
                damage: 52.8,
                walkDistance: 95.7,
                survival: 58.2,
                assists: 48.6,
                weapons: 65.3,
                boosts: 55.4,
                heals: 62.1
            },
            topFeatures: ["walkDistance_log", "walkDistance", "revives"],
            description: "극한의 이동성을 보이는 모험가형 플레이어"
        },
        5: {
            name: "Explorer (전술적)",
            type: "Explorer",
            count: 4046,
            percentage: 5.1,
            color: "#2196F3",
            characteristics: {
                kills: 45.2,
                damage: 58.7,
                walkDistance: 85.3,
                survival: 62.4,
                assists: 55.8,
                weapons: 72.1,
                boosts: 60.2,
                heals: 58.9
            },
            topFeatures: ["walkDistance_log", "walkDistance", "weaponsAcquired"],
            description: "전술적 이동과 무기 수집에 능한 플레이어"
        },
        6: {
            name: "Explorer (지구력)",
            type: "Explorer",
            count: 5391,
            percentage: 6.7,
            color: "#00BCD4",
            characteristics: {
                kills: 41.7,
                damage: 54.3,
                walkDistance: 92.1,
                survival: 64.8,
                assists: 50.2,
                weapons: 66.9,
                boosts: 57.8,
                heals: 61.5
            },
            topFeatures: ["walkDistance_log", "matchDuration", "walkDistance"],
            description: "장시간 게임에서 높은 지구력을 보이는 플레이어"
        },
        7: {
            name: "Aggressive (공격형)",
            type: "Aggressive",
            count: 89,
            percentage: 0.1,
            color: "#dc3545",
            characteristics: {
                kills: 92.5,
                damage: 95.8,
                walkDistance: 65.2,
                survival: 45.3,
                assists: 58.7,
                weapons: 78.9,
                boosts: 48.2,
                heals: 42.1
            },
            topFeatures: ["kill_efficiency", "damage_per_kill", "assists"],
            description: "극도로 공격적인 고실력 플레이어"
        }
    }
};

// ==================== 페이지 초기화 ====================
ClusterAnalysis.init = function() {
    console.log('🎯 클러스터 분석 페이지 초기화...');
    
    // DOM이 로드된 후 실행
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this.initPage());
    } else {
        this.initPage();
    }
};

ClusterAnalysis.initPage = function() {
    // 기본 데이터 설정
    this.data.clusters = this.clusterData;
    
    // 모든 클러스터 기본 선택
    Object.keys(this.clusterData).forEach(id => {
        this.data.selectedClusters.add(parseInt(id));
    });
    
    // UI 초기화
    this.initTabs();
    this.initFilters();
    this.initCharts();
    this.initEventListeners();
    
    // 초기 차트 생성
    this.renderOverviewCharts();
    this.renderFeatureTab();
    
    console.log('✅ 클러스터 분석 페이지 준비 완료');
};

// ==================== 탭 시스템 ====================
ClusterAnalysis.initTabs = function() {
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = tab.getAttribute('data-tab');
            
            // 활성 탭 변경
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
            
            this.data.currentTab = targetTab;
            this.renderCurrentTab();
        });
    });
};

// ==================== 필터 시스템 ====================
ClusterAnalysis.initFilters = function() {
    const filterContainer = document.querySelector('.cluster-filters');
    if (!filterContainer) return;
    
    // 클러스터 체크박스 생성
    let filtersHTML = '<div class="filter-controls">';
    filtersHTML += '<button class="btn btn-sm btn-secondary" id="selectAllClusters">전체 선택</button>';
    filtersHTML += '<button class="btn btn-sm btn-outline" id="deselectAllClusters">전체 해제</button>';
    filtersHTML += '</div><div class="cluster-checkboxes">';
    
    Object.entries(this.clusterData).forEach(([id, cluster]) => {
        filtersHTML += `
            <label class="cluster-checkbox">
                <input type="checkbox" value="${id}" checked>
                <span class="checkmark" style="border-color: ${cluster.color}"></span>
                <span class="cluster-name" style="color: ${cluster.color}">
                    ${cluster.name} (${cluster.percentage}%)
                </span>
            </label>
        `;
    });
    
    filtersHTML += '</div>';
    filterContainer.innerHTML = filtersHTML;
    
    // 이벤트 리스너 추가
    this.initFilterEvents();
};

ClusterAnalysis.initFilterEvents = function() {
    // 개별 체크박스
    document.querySelectorAll('.cluster-checkbox input').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const clusterId = parseInt(e.target.value);
            
            if (e.target.checked) {
                this.data.selectedClusters.add(clusterId);
            } else {
                this.data.selectedClusters.delete(clusterId);
            }
            
            this.updateCharts();
        });
    });
    
    // 전체 선택/해제
    document.getElementById('selectAllClusters')?.addEventListener('click', () => {
        this.selectAllClusters(true);
    });
    
    document.getElementById('deselectAllClusters')?.addEventListener('click', () => {
        this.selectAllClusters(false);
    });
};

ClusterAnalysis.selectAllClusters = function(select) {
    const checkboxes = document.querySelectorAll('.cluster-checkbox input');
    
    checkboxes.forEach(checkbox => {
        checkbox.checked = select;
        const clusterId = parseInt(checkbox.value);
        
        if (select) {
            this.data.selectedClusters.add(clusterId);
        } else {
            this.data.selectedClusters.delete(clusterId);
        }
    });
    
    this.updateCharts();
};

// ==================== 차트 초기화 ====================
ClusterAnalysis.initCharts = function() {
    // Chart.js 로드 확인
    if (typeof ChartUtils === 'undefined') {
        console.warn('⚠️ ChartUtils가 로드되지 않았습니다.');
        return;
    }
    
    this.data.charts.clear();
};

// ==================== 개요 차트 ====================
ClusterAnalysis.renderOverviewCharts = function() {
    // 클러스터 분포 차트
    this.renderDistributionChart();
    
    // 통계 카드 업데이트
    this.updateStatistics();
};

ClusterAnalysis.renderDistributionChart = function() {
    const selectedData = {};
    
    this.data.selectedClusters.forEach(clusterId => {
        if (this.clusterData[clusterId]) {
            selectedData[clusterId] = this.clusterData[clusterId];
        }
    });
    
    if (typeof ChartUtils !== 'undefined') {
        const chart = ChartUtils.createClusterDistributionChart('clusterDistributionChart', selectedData);
        this.data.charts.set('distribution', chart);
    }
};

ClusterAnalysis.updateStatistics = function() {
    const totalPlayers = Array.from(this.data.selectedClusters)
        .reduce((sum, clusterId) => sum + this.clusterData[clusterId].count, 0);
    
    const selectedCount = this.data.selectedClusters.size;
    const totalClusters = Object.keys(this.clusterData).length;
    
    // 통계 업데이트
    const playerCountEl = document.querySelector('.stat-value[data-stat="players"]');
    const clusterCountEl = document.querySelector('.stat-value[data-stat="clusters"]');
    
    if (playerCountEl) {
        ChartUtils.animateCounter(playerCountEl, 0, totalPlayers);
    }
    
    if (clusterCountEl) {
        clusterCountEl.textContent = `${selectedCount}/${totalClusters}`;
    }
};

// ==================== 탭별 렌더링 ====================
ClusterAnalysis.renderCurrentTab = function() {
    switch(this.data.currentTab) {
        case 'features':
            this.renderFeatureTab();
            break;
        case 'comparison':
            this.renderComparisonTab();
            break;
        case 'statistics':
            this.renderStatisticsTab();
            break;
    }
};

// ==================== 특성 탭 ====================
ClusterAnalysis.renderFeatureTab = function() {
    const container = document.getElementById('features-tab');
    if (!container) return;
    
    let html = '<div class="cluster-radar-grid">';
    
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.clusterData[clusterId];
        html += `
            <div class="radar-card">
                <h4 style="color: ${cluster.color}">${cluster.name}</h4>
                <div class="radar-chart-container">
                    <canvas id="radarChart${clusterId}" width="300" height="300"></canvas>
                </div>
                <div class="cluster-info">
                    <p class="cluster-description">${cluster.description}</p>
                    <div class="cluster-stats">
                        <span class="stat-item">
                            <i class="fas fa-users"></i>
                            ${cluster.count.toLocaleString()}명 (${cluster.percentage}%)
                        </span>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
    
    // 레이더 차트 생성
    setTimeout(() => {
        this.createRadarCharts();
    }, 100);
};

ClusterAnalysis.createRadarCharts = function() {
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.clusterData[clusterId];
        const characteristics = Object.values(cluster.characteristics);
        
        if (typeof ChartUtils !== 'undefined') {
            const chart = ChartUtils.createPlayerRadarChart(
                `radarChart${clusterId}`,
                characteristics,
                cluster.name
            );
            this.data.charts.set(`radar${clusterId}`, chart);
        }
    });
};

// ==================== 비교 탭 ====================
ClusterAnalysis.renderComparisonTab = function() {
    const container = document.getElementById('comparison-tab');
    if (!container) return;
    
    if (this.data.selectedClusters.size < 2) {
        container.innerHTML = `
            <div class="comparison-message">
                <i class="fas fa-info-circle"></i>
                <p>비교를 위해 2개 이상의 클러스터를 선택해주세요.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = `
        <div class="comparison-charts">
            <div class="chart-container">
                <h4>클러스터 특성 비교</h4>
                <canvas id="comparisonChart" width="800" height="400"></canvas>
            </div>
            <div class="comparison-insights">
                <h4>주요 차이점</h4>
                <div id="comparisonInsights"></div>
            </div>
        </div>
    `;
    
    setTimeout(() => {
        this.createComparisonChart();
        this.generateComparisonInsights();
    }, 100);
};

ClusterAnalysis.createComparisonChart = function() {
    const features = ['킬 수', '데미지', '이동거리', '생존력', '어시스트', '무기획득', '부스트', '치료'];
    const datasets = [];
    
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.clusterData[clusterId];
        const data = Object.values(cluster.characteristics);
        
        datasets.push({
            label: cluster.name,
            data: data,
            borderColor: cluster.color,
            backgroundColor: ChartUtils.adjustColorOpacity(cluster.color, 0.2),
            borderWidth: 2,
            pointBackgroundColor: cluster.color,
            pointBorderColor: '#fff',
            pointBorderWidth: 2
        });
    });
    
    if (typeof ChartUtils !== 'undefined') {
        const chart = ChartUtils.createRadarChart('comparisonChart', {
            labels: features,
            datasets: datasets
        }, { maxValue: 100 });
        
        this.data.charts.set('comparison', chart);
    }
};

ClusterAnalysis.generateComparisonInsights = function() {
    const container = document.getElementById('comparisonInsights');
    if (!container) return;
    
    const insights = [];
    const selectedClusters = Array.from(this.data.selectedClusters)
        .map(id => this.clusterData[id]);
    
    // 가장 공격적인 클러스터
    const mostAggressive = selectedClusters.reduce((max, cluster) => 
        cluster.characteristics.kills > max.characteristics.kills ? cluster : max
    );
    
    // 가장 생존력이 높은 클러스터
    const mostSurvival = selectedClusters.reduce((max, cluster) => 
        cluster.characteristics.survival > max.characteristics.survival ? cluster : max
    );
    
    // 가장 이동력이 높은 클러스터
    const mostMobile = selectedClusters.reduce((max, cluster) => 
        cluster.characteristics.walkDistance > max.characteristics.walkDistance ? cluster : max
    );
    
    insights.push(
        `<div class="insight-item">
            <i class="fas fa-crosshairs"></i>
            <span><strong>${mostAggressive.name}</strong>이 가장 공격적입니다 (킬 수: ${mostAggressive.characteristics.kills})</span>
        </div>`,
        `<div class="insight-item">
            <i class="fas fa-shield-alt"></i>
            <span><strong>${mostSurvival.name}</strong>이 가장 높은 생존력을 보입니다 (생존력: ${mostSurvival.characteristics.survival})</span>
        </div>`,
        `<div class="insight-item">
            <i class="fas fa-running"></i>
            <span><strong>${mostMobile.name}</strong>이 가장 활발한 이동 패턴을 보입니다 (이동거리: ${mostMobile.characteristics.walkDistance})</span>
        </div>`
    );
    
    container.innerHTML = insights.join('');
};

// ==================== 통계 탭 ====================
ClusterAnalysis.renderStatisticsTab = function() {
    const container = document.getElementById('statistics-tab');
    if (!container) return;
    
    container.innerHTML = `
        <div class="statistics-content">
            <div class="stats-table-container">
                <h4>상세 통계표</h4>
                <div class="table-responsive">
                    <table class="table" id="clusterStatsTable">
                        <thead>
                            <tr>
                                <th>클러스터</th>
                                <th>인원</th>
                                <th>비율</th>
                                <th>킬 수</th>
                                <th>데미지</th>
                                <th>생존력</th>
                                <th>이동거리</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
            <div class="export-options">
                <h4>데이터 내보내기</h4>
                <div class="export-buttons">
                    <button class="btn btn-primary" onclick="ClusterAnalysis.exportData('csv')">
                        <i class="fas fa-file-csv"></i> CSV 다운로드
                    </button>
                    <button class="btn btn-secondary" onclick="ClusterAnalysis.exportData('json')">
                        <i class="fas fa-file-code"></i> JSON 다운로드
                    </button>
                </div>
            </div>
        </div>
    `;
    
    this.populateStatsTable();
};

ClusterAnalysis.populateStatsTable = function() {
    const tbody = document.querySelector('#clusterStatsTable tbody');
    if (!tbody) return;
    
    let html = '';
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.clusterData[clusterId];
        html += `
            <tr>
                <td>
                    <span class="cluster-indicator" style="background-color: ${cluster.color}"></span>
                    ${cluster.name}
                </td>
                <td>${cluster.count.toLocaleString()}</td>
                <td>${cluster.percentage}%</td>
                <td>${cluster.characteristics.kills}</td>
                <td>${cluster.characteristics.damage}</td>
                <td>${cluster.characteristics.survival}</td>
                <td>${cluster.characteristics.walkDistance}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
};

// ==================== 데이터 내보내기 ====================
ClusterAnalysis.exportData = function(format) {
    const selectedData = {};
    this.data.selectedClusters.forEach(clusterId => {
        selectedData[clusterId] = this.clusterData[clusterId];
    });
    
    if (format === 'csv') {
        this.exportAsCSV(selectedData);
    } else if (format === 'json') {
        this.exportAsJSON(selectedData);
    }
};

ClusterAnalysis.exportAsCSV = function(data) {
    const headers = ['클러스터명', '인원', '비율', '킬수', '데미지', '생존력', '이동거리'];
    let csv = headers.join(',') + '\n';
    
    Object.values(data).forEach(cluster => {
        const row = [
            cluster.name,
            cluster.count,
            cluster.percentage,
            cluster.characteristics.kills,
            cluster.characteristics.damage,
            cluster.characteristics.survival,
            cluster.characteristics.walkDistance
        ];
        csv += row.join(',') + '\n';
    });
    
    this.downloadFile(csv, 'cluster-analysis.csv', 'text/csv');
};

ClusterAnalysis.exportAsJSON = function(data) {
    const json = JSON.stringify(data, null, 2);
    this.downloadFile(json, 'cluster-analysis.json', 'application/json');
};

ClusterAnalysis.downloadFile = function(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};

// ==================== 차트 업데이트 ====================
ClusterAnalysis.updateCharts = function() {
    // 개요 차트 업데이트
    this.renderDistributionChart();
    this.updateStatistics();
    
    // 현재 탭 다시 렌더링
    this.renderCurrentTab();
};

// ==================== 이벤트 리스너 ====================
ClusterAnalysis.initEventListeners = function() {
    // 모달 이벤트
    document.querySelectorAll('[data-modal="clusterDetailModal"]').forEach(trigger => {
        trigger.addEventListener('click', (e) => {
            const clusterId = e.target.closest('[data-cluster]')?.getAttribute('data-cluster');
            if (clusterId) {
                this.openClusterDetailModal(clusterId);
            }
        });
    });
    
    // 리사이즈 이벤트
    window.addEventListener('resize', App.utils.debounce(() => {
        this.data.charts.forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }, 250));
};

// ==================== 모달 ====================
ClusterAnalysis.openClusterDetailModal = function(clusterId) {
    const cluster = this.clusterData[clusterId];
    if (!cluster) return;
    
    const modal = document.getElementById('clusterDetailModal');
    if (!modal) return;
    
    // 모달 내용 업데이트
    modal.querySelector('.modal-title').textContent = cluster.name;
    modal.querySelector('.cluster-description').textContent = cluster.description;
    
    const statsContainer = modal.querySelector('.cluster-stats-detail');
    statsContainer.innerHTML = `
        <div class="stat-grid">
            <div class="stat-item">
                <span class="stat-label">인원</span>
                <span class="stat-value">${cluster.count.toLocaleString()}명</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">비율</span>
                <span class="stat-value">${cluster.percentage}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">주요 특징</span>
                <span class="stat-value">${cluster.topFeatures.join(', ')}</span>
            </div>
        </div>
    `;
    
    App.openModal('clusterDetailModal');
};

// ==================== 초기화 실행 ====================
ClusterAnalysis.init();

// ==================== 전역 접근 ====================
window.ClusterAnalysis = ClusterAnalysis;