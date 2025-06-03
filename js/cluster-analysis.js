/**
 * PUBG 플레이어 행동 분석 웹사이트 - 클러스터 분석 페이지
 * 실제 JSON 데이터를 활용한 클러스터 분석 로직
 */

// ==================== 클러스터 분석 객체 ====================
const ClusterAnalysis = {
    data: {
        clusters: {},
        selectedClusters: new Set(),
        currentTab: 'features',
        charts: new Map(),
        rawData: null // 로드된 JSON 데이터
    }
};

// ==================== 데이터 로딩 ====================
ClusterAnalysis.loadData = async function() {
    try {
        console.log('📊 클러스터 데이터 로딩 중...');
        
        const response = await fetch('../data/cluster-data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        this.data.rawData = await response.json();
        this.data.clusters = this.data.rawData.clusters;
        
        console.log('✅ 클러스터 데이터 로딩 완료:', Object.keys(this.data.clusters).length + '개 클러스터');
        return true;
        
    } catch (error) {
        console.error('❌ 클러스터 데이터 로딩 실패:', error);
        
        // 백업 데이터 사용
        this.data.clusters = this.getFallbackData();
        console.log('🔄 백업 데이터 사용');
        return false;
    }
};

// ==================== 백업 데이터 ====================
ClusterAnalysis.getFallbackData = function() {
    return {
        "0": {
            "id": 0,
            "name": "Survivor (보수적)",
            "type": "Survivor",
            "count": 14527,
            "percentage": 18.2,
            "color": "#56ab2f",
            "icon": "fa-shield-alt",
            "characteristics": {
                "killRate": 0.3,
                "damageDealt": 57,
                "walkDistance": 143,
                "healUsage": 0.13,
                "boostUsage": 0.07,
                "survivalTime": 85,
                "aggressiveness": 15,
                "teamwork": 90
            },
            "description": "생존을 최우선으로 하는 신중한 플레이어",
            "topFeatures": ["heal_boost_ratio", "assists", "has_swimDistance"]
        },
        "1": {
            "id": 1,
            "name": "Survivor (적극적)",
            "type": "Survivor",
            "count": 24981,
            "percentage": 31.2,
            "color": "#7cb342",
            "icon": "fa-heart",
            "characteristics": {
                "killRate": 0.59,
                "damageDealt": 95,
                "walkDistance": 550,
                "healUsage": 0.69,
                "boostUsage": 0.40,
                "survivalTime": 75,
                "aggressiveness": 35,
                "teamwork": 85
            },
            "description": "적극적이지만 신중한 생존 전략을 구사하는 플레이어",
            "topFeatures": ["heal_boost_ratio", "assists", "damage_per_kill"]
        },
        "7": {
            "id": 7,
            "name": "Aggressive (공격형)",
            "type": "Aggressive",
            "count": 89,
            "percentage": 0.1,
            "color": "#dc3545",
            "icon": "fa-fire",
            "characteristics": {
                "killRate": 2.06,
                "damageDealt": 259,
                "walkDistance": 2645,
                "healUsage": 3.17,
                "boostUsage": 3.06,
                "survivalTime": 40,
                "aggressiveness": 95,
                "teamwork": 60
            },
            "description": "극도로 공격적인 고위험 고수익 플레이어",
            "topFeatures": ["kill_efficiency", "damage_per_kill", "assists"]
        }
    };
};

// ==================== 페이지 초기화 ====================
ClusterAnalysis.init = function() {
    console.log('🎯 클러스터 분석 페이지 초기화...');
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this.initPage());
    } else {
        this.initPage();
    }
};

ClusterAnalysis.initPage = async function() {
    this.showLoading(true);
    
    try {
        await this.loadData();
        
        Object.keys(this.data.clusters).forEach(id => {
            this.data.selectedClusters.add(parseInt(id));
        });
        
        this.initTabs();
        this.initFilters();
        this.initCharts();
        this.initEventListeners();
        
        this.renderOverviewCharts();
        this.renderFeatureTab();
        
        console.log('✅ 클러스터 분석 페이지 준비 완료');
        
    } catch (error) {
        console.error('❌ 페이지 초기화 실패:', error);
        this.showError('데이터 로딩에 실패했습니다.');
    } finally {
        this.showLoading(false);
    }
};

// ==================== 로딩 및 오류 표시 ====================
ClusterAnalysis.showLoading = function(show) {
    const loader = document.querySelector('.page-loader');
    if (loader) {
        loader.style.display = show ? 'flex' : 'none';
    }
};

ClusterAnalysis.showError = function(message) {
    if (typeof App !== 'undefined' && App.showNotification) {
        App.showNotification(message, 'error');
    } else {
        alert(message);
    }
};

// ==================== 탭 시스템 ====================
ClusterAnalysis.initTabs = function() {
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = tab.getAttribute('data-tab');
            
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
    
    let filtersHTML = '<div class="filter-controls">';
    filtersHTML += '<button class="btn btn-sm btn-secondary" id="selectAllClusters">전체 선택</button>';
    filtersHTML += '<button class="btn btn-sm btn-outline" id="deselectAllClusters">전체 해제</button>';
    filtersHTML += '</div><div class="cluster-checkboxes">';
    
    Object.entries(this.data.clusters).forEach(([id, cluster]) => {
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
    
    this.initFilterEvents();
};

ClusterAnalysis.initFilterEvents = function() {
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
    if (typeof ChartUtils === 'undefined') {
        console.warn('⚠️ ChartUtils가 로드되지 않았습니다.');
        return;
    }
    
    this.data.charts.clear();
};

// ==================== 개요 차트 ====================
ClusterAnalysis.renderOverviewCharts = function() {
    this.renderDistributionChart();
    this.updateStatistics();
};

ClusterAnalysis.renderDistributionChart = function() {
    const selectedData = {};
    
    this.data.selectedClusters.forEach(clusterId => {
        if (this.data.clusters[clusterId]) {
            selectedData[clusterId] = this.data.clusters[clusterId];
        }
    });
    
    if (typeof ChartUtils !== 'undefined') {
        const chart = ChartUtils.createClusterDistributionChart('clusterDistributionChart', selectedData);
        this.data.charts.set('distribution', chart);
    }
};

ClusterAnalysis.updateStatistics = function() {
    const totalPlayers = Array.from(this.data.selectedClusters)
        .reduce((sum, clusterId) => sum + this.data.clusters[clusterId].count, 0);
    
    const selectedCount = this.data.selectedClusters.size;
    const totalClusters = Object.keys(this.data.clusters).length;
    
    const metadata = this.data.rawData?.metadata || {
        totalPlayers: 80000,
        silhouetteScore: 0.1391
    };
    
    const playerCountEl = document.querySelector('.stat-value[data-stat="players"]');
    const clusterCountEl = document.querySelector('.stat-value[data-stat="clusters"]');
    const qualityScoreEl = document.querySelector('.stat-value[data-stat="quality"]');
    
    if (playerCountEl && typeof ChartUtils !== 'undefined') {
        ChartUtils.animateCounter(playerCountEl, 0, totalPlayers);
    }
    
    if (clusterCountEl) {
        clusterCountEl.textContent = `${selectedCount}/${totalClusters}`;
    }
    
    if (qualityScoreEl) {
        qualityScoreEl.textContent = metadata.silhouetteScore.toFixed(3);
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
        const cluster = this.data.clusters[clusterId];
        
        html += `
            <div class="radar-card">
                <h4 style="color: ${cluster.color}">
                    <i class="fas ${cluster.icon}"></i>
                    ${cluster.name}
                </h4>
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
                        <span class="stat-item">
                            <i class="fas fa-tags"></i>
                            ${cluster.type}
                        </span>
                    </div>
                    <div class="top-features">
                        <h5>주요 특성:</h5>
                        <div class="feature-tags">
                            ${cluster.topFeatures ? cluster.topFeatures.map(feature => 
                                `<span class="feature-tag">${this.getFeatureDisplayName(feature)}</span>`
                            ).join('') : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
    
    setTimeout(() => {
        this.createRadarCharts();
    }, 100);
};

ClusterAnalysis.getFeatureDisplayName = function(feature) {
    const featureNames = {
        'heal_boost_ratio': '치료/부스트 비율',
        'assists': '어시스트',
        'has_swimDistance': '수영 활동',
        'damage_per_kill': '킬당 데미지',
        'walkDistance_log': '이동거리(로그)',
        'walkDistance': '보행거리',
        'revives': '소생',
        'longestKill': '최장킬거리',
        'has_kills': '킬 기록',
        'weaponsAcquired': '무기획득',
        'matchDuration': '게임시간',
        'kill_efficiency': '킬 효율성'
    };
    
    return featureNames[feature] || feature;
};

ClusterAnalysis.createRadarCharts = function() {
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.data.clusters[clusterId];
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
        const cluster = this.data.clusters[clusterId];
        const data = Object.values(cluster.characteristics);
        
        datasets.push({
            label: cluster.name,
            data: data,
            borderColor: cluster.color,
            backgroundColor: this.adjustColorOpacity(cluster.color, 0.2),
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

ClusterAnalysis.adjustColorOpacity = function(color, opacity) {
    if (color.startsWith('#')) {
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }
    return color;
};

ClusterAnalysis.generateComparisonInsights = function() {
    const container = document.getElementById('comparisonInsights');
    if (!container) return;
    
    const insights = [];
    const selectedClusters = Array.from(this.data.selectedClusters)
        .map(id => this.data.clusters[id]);
    
    if (selectedClusters.length < 2) return;
    
    const mostAggressive = selectedClusters.reduce((max, cluster) => 
        (cluster.characteristics.aggressiveness || 0) > (max.characteristics.aggressiveness || 0) ? cluster : max
    );
    
    const mostSurvival = selectedClusters.reduce((max, cluster) => 
        (cluster.characteristics.survivalTime || 0) > (max.characteristics.survivalTime || 0) ? cluster : max
    );
    
    const mostMobile = selectedClusters.reduce((max, cluster) => 
        (cluster.characteristics.walkDistance || 0) > (max.characteristics.walkDistance || 0) ? cluster : max
    );
    
    insights.push(
        `<div class="insight-item">
            <i class="fas fa-crosshairs" style="color: ${mostAggressive.color}"></i>
            <span><strong>${mostAggressive.name}</strong>이 가장 공격적입니다</span>
        </div>`,
        `<div class="insight-item">
            <i class="fas fa-shield-alt" style="color: ${mostSurvival.color}"></i>
            <span><strong>${mostSurvival.name}</strong>이 가장 높은 생존력을 보입니다</span>
        </div>`,
        `<div class="insight-item">
            <i class="fas fa-running" style="color: ${mostMobile.color}"></i>
            <span><strong>${mostMobile.name}</strong>이 가장 활발한 이동 패턴을 보입니다</span>
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
                                <th>유형</th>
                                <th>인원</th>
                                <th>비율</th>
                                <th>공격성</th>
                                <th>생존력</th>
                                <th>이동성</th>
                                <th>팀워크</th>
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
        const cluster = this.data.clusters[clusterId];
        html += `
            <tr>
                <td>
                    <span class="cluster-indicator" style="background-color: ${cluster.color}"></span>
                    ${cluster.name}
                </td>
                <td>
                    <span class="badge" style="background-color: ${cluster.color}">${cluster.type}</span>
                </td>
                <td>${cluster.count.toLocaleString()}</td>
                <td>${cluster.percentage}%</td>
                <td>${cluster.characteristics.aggressiveness || 'N/A'}</td>
                <td>${cluster.characteristics.survivalTime || 'N/A'}</td>
                <td>${cluster.characteristics.walkDistance || 'N/A'}</td>
                <td>${cluster.characteristics.teamwork || 'N/A'}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
};

// ==================== 데이터 내보내기 ====================
ClusterAnalysis.exportData = function(format) {
    const selectedData = {};
    this.data.selectedClusters.forEach(clusterId => {
        selectedData[clusterId] = this.data.clusters[clusterId];
    });
    
    if (format === 'csv') {
        this.exportAsCSV(selectedData);
    } else if (format === 'json') {
        this.exportAsJSON(selectedData);
    }
    
    if (typeof App !== 'undefined' && App.showNotification) {
        App.showNotification(`${format.toUpperCase()} 파일이 다운로드되었습니다.`, 'success');
    }
};

ClusterAnalysis.exportAsCSV = function(data) {
    const headers = ['클러스터명', '유형', '인원', '비율(%)', '공격성', '생존력', '이동성', '팀워크'];
    let csv = headers.join(',') + '\n';
    
    Object.values(data).forEach(cluster => {
        const row = [
            `"${cluster.name}"`,
            cluster.type,
            cluster.count,
            cluster.percentage,
            cluster.characteristics.aggressiveness || 0,
            cluster.characteristics.survivalTime || 0,
            cluster.characteristics.walkDistance || 0,
            cluster.characteristics.teamwork || 0
        ];
        csv += row.join(',') + '\n';
    });
    
    this.downloadFile(csv, 'pubg-cluster-analysis.csv', 'text/csv');
};

ClusterAnalysis.exportAsJSON = function(data) {
    const exportData = {
        exportDate: new Date().toISOString(),
        selectedClusters: Object.keys(data),
        totalSelected: Object.keys(data).length,
        data: data
    };
    
    const json = JSON.stringify(exportData, null, 2);
    this.downloadFile(json, 'pubg-cluster-analysis.json', 'application/json');
};

ClusterAnalysis.downloadFile = function(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

// ==================== 차트 업데이트 ====================
ClusterAnalysis.updateCharts = function() {
    this.renderDistributionChart();
    this.updateStatistics();
    this.renderCurrentTab();
};

// ==================== 이벤트 리스너 ====================
ClusterAnalysis.initEventListeners = function() {
    window.addEventListener('resize', this.debounce(() => {
        this.data.charts.forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }, 250));
};

// ==================== 유틸리티 ====================
ClusterAnalysis.debounce = function(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// ==================== 초기화 실행 ====================
ClusterAnalysis.init();

// ==================== 전역 접근 ====================
window.ClusterAnalysis = ClusterAnalysis;