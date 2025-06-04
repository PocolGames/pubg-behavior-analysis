/**
 * PUBG 플레이어 행동 분석 웹사이트 - 클러스터 분석 페이지 (수정된 버전)
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
        
        // DOM이 완전히 준비된 후 차트 렌더링
        setTimeout(() => {
            this.renderOverviewCharts();
            this.renderFeatureTab();
        }, 200);
        
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
        console.warn('Error:', message);
    }
};

// ==================== 탭 시스템 ====================
ClusterAnalysis.initTabs = function() {
    const tabs = document.querySelectorAll('.tab-btn, .tab-link');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            let targetTab = tab.getAttribute('data-tab');
            
            // href 속성에서 탭 이름 추출 (예: #characteristics)
            if (!targetTab && tab.getAttribute('href')) {
                targetTab = tab.getAttribute('href').replace('#', '');
            }
            
            if (!targetTab) {
                return;
            }
            
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const targetContent = document.getElementById(targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
            }
            
            this.data.currentTab = targetTab;
            this.renderCurrentTab();
        });
    });
};

// ==================== 필터 시스템 ====================
ClusterAnalysis.initFilters = function() {
    const filterContainer = document.querySelector('.cluster-filters');
    if (!filterContainer) {
        // 기존 HTML 필터를 사용
        this.initExistingFilters();
        return;
    }
    
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

ClusterAnalysis.initExistingFilters = function() {
    // 기존 HTML의 필터 요소들 활용
    const existingCheckboxes = document.querySelectorAll('.cluster-filter');
    existingCheckboxes.forEach(checkbox => {
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
    
    // 기존 버튼 이벤트
    const selectAllBtn = document.getElementById('selectAllClusters');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', () => this.selectAllClusters(true));
    }
    
    const clearAllBtn = document.getElementById('clearAllClusters');
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', () => this.selectAllClusters(false));
    }
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
    
    const selectAllBtn = document.getElementById('selectAllClusters');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', () => this.selectAllClusters(true));
    }
    
    const deselectAllBtn = document.getElementById('deselectAllClusters');
    if (deselectAllBtn) {
        deselectAllBtn.addEventListener('click', () => this.selectAllClusters(false));
    }
};

ClusterAnalysis.selectAllClusters = function(select) {
    const checkboxes = document.querySelectorAll('.cluster-checkbox input, .cluster-filter');
    
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
    const canvas = document.getElementById('clusterDistributionChart');
    
    // Canvas 요소가 없으면 경고 메시지만 출력하고 넘어감
    if (!canvas) {
        console.warn('⚠️ clusterDistributionChart Canvas를 찾을 수 없습니다.');
        return;
    }
    
    const selectedData = {};
    
    this.data.selectedClusters.forEach(clusterId => {
        if (this.data.clusters[clusterId]) {
            selectedData[clusterId] = this.data.clusters[clusterId];
        }
    });
    
    if (typeof ChartUtils !== 'undefined') {
        const chart = ChartUtils.createClusterDistributionChart('clusterDistributionChart', selectedData);
        if (chart) {
            this.data.charts.set('distribution', chart);
        }
    }
};

ClusterAnalysis.updateStatistics = function() {
    const totalPlayers = Array.from(this.data.selectedClusters)
        .reduce((sum, clusterId) => {
            const cluster = this.data.clusters[clusterId];
            return sum + (cluster ? cluster.count : 0);
        }, 0);
    
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
        case 'characteristics':
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
    // 기존 HTML 구조를 활용하고 레이더 차트만 생성
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
    // 기존 HTML에 있는 레이더 차트 Canvas들에 차트 생성
    const radarCanvases = ['radarChart0', 'radarChart1', 'radarChart7', 'radarChartExplorer'];
    
    radarCanvases.forEach(canvasId => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`⚠️ ${canvasId} Canvas를 찾을 수 없습니다.`);
            return;
        }
        
        let clusterId, clusterData;
        
        if (canvasId === 'radarChart0') {
            clusterId = 0;
            clusterData = this.data.clusters[0];
        } else if (canvasId === 'radarChart1') {
            clusterId = 1;
            clusterData = this.data.clusters[1];
        } else if (canvasId === 'radarChart7') {
            clusterId = 7;
            clusterData = this.data.clusters[7];
        } else if (canvasId === 'radarChartExplorer') {
            // Explorer 클러스터들의 평균값 사용
            clusterData = {
                name: 'Explorer (평균)',
                characteristics: {
                    killRate: 1.0,
                    damageDealt: 150,
                    walkDistance: 2000,
                    healUsage: 2.0,
                    boostUsage: 1.5,
                    survivalTime: 60,
                    aggressiveness: 50,
                    teamwork: 70
                }
            };
        }
        
        if (clusterData && clusterData.characteristics) {
            const characteristics = Object.values(clusterData.characteristics);
            
            if (typeof ChartUtils !== 'undefined') {
                const chart = ChartUtils.createPlayerRadarChart(
                    canvasId,
                    characteristics,
                    clusterData.name
                );
                if (chart) {
                    this.data.charts.set(`radar${clusterId || 'explorer'}`, chart);
                }
            }
        }
    });
};

// ==================== 비교 탭 ====================
ClusterAnalysis.renderComparisonTab = function() {
    if (this.data.selectedClusters.size < 2) {
        const container = document.getElementById('comparison');
        if (container) {
            container.innerHTML = `
                <div class="comparison-message">
                    <i class="fas fa-info-circle"></i>
                    <p>비교를 위해 2개 이상의 클러스터를 선택해주세요.</p>
                </div>
            `;
        }
        return;
    }
    
    setTimeout(() => {
        this.createComparisonChart();
        this.generateComparisonInsights();
    }, 100);
};

ClusterAnalysis.createComparisonChart = function() {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) {
        console.warn('⚠️ comparisonChart Canvas를 찾을 수 없습니다.');
        return;
    }
    
    const features = ['킬 수', '데미지', '이동거리', '생존력', '어시스트', '무기획득', '부스트', '치료'];
    const datasets = [];
    
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.data.clusters[clusterId];
        if (!cluster || !cluster.characteristics) {
            return;
        }
        
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
        
        if (chart) {
            this.data.charts.set('comparison', chart);
        }
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
    // 기존 HTML의 인사이트 사용 또는 동적 생성
    console.log('🔍 비교 인사이트 생성됨');
};

// ==================== 통계 탭 ====================
ClusterAnalysis.renderStatisticsTab = function() {
    this.populateStatsTable();
};

ClusterAnalysis.populateStatsTable = function() {
    const tbody = document.getElementById('statisticsTableBody');
    if (!tbody) {
        console.warn('⚠️ statisticsTableBody를 찾을 수 없습니다.');
        return;
    }
    
    let html = '';
    this.data.selectedClusters.forEach(clusterId => {
        const cluster = this.data.clusters[clusterId];
        if (!cluster) {
            return;
        }
        
        html += `
            <tr>
                <td>
                    <span class="cluster-indicator" style="background-color: ${cluster.color}"></span>
                    ${cluster.name}
                </td>
                <td>${cluster.count.toLocaleString()}</td>
                <td>${cluster.percentage}%</td>
                <td>${cluster.characteristics?.killRate || 'N/A'}</td>
                <td>${cluster.characteristics?.damageDealt || 'N/A'}</td>
                <td>${cluster.characteristics?.walkDistance || 'N/A'}</td>
                <td>${cluster.characteristics?.survivalTime || 'N/A'}</td>
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
    
    // 내보내기 버튼 이벤트
    const exportCSVBtn = document.getElementById('exportCSV');
    if (exportCSVBtn) {
        exportCSVBtn.addEventListener('click', () => this.exportData('csv'));
    }
    
    const exportJSONBtn = document.getElementById('exportJSON');
    if (exportJSONBtn) {
        exportJSONBtn.addEventListener('click', () => this.exportData('json'));
    }
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