// Dashboard UI Controller - 대시보드 UI 상호작용 관리
class DashboardUI {
    constructor() {
        this.currentCluster = 0;
        this.initializeUI();
    }

    // UI 초기화
    initializeUI() {
        this.updateMetrics();
        this.createClusterTabs();
        this.showClusterDetails(0);
    }

    // 메트릭 업데이트
    updateMetrics() {
        const metrics = DASHBOARD_DATA.modelMetrics;
        
        // 메인 메트릭 업데이트
        this.updateElement('modelAccuracy', `${(metrics.accuracy * 100).toFixed(3)}%`);
        this.updateElement('featureCount', metrics.featureCount.toString());
        this.updateElement('clusterCount', metrics.clusterCount.toString());
        this.updateElement('sampleCount', metrics.sampleCount.toLocaleString());
    }

    // 엘리먼트 업데이트 헬퍼 함수
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    // 클러스터 탭 생성
    createClusterTabs() {
        const tabsContainer = document.getElementById('clusterTabs');
        if (!tabsContainer) return;

        tabsContainer.innerHTML = '';

        Object.entries(DASHBOARD_DATA.clusters).forEach(([clusterId, clusterInfo]) => {
            const tab = document.createElement('div');
            tab.className = `cluster-tab ${clusterId == 0 ? 'active' : ''}`;
            tab.setAttribute('data-cluster', clusterId);
            tab.textContent = `${clusterInfo.name} ${clusterId}`;
            
            tab.addEventListener('click', () => {
                this.switchClusterTab(parseInt(clusterId));
            });
            
            tabsContainer.appendChild(tab);
        });
    }

    // 클러스터 탭 전환
    switchClusterTab(clusterId) {
        // 활성 탭 업데이트
        document.querySelectorAll('.cluster-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        const activeTab = document.querySelector(`[data-cluster="${clusterId}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }

        // 클러스터 상세 내용 표시
        this.showClusterDetails(clusterId);
        this.currentCluster = clusterId;
    }

    // 클러스터 상세 정보 표시
    showClusterDetails(clusterId) {
        const contentContainer = document.getElementById('clusterContent');
        if (!contentContainer) return;

        const clusterInfo = DASHBOARD_DATA.clusters[clusterId];
        if (!clusterInfo) return;

        contentContainer.innerHTML = `
            <div class="cluster-info">
                <div class="cluster-stats">
                    <h3>${clusterInfo.name} (${clusterInfo.type})</h3>
                    <p class="cluster-description">${clusterInfo.description}</p>
                    
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">플레이어 수</span>
                            <span class="stat-value">${clusterInfo.count.toLocaleString()}명</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">전체 비율</span>
                            <span class="stat-value">${clusterInfo.percentage}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">유형</span>
                            <span class="stat-value">${clusterInfo.type}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">대표 색상</span>
                            <span class="stat-value">
                                <div class="color-preview" style="background-color: ${clusterInfo.color}; width: 20px; height: 20px; border-radius: 50%; display: inline-block;"></div>
                            </span>
                        </div>
                    </div>
                </div>
                
                <div class="cluster-features">
                    <h4>주요 특성 (평균 대비 배수)</h4>
                    <div class="feature-list">
                        ${clusterInfo.keyFeatures.map(feature => `
                            <div class="feature-item">
                                <span class="feature-name">${translateFeatureName(feature.name)}</span>
                                <span class="feature-ratio">${feature.ratio.toFixed(2)}배</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        // 콘텐츠 표시
        contentContainer.className = 'cluster-content active';
    }

    // 차트 초기화
    initializeCharts() {
        if (!dashboardCharts) {
            dashboardCharts = new DashboardCharts();
        }
    }

    // 데이터 새로고침
    refreshData() {
        this.updateMetrics();
        if (dashboardCharts) {
            dashboardCharts.destroyAllCharts();
            dashboardCharts = new DashboardCharts();
        }
    }

    // 성능 요약 정보 생성
    generatePerformanceSummary() {
        const metrics = DASHBOARD_DATA.modelMetrics;
        const clusters = DASHBOARD_DATA.clusters;
        
        const totalPlayers = Object.values(clusters).reduce((sum, cluster) => sum + cluster.count, 0);
        const avgClusterSize = totalPlayers / Object.keys(clusters).length;
        
        return {
            accuracy: `${(metrics.accuracy * 100).toFixed(3)}%`,
            totalFeatures: metrics.featureCount,
            totalClusters: metrics.clusterCount,
            totalPlayers: totalPlayers.toLocaleString(),
            avgClusterSize: Math.round(avgClusterSize).toLocaleString(),
            silhouetteScore: metrics.silhouetteScore.toFixed(4)
        };
    }

    // 인사이트 생성
    generateInsights() {
        const clusters = DASHBOARD_DATA.clusters;
        const insights = [];
        
        // 가장 큰 클러스터
        const largestCluster = Object.values(clusters).reduce((max, cluster) => 
            cluster.count > max.count ? cluster : max
        );
        insights.push(`가장 큰 플레이어 그룹은 ${largestCluster.name} (${largestCluster.type})으로 전체의 ${largestCluster.percentage}%를 차지합니다.`);
        
        // 가장 작은 클러스터
        const smallestCluster = Object.values(clusters).reduce((min, cluster) => 
            cluster.count < min.count ? cluster : min
        );
        insights.push(`가장 희귀한 플레이어 유형은 ${smallestCluster.name} (${smallestCluster.type})으로 전체의 ${smallestCluster.percentage}%에 불과합니다.`);
        
        return insights;
    }
}

// 대시보드 UI 인스턴스
let dashboardUI = null;

// 페이지 로드시 초기화
document.addEventListener('DOMContentLoaded', function() {
    dashboardUI = new DashboardUI();
    
    // 차트 생성 약간의 지연
    setTimeout(() => {
        if (!dashboardCharts) {
            dashboardCharts = new DashboardCharts();
        }
    }, 500);
});

// 윈도우 리사이즈 이벤트
window.addEventListener('resize', function() {
    if (dashboardCharts) {
        // 차트 리사이즈 처리
        Object.values(dashboardCharts.charts).forEach(chart => {
            if (chart) {
                chart.resize();
            }
        });
    }
});

// 유틸리티 함수들
function formatNumber(num) {
    return num.toLocaleString();
}

function formatPercentage(num, decimals = 1) {
    return `${num.toFixed(decimals)}%`;
}

function getClusterColor(clusterId) {
    const cluster = DASHBOARD_DATA.clusters[clusterId];
    return cluster ? cluster.color : '#666666';
}

function getClusterInfo(clusterId) {
    return DASHBOARD_DATA.clusters[clusterId] || null;
}
