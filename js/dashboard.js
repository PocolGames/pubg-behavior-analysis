// 메인 대시보드 초기화 및 관리
class Dashboard {
    constructor() {
        this.charts = new DashboardCharts();
        this.currentCluster = 0;
        this.init();
    }

    // 대시보드 초기화
    init() {
        this.updateMetrics();
        this.setupClusterTabs();
        this.initializeComponents();
        
        // DOM이 완전히 로드된 후 차트 생성
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.initializeCharts();
            });
        } else {
            this.initializeCharts();
        }
    }

    // 메트릭 업데이트
    updateMetrics() {
        const elements = {
            modelAccuracy: document.getElementById('modelAccuracy'),
            clusterCount: document.getElementById('clusterCount'),
            featureCount: document.getElementById('featureCount'),
            totalPlayers: document.getElementById('totalPlayers')
        };

        if (elements.modelAccuracy) {
            elements.modelAccuracy.textContent = (ModelData.model.accuracy * 100).toFixed(3) + '%';
        }
        if (elements.clusterCount) {
            elements.clusterCount.textContent = ModelData.model.cluster_count + '개';
        }
        if (elements.featureCount) {
            elements.featureCount.textContent = ModelData.model.feature_count + '개';
        }
        if (elements.totalPlayers) {
            elements.totalPlayers.textContent = ModelData.model.total_players.toLocaleString() + '+';
        }
    }

    // 클러스터 탭 설정
    setupClusterTabs() {
        const tabsContainer = document.getElementById('clusterTabs');
        if (!tabsContainer) return;

        tabsContainer.innerHTML = '';

        ModelData.clusters.forEach(cluster => {
            const tab = document.createElement('div');
            tab.className = `cluster-tab ${cluster.id === this.currentCluster ? 'active' : ''}`;
            tab.textContent = cluster.name;
            tab.style.borderLeftColor = cluster.color;
            
            tab.addEventListener('click', () => {
                this.selectCluster(cluster.id);
            });
            
            tabsContainer.appendChild(tab);
        });
    }

    // 클러스터 선택
    selectCluster(clusterId) {
        this.currentCluster = clusterId;
        
        // 탭 활성화 업데이트
        document.querySelectorAll('.cluster-tab').forEach((tab, index) => {
            tab.classList.toggle('active', index === clusterId);
        });
        
        // 클러스터 상세 정보 업데이트
        this.updateClusterDetails(clusterId);
    }

    // 클러스터 상세 정보 업데이트
    updateClusterDetails(clusterId) {
        const container = document.getElementById('clusterContent');
        if (!container) return;

        const cluster = ModelDataUtils.getClusterById(clusterId);
        if (!cluster) return;

        container.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div>
                    <h3 style="color: ${cluster.color}; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 12px; height: 12px; background: ${cluster.color}; border-radius: 50%;"></div>
                        ${cluster.name}
                    </h3>
                    <div style="display: grid; gap: 1rem;">
                        <div class="detail-card">
                            <div class="detail-label">플레이어 수</div>
                            <div class="detail-value">${cluster.count.toLocaleString()}명</div>
                        </div>
                        <div class="detail-card">
                            <div class="detail-label">전체 비율</div>
                            <div class="detail-value">${cluster.percentage}%</div>
                        </div>
                        <div class="detail-card">
                            <div class="detail-label">분류 정확도</div>
                            <div class="detail-value">${(cluster.accuracy * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
                <div>
                    <h4 style="color: #f1f5f9; margin-bottom: 1rem;">주요 특성</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        ${cluster.characteristics.map(char => `
                            <div style="padding: 0.75rem; background: rgba(51, 65, 85, 0.5); border-radius: 8px; border-left: 3px solid ${cluster.color};">
                                <div style="font-size: 0.875rem; color: #e2e8f0;">${char}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        // 상세 카드 스타일 추가
        const style = document.createElement('style');
        style.textContent = `
            .detail-card {
                padding: 1rem;
                background: rgba(51, 65, 85, 0.3);
                border-radius: 8px;
                border: 1px solid #475569;
            }
            .detail-label {
                font-size: 0.875rem;
                color: #94a3b8;
                margin-bottom: 0.25rem;
            }
            .detail-value {
                font-size: 1.25rem;
                font-weight: 600;
                color: #f1f5f9;
            }
        `;
        if (!document.querySelector('style[data-detail-cards]')) {
            style.setAttribute('data-detail-cards', 'true');
            document.head.appendChild(style);
        }
    }

    // 컴포넌트 초기화
    initializeComponents() {
        createConfusionMatrix();
        createClusterLegend();
        createTopFeaturesList();
        this.updateClusterDetails(this.currentCluster);
    }

    // 차트 초기화
    initializeCharts() {
        // 약간의 지연을 주어 DOM이 완전히 준비되도록 함
        setTimeout(() => {
            this.charts.initializeAllCharts();
        }, 100);
    }

    // 대시보드 새로고침
    refresh() {
        this.charts.destroyAllCharts();
        this.initializeCharts();
        this.initializeComponents();
    }

    // 데이터 내보내기
    exportData() {
        const exportData = {
            model: ModelData.model,
            clusters: ModelData.clusters,
            features: ModelData.features,
            insights: ModelData.insights,
            timestamp: new Date().toISOString()
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'pubg-model-performance.json';
        link.click();
    }
}

// 대시보드 초기화
let dashboard;

// DOM 로드 후 대시보드 초기화
document.addEventListener('DOMContentLoaded', function() {
    dashboard = new Dashboard();
    
    // 키보드 단축키 설정
    document.addEventListener('keydown', function(e) {
        // Ctrl+R: 새로고침
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            dashboard.refresh();
        }
        // Ctrl+E: 데이터 내보내기
        if (e.ctrlKey && e.key === 'e') {
            e.preventDefault();
            dashboard.exportData();
        }
    });
});

// 윈도우 리사이즈 시 차트 재조정
window.addEventListener('resize', function() {
    if (dashboard && dashboard.charts) {
        Object.values(dashboard.charts.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
});

// 페이지 언로드 시 정리
window.addEventListener('beforeunload', function() {
    if (dashboard && dashboard.charts) {
        dashboard.charts.destroyAllCharts();
    }
});
