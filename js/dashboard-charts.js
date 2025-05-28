// 대시보드 차트 생성 및 관리
class DashboardCharts {
    constructor() {
        this.charts = {};
        this.chartConfigs = this.getChartConfigs();
    }

    // 차트 기본 설정
    getChartConfigs() {
        return {
            defaultOptions: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#e2e8f0',
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: '#334155'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: '#334155'
                        }
                    }
                }
            }
        };
    }

    // 클러스터 분포 도넛 차트 생성
    createClusterDistributionChart() {
        const ctx = document.getElementById('clusterDistributionChart');
        if (!ctx) return;

        const data = {
            labels: ModelData.clusters.map(cluster => cluster.name),
            datasets: [{
                data: ModelData.clusters.map(cluster => cluster.percentage),
                backgroundColor: ModelData.clusters.map(cluster => cluster.color),
                borderColor: '#1e293b',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        };

        const options = {
            ...this.chartConfigs.defaultOptions,
            plugins: {
                ...this.chartConfigs.defaultOptions.plugins,
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const cluster = ModelData.clusters[context.dataIndex];
                            return `${cluster.name}: ${cluster.percentage}% (${cluster.count.toLocaleString()}명)`;
                        }
                    }
                }
            }
        };

        this.charts.clusterDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: options
        });
    }

    // 클러스터별 정확도 바 차트 생성
    createClusterAccuracyChart() {
        const ctx = document.getElementById('clusterAccuracyChart');
        if (!ctx) return;

        const data = {
            labels: ModelData.clusters.map(cluster => cluster.name),
            datasets: [{
                label: '분류 정확도',
                data: ModelData.clusters.map(cluster => cluster.accuracy * 100),
                backgroundColor: ModelData.clusters.map(cluster => cluster.color + '80'),
                borderColor: ModelData.clusters.map(cluster => cluster.color),
                borderWidth: 1
            }]
        };

        const options = {
            ...this.chartConfigs.defaultOptions,
            scales: {
                ...this.chartConfigs.defaultOptions.scales,
                y: {
                    ...this.chartConfigs.defaultOptions.scales.y,
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        ...this.chartConfigs.defaultOptions.scales.y.ticks,
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                ...this.chartConfigs.defaultOptions.plugins,
                legend: {
                    display: false
                }
            }
        };

        this.charts.clusterAccuracy = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: options
        });
    }

    // 특성 중요도 수평 바 차트 생성
    createFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        const topFeatures = ModelDataUtils.getTopFeatures(15);
        
        const data = {
            labels: topFeatures.map(feature => feature.name),
            datasets: [{
                label: '특성 중요도',
                data: topFeatures.map(feature => feature.importance),
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: '#3b82f6',
                borderWidth: 1
            }]
        };

        const options = {
            ...this.chartConfigs.defaultOptions,
            indexAxis: 'y',
            plugins: {
                ...this.chartConfigs.defaultOptions.plugins,
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    ...this.chartConfigs.defaultOptions.scales.x,
                    beginAtZero: true
                },
                y: {
                    ...this.chartConfigs.defaultOptions.scales.y
                }
            }
        };

        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: options
        });
    }

    // 모든 차트 초기화
    initializeAllCharts() {
        this.createClusterDistributionChart();
        this.createClusterAccuracyChart();
        this.createFeatureImportanceChart();
    }

    // 차트 업데이트
    updateChart(chartName, newData) {
        if (this.charts[chartName]) {
            this.charts[chartName].data = newData;
            this.charts[chartName].update();
        }
    }

    // 차트 제거
    destroyChart(chartName) {
        if (this.charts[chartName]) {
            this.charts[chartName].destroy();
            delete this.charts[chartName];
        }
    }

    // 모든 차트 제거
    destroyAllCharts() {
        Object.keys(this.charts).forEach(chartName => {
            this.destroyChart(chartName);
        });
    }
}

// Confusion Matrix 생성 함수
function createConfusionMatrix() {
    const container = document.getElementById('confusionMatrix');
    if (!container) return;

    container.innerHTML = '';
    
    ModelData.confusionMatrix.forEach((row, i) => {
        row.forEach((value, j) => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.textContent = (value * 100).toFixed(1) + '%';
            
            // 색상 강도 설정 (대각선은 진한색, 오차는 연한색)
            const intensity = i === j ? value : value * 0.5;
            const color = i === j ? 
                `rgba(34, 197, 94, ${intensity})` : 
                `rgba(239, 68, 68, ${intensity})`;
            
            cell.style.backgroundColor = color;
            container.appendChild(cell);
        });
    });
}

// 클러스터 범례 생성 함수
function createClusterLegend() {
    const container = document.getElementById('clusterLegend');
    if (!container) return;

    container.innerHTML = '<h3 style="color: #f1f5f9; margin-bottom: 1rem;">플레이어 유형</h3>';
    
    const stats = ModelDataUtils.getClusterStats();
    
    Object.entries(stats).forEach(([label, data]) => {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        legendItem.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: rgba(51, 65, 85, 0.5);
            border-radius: 8px;
            border-left: 4px solid ${data.clusters[0].color};
        `;
        
        legendItem.innerHTML = `
            <div>
                <div style="color: #f1f5f9; font-weight: 600;">${label}</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">${data.clusters.length}개 하위 클러스터</div>
            </div>
            <div style="text-align: right;">
                <div style="color: #f1f5f9; font-weight: 600;">${data.percentage.toFixed(1)}%</div>
                <div style="color: #94a3b8; font-size: 0.875rem;">${data.count.toLocaleString()}명</div>
            </div>
        `;
        
        container.appendChild(legendItem);
    });
}

// 상위 특성 목록 생성 함수
function createTopFeaturesList() {
    const container = document.getElementById('featuresList');
    if (!container) return;

    const topFeatures = ModelDataUtils.getTopFeatures(10);
    
    container.innerHTML = '';
    
    topFeatures.forEach((feature, index) => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        
        featureItem.innerHTML = `
            <div>
                <span style="color: #64748b; font-size: 0.875rem;">#${index + 1}</span>
                <span class="feature-name">${feature.name}</span>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 2px;">
                    ${feature.description}
                </div>
            </div>
            <span class="feature-importance">${feature.importance.toFixed(4)}</span>
        `;
        
        container.appendChild(featureItem);
    });
}
