// Dashboard Charts - Chart.js를 이용한 차트 생성
class DashboardCharts {
    constructor() {
        this.charts = {};
        this.initializeCharts();
    }

    // 모든 차트 초기화
    initializeCharts() {
        this.createClusterDistributionChart();
        this.createFeatureImportanceChart();
        this.createAccuracyChart();
        this.createQualityChart();
    }

    // 클러스터 분포 도넛 차트
    createClusterDistributionChart() {
        const ctx = document.getElementById('clusterDistributionChart');
        if (!ctx) return;

        const data = getClusterDistributionData();
        
        this.charts.clusterDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.data,
                    backgroundColor: data.colors,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 2,
                    hoverBorderWidth: 3,
                    hoverBorderColor: 'rgba(255, 255, 255, 0.3)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: (context) => {
                                const label = context.label;
                                const value = context.parsed;
                                const count = data.counts[context.dataIndex];
                                return `${label}: ${value}% (${count.toLocaleString()}명)`;
                            }
                        }
                    }
                },
                cutout: '60%'
            }
        });

        // 범례 생성
        this.createClusterLegend(data);
    }

    // 클러스터 범례 생성
    createClusterLegend(data) {
        const legendContainer = document.getElementById('clusterLegend');
        if (!legendContainer) return;

        legendContainer.innerHTML = '';

        Object.entries(DASHBOARD_DATA.clusters).forEach((cluster, index) => {
            const [id, info] = cluster;
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            
            legendItem.innerHTML = `
                <div class="legend-color" style="background-color: ${info.color}"></div>
                <div class="legend-info">
                    <div class="legend-name">${info.name} (${info.type})</div>
                    <div class="legend-stats">${info.count.toLocaleString()}명 • ${info.percentage}%</div>
                </div>
            `;
            
            legendContainer.appendChild(legendItem);
        });
    }

    // 특성 중요도 수평 막대 차트
    createFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        const data = getTopFeatureImportance();
        
        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels.map(label => translateFeatureName(label)),
                datasets: [{
                    label: '중요도',
                    data: data.data,
                    backgroundColor: 'rgba(79, 195, 247, 0.6)',
                    borderColor: 'rgba(79, 195, 247, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#a0a0a0'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#ffffff',
                            font: {
                                size: 11
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: (context) => {
                                return `중요도: ${context.parsed.x.toFixed(4)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // 모델 정확도 게이지 차트
    createAccuracyChart() {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) return;

        const accuracy = DASHBOARD_DATA.modelMetrics.accuracy * 100;
        const remaining = 100 - accuracy;

        this.charts.accuracy = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [accuracy, remaining],
                    backgroundColor: [
                        'rgba(76, 175, 80, 0.8)',
                        'rgba(255, 255, 255, 0.1)'
                    ],
                    borderColor: [
                        'rgba(76, 175, 80, 1)',
                        'rgba(255, 255, 255, 0.2)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            },
            plugins: [{
                id: 'centerText',
                beforeDraw: (chart) => {
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.font = '24px Arial';
                    ctx.fillStyle = '#ffffff';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    
                    const centerX = chart.width / 2;
                    const centerY = chart.height / 2;
                    ctx.fillText(`${accuracy.toFixed(3)}%`, centerX, centerY);
                    ctx.restore();
                }
            }]
        });
    }

    // 클러스터링 품질 점수 차트
    createQualityChart() {
        const ctx = document.getElementById('qualityChart');
        if (!ctx) return;

        const silhouetteScore = DASHBOARD_DATA.modelMetrics.silhouetteScore;
        const normalizedScore = (silhouetteScore + 1) / 2 * 100;

        this.charts.quality = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [normalizedScore, 100 - normalizedScore],
                    backgroundColor: [
                        'rgba(255, 152, 0, 0.8)',
                        'rgba(255, 255, 255, 0.1)'
                    ],
                    borderColor: [
                        'rgba(255, 152, 0, 1)',
                        'rgba(255, 255, 255, 0.2)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            },
            plugins: [{
                id: 'centerText',
                beforeDraw: (chart) => {
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.font = '20px Arial';
                    ctx.fillStyle = '#ffffff';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    
                    const centerX = chart.width / 2;
                    const centerY = chart.height / 2;
                    ctx.fillText(silhouetteScore.toFixed(4), centerX, centerY);
                    ctx.restore();
                }
            }]
        });
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

// 차트 인스턴스 전역 변수
let dashboardCharts = null;
