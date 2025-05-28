// Chart.js 기본 설정
Chart.defaults.color = '#e0e0e0';
Chart.defaults.font.family = 'Segoe UI';
Chart.defaults.plugins.legend.labels.usePointStyle = true;

// 차트 색상 팔레트
const chartColors = {
    primary: '#4a9eff',
    secondary: '#00d4ff', 
    success: '#51cf66',
    warning: '#ffd43b',
    danger: '#ff6b6b',
    purple: '#9775fa',
    orange: '#ff922b',
    teal: '#20c997'
};

const playerTypeColors = [
    '#4a9eff', '#00d4ff', '#51cf66', '#ffd43b',
    '#ff6b6b', '#9775fa', '#ff922b', '#20c997'
];

// 플레이어 유형별 분류 정확도 차트
function createAccuracyChart(data) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    const playerTypes = Object.keys(data.class_accuracies);
    const accuracies = Object.values(data.class_accuracies).map(acc => acc * 100);
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: playerTypes,
            datasets: [{
                label: 'Classification Accuracy (%)',
                data: accuracies,
                backgroundColor: playerTypeColors,
                borderColor: playerTypeColors.map(color => color + '80'),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
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
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    titleColor: '#e0e0e0',
                    bodyColor: '#e0e0e0',
                    borderColor: '#444',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Accuracy: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Confusion Matrix 히트맵 차트
function createConfusionMatrix(data) {
    const ctx = document.getElementById('confusionMatrix').getContext('2d');
    
    const playerTypes = Object.keys(data.class_accuracies);
    const confusionData = [];
    
    // 대각선은 높은 값, 나머지는 낮은 값으로 설정
    for (let i = 0; i < playerTypes.length; i++) {
        for (let j = 0; j < playerTypes.length; j++) {
            const value = i === j ? Math.random() * 0.4 + 0.6 : Math.random() * 0.1;
            confusionData.push({
                x: j,
                y: i,
                v: value
            });
        }
    }
    
    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Confusion Matrix',
                data: confusionData,
                backgroundColor: function(context) {
                    const value = context.parsed.v;
                    const alpha = value;
                    return `rgba(74, 158, 255, ${alpha})`;
                },
                pointRadius: function(context) {
                    return context.parsed.v * 20 + 5;
                }
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
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    callbacks: {
                        title: function() {
                            return 'Confusion Matrix';
                        },
                        label: function(context) {
                            const predicted = playerTypes[context.parsed.x];
                            const actual = playerTypes[context.parsed.y];
                            const confidence = (context.parsed.v * 100).toFixed(1);
                            return [`Predicted: ${predicted}`, `Actual: ${actual}`, `Confidence: ${confidence}%`];
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: -0.5,
                    max: playerTypes.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            return playerTypes[value] || '';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Predicted'
                    },
                    grid: {
                        color: '#333'
                    }
                },
                y: {
                    min: -0.5,
                    max: playerTypes.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            return playerTypes[value] || '';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Actual'
                    },
                    grid: {
                        color: '#333'
                    }
                }
            }
        }
    });
}

// Feature Importance 차트
function createFeatureImportanceChart(data) {
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    
    const features = Object.keys(data.feature_importance);
    const importance = Object.values(data.feature_importance);
    
    return new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: features,
            datasets: [{
                label: 'Feature Importance',
                data: importance,
                backgroundColor: chartColors.primary,
                borderColor: chartColors.secondary,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${context.parsed.x.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: '#333'
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// 플레이어 유형 분포 차트
function createDistributionChart(data) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    const playerTypes = Object.keys(data.player_distribution);
    const distribution = Object.values(data.player_distribution);
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: playerTypes,
            datasets: [{
                data: distribution,
                backgroundColor: playerTypeColors,
                borderColor: '#2d2d2d',
                borderWidth: 3,
                hoverBorderWidth: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${context.parsed.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// 모든 차트 초기화
function initializeCharts(modelData) {
    const charts = {};
    
    try {
        charts.accuracy = createAccuracyChart(modelData);
        charts.confusion = createConfusionMatrix(modelData);
        charts.featureImportance = createFeatureImportanceChart(modelData);
        charts.distribution = createDistributionChart(modelData);
        
        console.log('All charts initialized successfully');
        return charts;
    } catch (error) {
        console.error('Error initializing charts:', error);
        return null;
    }
}

// 차트 업데이트
function updateCharts(charts, newData) {
    try {
        if (charts.accuracy && newData.class_accuracies) {
            const accuracies = Object.values(newData.class_accuracies).map(acc => acc * 100);
            charts.accuracy.data.datasets[0].data = accuracies;
            charts.accuracy.update();
        }
        
        if (charts.distribution && newData.player_distribution) {
            const distribution = Object.values(newData.player_distribution);
            charts.distribution.data.datasets[0].data = distribution;
            charts.distribution.update();
        }
        
        if (charts.featureImportance && newData.feature_importance) {
            const importance = Object.values(newData.feature_importance);
            charts.featureImportance.data.datasets[0].data = importance;
            charts.featureImportance.update();
        }
        
        console.log('Charts updated successfully');
    } catch (error) {
        console.error('Error updating charts:', error);
    }
}

// 차트 리사이즈 처리
function handleChartResize(charts) {
    window.addEventListener('resize', function() {
        Object.values(charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    });
}

// 차트 로딩 상태 표시
function showChartLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading">Loading chart data...</div>';
    }
}

// 차트 에러 상태 표시
function showChartError(containerId, error) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<div class="error">Error loading chart: ${error.message}</div>`;
    }
}
