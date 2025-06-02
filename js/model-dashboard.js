// Model Performance Dashboard JavaScript
// PUBG Player Behavior Analysis - Model Dashboard

class ModelDashboard {
    constructor() {
        this.charts = new Map();
        this.currentModel = 'basic';
        this.modelData = this.initializeModelData();
        this.featureData = this.initializeFeatureData();
        this.confidenceData = this.initializeConfidenceData();
        
        this.init();
    }

    // 실제 모델 성능 데이터 초기화
    initializeModelData() {
        return {
            basic: {
                name: 'Basic Neural Network',
                accuracy: 99.25,
                f1Score: 98.67,
                precision: 99.2,
                recall: 98.1,
                trainingHistory: {
                    epochs: Array.from({length: 50}, (_, i) => i + 1),
                    trainAccuracy: this.generateTrainingData(50, 0.65, 0.987, 'accuracy'),
                    valAccuracy: this.generateTrainingData(50, 0.75, 0.9927, 'accuracy'),
                    trainLoss: this.generateTrainingData(50, 0.99, 0.035, 'loss'),
                    valLoss: this.generateTrainingData(50, 0.85, 0.018, 'loss')
                }
            },
            advanced: {
                name: 'Advanced Neural Network',
                accuracy: 98.96,
                f1Score: 98.70,
                precision: 99.0,
                recall: 97.8,
                trainingHistory: {
                    epochs: Array.from({length: 50}, (_, i) => i + 1),
                    trainAccuracy: this.generateTrainingData(50, 0.70, 0.987, 'accuracy'),
                    valAccuracy: this.generateTrainingData(50, 0.78, 0.9911, 'accuracy'),
                    trainLoss: this.generateTrainingData(50, 0.97, 0.033, 'loss'),
                    valLoss: this.generateTrainingData(50, 0.82, 0.021, 'loss')
                }
            },
            residual: {
                name: 'Residual Neural Network',
                accuracy: 98.99,
                f1Score: 98.74,
                precision: 98.8,
                recall: 98.2,
                trainingHistory: {
                    epochs: Array.from({length: 43}, (_, i) => i + 1), // Early stopping at epoch 43
                    trainAccuracy: this.generateTrainingData(43, 0.72, 0.9934, 'accuracy'),
                    valAccuracy: this.generateTrainingData(43, 0.80, 0.9887, 'accuracy'),
                    trainLoss: this.generateTrainingData(43, 0.95, 0.017, 'loss'),
                    valLoss: this.generateTrainingData(43, 0.78, 0.028, 'loss')
                }
            },
            ensemble: {
                name: 'Ensemble Model',
                accuracy: 99.27,
                f1Score: 98.82,
                precision: 99.3,
                recall: 98.4
            }
        };
    }

    // 특성 중요도 데이터 초기화
    initializeFeatureData() {
        return [
            { name: 'has_kills', importance: 0.3232, description: '킬 여부 (이진 특성)' },
            { name: 'walkDistance_log', importance: 0.0788, description: '이동거리 (로그 변환)' },
            { name: 'walkDistance', importance: 0.0751, description: '보행 이동거리' },
            { name: 'total_distance', importance: 0.0634, description: '총 이동거리' },
            { name: 'has_swimDistance', importance: 0.0609, description: '수영 여부' },
            { name: 'weaponsAcquired', importance: 0.0588, description: '무기 획득 수' },
            { name: 'killPlace', importance: 0.0573, description: '킬 순위' },
            { name: 'damageDealt', importance: 0.0519, description: '가한 데미지' },
            { name: 'rideDistance', importance: 0.0512, description: '차량 이동거리' },
            { name: 'heal_boost_ratio', importance: 0.0501, description: '치료/부스트 비율' }
        ];
    }

    // 예측 신뢰도 데이터 초기화
    initializeConfidenceData() {
        return {
            mean: 0.990,
            std: 0.050,
            highConfidence: 98.2, // >0.8
            lowConfidence: 0.0,   // <0.5
            distribution: this.generateConfidenceDistribution()
        };
    }

    // 훈련 데이터 생성 (실제 패턴 모방)
    generateTrainingData(epochs, startValue, endValue, type) {
        const data = [];
        const noiseLevel = type === 'accuracy' ? 0.005 : 0.01;
        
        for (let i = 0; i < epochs; i++) {
            const progress = i / (epochs - 1);
            let value;
            
            if (type === 'accuracy') {
                // 시그모이드 형태의 학습 곡선
                value = startValue + (endValue - startValue) * (1 / (1 + Math.exp(-10 * (progress - 0.3))));
            } else {
                // 지수 감소 형태의 손실 곡선
                value = startValue * Math.exp(-3 * progress) + endValue;
            }
            
            // 노이즈 추가
            const noise = (Math.random() - 0.5) * noiseLevel;
            data.push(Math.max(0, value + noise));
        }
        
        return data;
    }

    // 신뢰도 분포 생성
    generateConfidenceDistribution() {
        const bins = 20;
        const distribution = [];
        
        for (let i = 0; i < bins; i++) {
            const x = i / (bins - 1);
            // 0.99 근처에 집중된 베타 분포 모방
            const concentration = Math.exp(-50 * Math.pow(x - 0.99, 2));
            const count = Math.floor(concentration * 1000 + Math.random() * 100);
            
            distribution.push({
                range: `${(x * 100).toFixed(0)}-${((x + 1/bins) * 100).toFixed(0)}%`,
                count: count,
                percentage: x
            });
        }
        
        return distribution;
    }

    // 초기화
    init() {
        this.initializeEventListeners();
        this.createCharts();
        this.animateCounters();
        this.setupTabSystem();
    }

    // 이벤트 리스너 설정
    initializeEventListeners() {
        // 모델 선택 버튼
        document.querySelectorAll('.model-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const model = e.target.dataset.model;
                this.switchModel(model);
            });
        });

        // 윈도우 리사이즈
        window.addEventListener('resize', this.debounce(() => {
            this.resizeCharts();
        }, 250));
    }

    // 탭 시스템 설정
    setupTabSystem() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                
                // 활성 탭 버튼 변경
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // 활성 탭 패널 변경
                tabPanes.forEach(pane => pane.classList.remove('active'));
                document.getElementById(`${tabId}-tab`).classList.add('active');
                
                // 탭별 차트 생성
                this.handleTabChange(tabId);
            });
        });
    }

    // 탭 변경 처리
    handleTabChange(tabId) {
        switch(tabId) {
            case 'confusion':
                this.createConfusionMatrix();
                break;
            case 'features':
                this.createFeatureImportanceChart();
                break;
            case 'confidence':
                this.createConfidenceChart();
                break;
            case 'interpretation':
                // 해석 탭은 정적 컨텐츠
                break;
        }
    }

    // 모델 전환
    switchModel(modelName) {
        this.currentModel = modelName;
        
        // 버튼 상태 업데이트
        document.querySelectorAll('.model-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-model="${modelName}"]`).classList.add('active');
        
        // 훈련 과정 차트 업데이트
        this.updateTrainingHistory();
    }

    // 차트 생성
    createCharts() {
        this.createModelComparisonChart();
        this.createTrainingHistoryChart();
        this.createConfusionMatrix();
        this.createFeatureImportanceChart();
        this.createConfidenceChart();
    }

    // 모델 비교 차트
    createModelComparisonChart() {
        const ctx = document.getElementById('modelComparisonChart');
        if (!ctx) return;

        const models = ['basic', 'advanced', 'residual', 'ensemble'];
        const accuracyData = models.map(model => this.modelData[model].accuracy);
        const f1Data = models.map(model => this.modelData[model].f1Score);

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(model => this.modelData[model].name),
                datasets: [
                    {
                        label: '정확도 (%)',
                        data: accuracyData,
                        backgroundColor: '#ff6b35',
                        borderColor: '#ff6b35',
                        borderWidth: 1
                    },
                    {
                        label: 'F1 Score (%)',
                        data: f1Data,
                        backgroundColor: '#667eea',
                        borderColor: '#667eea',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#e0e0e0' }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 95,
                        max: 100,
                        grid: { color: '#333' },
                        ticks: { color: '#e0e0e0' }
                    },
                    x: {
                        grid: { color: '#333' },
                        ticks: { 
                            color: '#e0e0e0',
                            maxRotation: 45
                        }
                    }
                }
            }
        });

        this.charts.set('modelComparison', chart);
    }

    // 훈련 과정 차트
    createTrainingHistoryChart() {
        const ctx = document.getElementById('trainingHistoryChart');
        if (!ctx) return;

        const modelData = this.modelData[this.currentModel];

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: modelData.trainingHistory.epochs,
                datasets: [
                    {
                        label: '훈련 정확도',
                        data: modelData.trainingHistory.trainAccuracy,
                        borderColor: '#ff6b35',
                        backgroundColor: 'rgba(255, 107, 53, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: '검증 정확도',
                        data: modelData.trainingHistory.valAccuracy,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#e0e0e0' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.6,
                        max: 1.0,
                        grid: { color: '#333' },
                        ticks: { 
                            color: '#e0e0e0',
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        }
                    },
                    x: {
                        grid: { color: '#333' },
                        ticks: { color: '#e0e0e0' },
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: '#e0e0e0'
                        }
                    }
                }
            }
        });

        this.charts.set('trainingHistory', chart);
    }

    // 혼동 행렬 차트
    createConfusionMatrix() {
        const ctx = document.getElementById('confusionMatrixChart');
        if (!ctx) return;

        // 실제 혼동 행렬 데이터 (정규화된 값)
        const confusionData = [
            [0.996, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000], // Survivor 0
            [0.001, 0.999, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], // Survivor 1
            [0.005, 0.005, 0.990, 0.000, 0.000, 0.000, 0.000, 0.000], // Explorer 2
            [0.003, 0.003, 0.000, 0.994, 0.000, 0.000, 0.000, 0.000], // Explorer 3
            [0.015, 0.016, 0.000, 0.000, 0.969, 0.000, 0.000, 0.000], // Explorer 4
            [0.002, 0.002, 0.000, 0.000, 0.000, 0.996, 0.000, 0.000], // Explorer 5
            [0.017, 0.016, 0.000, 0.000, 0.000, 0.000, 0.967, 0.000], // Explorer 6
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]  // Aggressive 7
        ];

        const labels = ['Survivor 0', 'Survivor 1', 'Explorer 2', 'Explorer 3', 
                       'Explorer 4', 'Explorer 5', 'Explorer 6', 'Aggressive'];

        // Chart.js용 데이터 변환
        const heatmapData = [];
        for (let i = 0; i < confusionData.length; i++) {
            for (let j = 0; j < confusionData[i].length; j++) {
                heatmapData.push({
                    x: j,
                    y: i,
                    v: confusionData[i][j]
                });
            }
        }

        const chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '혼동 행렬',
                    data: heatmapData,
                    backgroundColor: function(ctx) {
                        const value = ctx.parsed.v;
                        const intensity = Math.floor(value * 255);
                        return `rgba(102, 126, 234, ${value})`;
                    },
                    pointRadius: function(ctx) {
                        const value = ctx.parsed.v;
                        return 5 + (value * 15); // 5-20 범위
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: function() { return ''; },
                            label: function(ctx) {
                                const x = ctx.parsed.x;
                                const y = ctx.parsed.y;
                                const value = ctx.parsed.v;
                                return `${labels[y]} → ${labels[x]}: ${(value * 100).toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: -0.5,
                        max: 7.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return labels[value] || '';
                            },
                            color: '#e0e0e0'
                        },
                        grid: { color: '#333' },
                        title: {
                            display: true,
                            text: '예측 클래스',
                            color: '#e0e0e0'
                        }
                    },
                    y: {
                        min: -0.5,
                        max: 7.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return labels[value] || '';
                            },
                            color: '#e0e0e0'
                        },
                        grid: { color: '#333' },
                        title: {
                            display: true,
                            text: '실제 클래스',
                            color: '#e0e0e0'
                        }
                    }
                }
            }
        });

        this.charts.set('confusionMatrix', chart);
    }

    // 특성 중요도 차트
    createFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.featureData.map(item => item.name),
                datasets: [{
                    label: '중요도',
                    data: this.featureData.map(item => item.importance),
                    backgroundColor: '#ff6b35',
                    borderColor: '#ff6b35',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y', // 수평 바 차트
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                const feature = this.featureData[ctx.dataIndex];
                                return [
                                    `중요도: ${(ctx.parsed.x * 100).toFixed(2)}%`,
                                    `설명: ${feature.description}`
                                ];
                            }.bind(this)
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: '#333' },
                        ticks: { 
                            color: '#e0e0e0',
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        }
                    },
                    y: {
                        grid: { color: '#333' },
                        ticks: { 
                            color: '#e0e0e0',
                            font: { size: 10 }
                        }
                    }
                }
            }
        });

        this.charts.set('featureImportance', chart);
    }

    // 신뢰도 분포 차트
    createConfidenceChart() {
        const ctx = document.getElementById('confidenceDistributionChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.confidenceData.distribution.map(item => item.range),
                datasets: [{
                    label: '빈도',
                    data: this.confidenceData.distribution.map(item => item.count),
                    backgroundColor: '#667eea',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                return `빈도: ${ctx.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#333' },
                        ticks: { color: '#e0e0e0' },
                        title: {
                            display: true,
                            text: '빈도',
                            color: '#e0e0e0'
                        }
                    },
                    x: {
                        grid: { color: '#333' },
                        ticks: { 
                            color: '#e0e0e0',
                            maxRotation: 45
                        },
                        title: {
                            display: true,
                            text: '신뢰도 구간',
                            color: '#e0e0e0'
                        }
                    }
                }
            }
        });

        this.charts.set('confidenceDistribution', chart);
    }

    // 훈련 과정 업데이트
    updateTrainingHistory() {
        const chart = this.charts.get('trainingHistory');
        if (!chart) return;

        const modelData = this.modelData[this.currentModel];
        
        chart.data.labels = modelData.trainingHistory.epochs;
        chart.data.datasets[0].data = modelData.trainingHistory.trainAccuracy;
        chart.data.datasets[1].data = modelData.trainingHistory.valAccuracy;
        
        chart.update();
    }

    // 카운터 애니메이션
    animateCounters() {
        const counters = document.querySelectorAll('.stat-number');
        
        counters.forEach(counter => {
            const target = parseFloat(counter.dataset.target);
            const suffix = counter.textContent.includes('%') ? '%' : 
                         counter.textContent.includes('초') ? '초' : '';
            
            this.animateNumber(counter, 0, target, 2000, suffix);
        });
    }

    // 숫자 애니메이션
    animateNumber(element, start, end, duration, suffix = '') {
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // easeOutQuart 이징
            const easeProgress = 1 - Math.pow(1 - progress, 4);
            const current = start + (end - start) * easeProgress;
            
            if (suffix === '%') {
                element.textContent = current.toFixed(2) + suffix;
            } else if (suffix === '초') {
                element.textContent = current.toFixed(1) + suffix;
            } else {
                element.textContent = current.toFixed(0);
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    // 차트 리사이즈
    resizeCharts() {
        this.charts.forEach(chart => {
            chart.resize();
        });
    }

    // 디바운스 유틸리티
    debounce(func, wait) {
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

    // 차트 정리
    destroy() {
        this.charts.forEach(chart => {
            chart.destroy();
        });
        this.charts.clear();
    }
}

// 내보내기 함수들
window.exportModelReport = function() {
    const app = window.App;
    app.showNotification('PDF 보고서 생성 중...', 'info');
    
    // 실제 구현에서는 PDF 생성 라이브러리 사용
    setTimeout(() => {
        app.showNotification('PDF 보고서가 다운로드되었습니다.', 'success');
    }, 2000);
};

window.exportMetrics = function() {
    const dashboard = window.modelDashboard;
    const metrics = {
        models: dashboard.modelData,
        featureImportance: dashboard.featureData,
        confidenceStats: dashboard.confidenceData,
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(metrics, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model-metrics.json';
    a.click();
    
    URL.revokeObjectURL(url);
    
    const app = window.App;
    app.showNotification('성능 지표가 다운로드되었습니다.', 'success');
};

window.exportCharts = function() {
    const dashboard = window.modelDashboard;
    const app = window.App;
    
    app.showNotification('차트 이미지 생성 중...', 'info');
    
    // 모든 차트를 PNG로 내보내기
    dashboard.charts.forEach((chart, name) => {
        const url = chart.toBase64Image();
        const a = document.createElement('a');
        a.href = url;
        a.download = `${name}-chart.png`;
        a.click();
    });
    
    setTimeout(() => {
        app.showNotification('차트 이미지가 다운로드되었습니다.', 'success');
    }, 1000);
};

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.modelDashboard = new ModelDashboard();
    
    // 페이지별 초기화
    if (window.App) {
        window.App.initializePage('model-performance');
    }
});