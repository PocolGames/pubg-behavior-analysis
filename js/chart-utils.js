/**
 * PUBG 플레이어 행동 분석 웹사이트 - Chart.js 유틸리티 (수정된 버전)
 * Chart.js 기반 차트 생성 및 관리 함수들
 */

// ==================== Chart 유틸리티 객체 ====================
const ChartUtils = {
    charts: new Map(),
    defaultColors: {
        primary: '#ff6b35',
        secondary: '#667eea', 
        success: '#56ab2f',
        danger: '#dc3545',
        warning: '#ffc107',
        info: '#17a2b8',
        light: '#f8f9fa',
        dark: '#343a40'
    },
    playerTypeColors: {
        'Survivor': '#56ab2f',
        'Explorer': '#667eea', 
        'Aggressive': '#dc3545',
        'Balanced': '#ff6b35'
    }
};

// ==================== Chart.js 기본 설정 ====================
ChartUtils.initChartJS = function() {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.font.family = "'Noto Sans KR', sans-serif";
        Chart.defaults.color = '#e0e0e0';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
        Chart.defaults.backgroundColor = 'rgba(255, 107, 53, 0.1)';
        
        // 글로벌 플러그인 등록
        Chart.register({
            id: 'customTooltip',
            beforeDraw: function(chart) {
                if (chart.config.options.plugins?.customBackground) {
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
                    ctx.fillRect(0, 0, chart.width, chart.height);
                    ctx.restore();
                }
            }
        });
        
        console.log('✅ Chart.js 초기화 완료');
    } else {
        console.warn('⚠️ Chart.js가 로드되지 않았습니다.');
    }
};

// ==================== 공통 차트 옵션 ====================
ChartUtils.getBaseOptions = function(customOptions = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#e0e0e0',
                    font: {
                        size: 12,
                        weight: '500'
                    },
                    usePointStyle: true,
                    padding: 20
                }
            },
            tooltip: {
                backgroundColor: 'rgba(23, 25, 35, 0.95)',
                titleColor: '#ffffff',
                bodyColor: '#e0e0e0',
                borderColor: 'rgba(255, 107, 53, 0.5)',
                borderWidth: 1,
                cornerRadius: 8,
                displayColors: true,
                ...customOptions.tooltip
            }
        },
        scales: customOptions.scales || {},
        animation: {
            duration: 1000,
            easing: 'easeOutQuart'
        },
        ...customOptions
    };
};

// ==================== 도넛 차트 생성 ====================
ChartUtils.createDonutChart = function(canvasIdOrElement, data, options = {}) {
    // Canvas 요소 또는 ID 문자열 모두 처리
    let canvas;
    let canvasId;
    
    if (typeof canvasIdOrElement === 'string') {
        canvas = document.getElementById(canvasIdOrElement);
        canvasId = canvasIdOrElement;
    } else if (canvasIdOrElement instanceof HTMLCanvasElement) {
        canvas = canvasIdOrElement;
        canvasId = canvas.id || 'chart_' + Math.random().toString(36).substr(2, 9);
    } else {
        console.error('Invalid canvas parameter:', canvasIdOrElement);
        return null;
    }
    
    if (!canvas) {
        console.error(`Canvas with id '${canvasId}' not found`);
        return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    // 기존 차트 파괴
    if (this.charts.has(canvasId)) {
        this.charts.get(canvasId).destroy();
    }
    
    const chartOptions = this.getBaseOptions({
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    generateLabels: function(chart) {
                        const data = chart.data;
                        if (data.labels.length && data.datasets.length) {
                            return data.labels.map((label, i) => {
                                const dataset = data.datasets[0];
                                const value = dataset.data[i];
                                const total = dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                
                                return {
                                    text: `${label}: ${percentage}%`,
                                    fillStyle: dataset.backgroundColor[i],
                                    strokeStyle: dataset.borderColor[i],
                                    lineWidth: 2,
                                    hidden: false,
                                    index: i
                                };
                            });
                        }
                        return [];
                    }
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const label = context.label || '';
                        const value = context.parsed;
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = ((value / total) * 100).toFixed(1);
                        return `${label}: ${value.toLocaleString()}명 (${percentage}%)`;
                    }
                }
            }
        },
        ...options
    });
    
    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: chartOptions
    });
    
    this.charts.set(canvasId, chart);
    return chart;
};

// ==================== 레이더 차트 생성 ====================
ChartUtils.createRadarChart = function(canvasIdOrElement, data, options = {}) {
    // Canvas 요소 또는 ID 문자열 모두 처리
    let canvas;
    let canvasId;
    
    if (typeof canvasIdOrElement === 'string') {
        canvas = document.getElementById(canvasIdOrElement);
        canvasId = canvasIdOrElement;
    } else if (canvasIdOrElement instanceof HTMLCanvasElement) {
        canvas = canvasIdOrElement;
        canvasId = canvas.id || 'chart_' + Math.random().toString(36).substr(2, 9);
    } else {
        console.error('Invalid canvas parameter:', canvasIdOrElement);
        return null;
    }
    
    if (!canvas) {
        console.error(`Canvas with id '${canvasId}' not found`);
        return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    // 기존 차트 파괴
    if (this.charts.has(canvasId)) {
        this.charts.get(canvasId).destroy();
    }
    
    const chartOptions = this.getBaseOptions({
        scales: {
            r: {
                angleLines: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                pointLabels: {
                    color: '#e0e0e0',
                    font: {
                        size: 11
                    }
                },
                ticks: {
                    color: '#888',
                    backdropColor: 'transparent',
                    maxTicksLimit: 5
                },
                min: 0,
                max: options.maxValue || 100
            }
        },
        plugins: {
            legend: {
                position: 'top'
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `${context.dataset.label}: ${context.parsed.r.toFixed(2)}`;
                    }
                }
            }
        },
        ...options
    });
    
    const chart = new Chart(ctx, {
        type: 'radar',
        data: data,
        options: chartOptions
    });
    
    this.charts.set(canvasId, chart);
    return chart;
};

// ==================== 바 차트 생성 ====================
ChartUtils.createBarChart = function(canvasIdOrElement, data, options = {}) {
    // Canvas 요소 또는 ID 문자열 모두 처리
    let canvas;
    let canvasId;
    
    if (typeof canvasIdOrElement === 'string') {
        canvas = document.getElementById(canvasIdOrElement);
        canvasId = canvasIdOrElement;
    } else if (canvasIdOrElement instanceof HTMLCanvasElement) {
        canvas = canvasIdOrElement;
        canvasId = canvas.id || 'chart_' + Math.random().toString(36).substr(2, 9);
    } else {
        console.error('Invalid canvas parameter:', canvasIdOrElement);
        return null;
    }
    
    if (!canvas) {
        console.error(`Canvas with id '${canvasId}' not found`);
        return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    // 기존 차트 파괴
    if (this.charts.has(canvasId)) {
        this.charts.get(canvasId).destroy();
    }
    
    const isHorizontal = options.indexAxis === 'y';
    
    const chartOptions = this.getBaseOptions({
        indexAxis: options.indexAxis || 'x',
        scales: {
            x: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#e0e0e0'
                }
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#e0e0e0'
                }
            }
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const label = context.dataset.label || '';
                        const value = context.parsed[isHorizontal ? 'x' : 'y'];
                        return `${label}: ${value.toLocaleString()}`;
                    }
                }
            }
        },
        ...options
    });
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: chartOptions
    });
    
    this.charts.set(canvasId, chart);
    return chart;
};

// ==================== 라인 차트 생성 ====================
ChartUtils.createLineChart = function(canvasIdOrElement, data, options = {}) {
    // Canvas 요소 또는 ID 문자열 모두 처리
    let canvas;
    let canvasId;
    
    if (typeof canvasIdOrElement === 'string') {
        canvas = document.getElementById(canvasIdOrElement);
        canvasId = canvasIdOrElement;
    } else if (canvasIdOrElement instanceof HTMLCanvasElement) {
        canvas = canvasIdOrElement;
        canvasId = canvas.id || 'chart_' + Math.random().toString(36).substr(2, 9);
    } else {
        console.error('Invalid canvas parameter:', canvasIdOrElement);
        return null;
    }
    
    if (!canvas) {
        console.error(`Canvas with id '${canvasId}' not found`);
        return null;
    }
    
    const ctx = canvas.getContext('2d');
    
    // 기존 차트 파괴
    if (this.charts.has(canvasId)) {
        this.charts.get(canvasId).destroy();
    }
    
    const chartOptions = this.getBaseOptions({
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#e0e0e0'
                }
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#e0e0e0'
                }
            }
        },
        elements: {
            line: {
                tension: 0.4
            },
            point: {
                radius: 6,
                hoverRadius: 8
            }
        },
        ...options
    });
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: data,
        options: chartOptions
    });
    
    this.charts.set(canvasId, chart);
    return chart;
};

// ==================== 플레이어 클러스터 분포 차트 ====================
ChartUtils.createClusterDistributionChart = function(canvasId, clusterData) {
    const labels = [];
    const data = [];
    const backgroundColors = [];
    const borderColors = [];
    
    // 클러스터 데이터 처리
    Object.entries(clusterData).forEach(([clusterId, clusterInfo]) => {
        labels.push(clusterInfo.name || `클러스터 ${clusterId}`);
        data.push(clusterInfo.count || 0);
        
        // 플레이어 유형별 색상
        const baseColor = this.getPlayerTypeColor(clusterInfo.type || clusterInfo.name);
        backgroundColors.push(baseColor);
        borderColors.push(this.adjustColorOpacity(baseColor, 1));
    });
    
    return this.createDonutChart(canvasId, {
        labels: labels,
        datasets: [{
            data: data,
            backgroundColor: backgroundColors,
            borderColor: borderColors,
            borderWidth: 2
        }]
    });
};

// ==================== 플레이어 특성 레이더 차트 ====================
ChartUtils.createPlayerRadarChart = function(canvasId, playerData, clusterName) {
    const features = [
        '킬 수', '데미지', '이동거리', '생존력', 
        '어시스트', '무기획득', '부스트', '치료'
    ];
    
    const color = this.getPlayerTypeColor(clusterName);
    
    return this.createRadarChart(canvasId, {
        labels: features,
        datasets: [{
            label: clusterName,
            data: playerData,
            borderColor: color,
            backgroundColor: this.adjustColorOpacity(color, 0.2),
            borderWidth: 2,
            pointBackgroundColor: color,
            pointBorderColor: '#fff',
            pointBorderWidth: 2
        }]
    }, {
        maxValue: 100
    });
};

// ==================== 모델 성능 차트 ====================
ChartUtils.createModelPerformanceChart = function(canvasId, performanceData) {
    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
    const values = [
        performanceData.accuracy * 100,
        performanceData.precision * 100,
        performanceData.recall * 100,
        performanceData.f1Score * 100
    ];
    
    return this.createBarChart(canvasId, {
        labels: metrics,
        datasets: [{
            label: '성능 지표 (%)',
            data: values,
            backgroundColor: [
                'rgba(255, 107, 53, 0.8)',
                'rgba(102, 126, 234, 0.8)', 
                'rgba(86, 171, 47, 0.8)',
                'rgba(255, 193, 7, 0.8)'
            ],
            borderColor: [
                '#ff6b35',
                '#667eea',
                '#56ab2f', 
                '#ffc107'
            ],
            borderWidth: 2
        }]
    });
};

// ==================== 특성 중요도 차트 ====================
ChartUtils.createFeatureImportanceChart = function(canvasId, featuresData) {
    const labels = featuresData.map(item => item.feature);
    const importance = featuresData.map(item => item.importance * 100);
    
    return this.createBarChart(canvasId, {
        labels: labels,
        datasets: [{
            label: '특성 중요도 (%)',
            data: importance,
            backgroundColor: 'rgba(255, 107, 53, 0.8)',
            borderColor: '#ff6b35',
            borderWidth: 2
        }]
    }, {
        indexAxis: 'y',
        plugins: {
            legend: {
                display: false
            }
        }
    });
};

// ==================== 색상 유틸리티 ====================
ChartUtils.getPlayerTypeColor = function(type) {
    if (typeof type === 'string') {
        if (type.includes('Survivor')) {
            return this.playerTypeColors.Survivor;
        }
        if (type.includes('Explorer')) {
            return this.playerTypeColors.Explorer;
        }
        if (type.includes('Aggressive')) {
            return this.playerTypeColors.Aggressive;
        }
    }
    return this.defaultColors.primary;
};

ChartUtils.adjustColorOpacity = function(color, opacity) {
    // RGB/RGBA 색상을 투명도 조정
    if (color.startsWith('rgba')) {
        return color.replace(/[\d\.]+\)$/g, `${opacity})`);
    } else if (color.startsWith('rgb')) {
        return color.replace('rgb', 'rgba').replace(')', `, ${opacity})`);
    } else if (color.startsWith('#')) {
        const hex = color.replace('#', '');
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }
    return color;
};

ChartUtils.generateColorPalette = function(count) {
    const colors = [
        '#ff6b35', '#667eea', '#56ab2f', '#dc3545',
        '#ffc107', '#17a2b8', '#6f42c1', '#e83e8c'
    ];
    
    const palette = [];
    for (let i = 0; i < count; i++) {
        palette.push(colors[i % colors.length]);
    }
    return palette;
};

// ==================== 차트 관리 ====================
ChartUtils.updateChart = function(canvasId, newData) {
    const chart = this.charts.get(canvasId);
    if (chart) {
        chart.data = newData;
        chart.update('active');
    }
};

ChartUtils.resizeChart = function(canvasId) {
    const chart = this.charts.get(canvasId);
    if (chart) {
        chart.resize();
    }
};

ChartUtils.destroyChart = function(canvasId) {
    const chart = this.charts.get(canvasId);
    if (chart) {
        chart.destroy();
        this.charts.delete(canvasId);
    }
};

ChartUtils.destroyAllCharts = function() {
    this.charts.forEach((chart, canvasId) => {
        chart.destroy();
    });
    this.charts.clear();
};

// ==================== 반응형 차트 ====================
ChartUtils.initResponsiveCharts = function() {
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            this.charts.forEach((chart, canvasId) => {
                this.resizeChart(canvasId);
            });
        }, 250);
    });
};

// ==================== 애니메이션 효과 ====================
ChartUtils.animateCounter = function(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        
        if (element) {
            if (current % 1 === 0) {
                element.textContent = Math.floor(current).toLocaleString();
            } else {
                element.textContent = current.toFixed(1);
            }
        }
    }, 16);
};

// ==================== 데이터 내보내기 ====================
ChartUtils.exportChartAsImage = function(canvasId, filename = 'chart') {
    const chart = this.charts.get(canvasId);
    if (chart) {
        const url = chart.toBase64Image();
        const link = document.createElement('a');
        link.download = `${filename}.png`;
        link.href = url;
        link.click();
    }
};

ChartUtils.getChartData = function(canvasId) {
    const chart = this.charts.get(canvasId);
    return chart ? chart.data : null;
};

// ==================== 초기화 ====================
document.addEventListener('DOMContentLoaded', function() {
    ChartUtils.initChartJS();
    ChartUtils.initResponsiveCharts();
});

// ==================== 전역 접근 ====================
window.ChartUtils = ChartUtils;