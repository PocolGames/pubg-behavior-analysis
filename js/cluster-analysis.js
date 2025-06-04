/**
 * PUBG Cluster Analysis Page JavaScript
 * 클러스터 분석 페이지의 모든 인터랙티브 기능 관리
 */

// ===============================
// 클러스터 데이터 정의
// ===============================
const CLUSTER_DATA = {
    clusters: [
        {
            id: 0,
            name: "Survivor (보수적)",
            type: "survivor",
            count: 14527,
            percentage: 18.2,
            color: "#4CAF50",
            icon: "fas fa-shield-alt",
            description: "치료 아이템을 많이 사용하는 보수적인 생존형 플레이어",
            features: {
                heal_boost_ratio: 775.29,
                assists: 479.86,
                has_swimDistance: 294.02,
                walkDistance: 1.2,
                kills: 0.8
            },
            characteristics: [
                "치료 아이템 사용 빈도 매우 높음",
                "팀플레이 지향적 (어시스트 높음)",
                "안전한 플레이 스타일",
                "수영 거리 상대적으로 높음"
            ]
        },
        {
            id: 1,
            name: "Survivor (적극적)",
            type: "survivor",
            count: 24981,
            percentage: 31.2,
            color: "#8BC34A",
            icon: "fas fa-heart",
            description: "더 적극적이면서도 생존을 중시하는 플레이어",
            features: {
                heal_boost_ratio: 1861.49,
                assists: 964.62,
                damage_per_kill: 864.65,
                walkDistance: 1.5,
                kills: 1.2
            },
            characteristics: [
                "치료 아이템 사용 최고 수준",
                "킬당 데미지 효율성 높음",
                "어시스트 비율 최상위",
                "전체 플레이어의 1/3 차지"
            ]
        },
        {
            id: 2,
            name: "Explorer (Type 1)",
            type: "explorer",
            count: 10756,
            percentage: 13.4,
            color: "#FF9800",
            icon: "fas fa-route",
            description: "이동과 탐색을 중시하는 플레이어",
            features: {
                walkDistance_log: 3743.08,
                walkDistance: 1179.09,
                revives: 626.25,
                kills: 1.0,
                heal_boost_ratio: 2.0
            },
            characteristics: [
                "높은 이동거리",
                "맵 탐색 중시",
                "팀원 부활 적극적",
                "안정적인 플레이"
            ]
        },
        {
            id: 3,
            name: "Explorer (Type 2)",
            type: "explorer",
            count: 15898,
            percentage: 19.9,
            color: "#FF7043",
            icon: "fas fa-route",
            description: "전투와 탐색을 병행하는 플레이어",
            features: {
                walkDistance_log: 2245.80,
                longestKill: 610.84,
                has_kills: 501.94,
                kills: 1.8,
                heal_boost_ratio: 1.5
            },
            characteristics: [
                "중거리 이동 선호",
                "장거리 저격 능력",
                "킬 참여도 높음",
                "균형잡힌 플레이"
            ]
        },
        {
            id: 4,
            name: "Explorer (Type 3)",
            type: "explorer",
            count: 4312,
            percentage: 5.4,
            color: "#FB8C00",
            icon: "fas fa-route",
            description: "극한의 이동과 부활을 중시하는 플레이어",
            features: {
                walkDistance_log: 4451.52,
                walkDistance: 1845.04,
                revives: 1551.02,
                kills: 0.9,
                heal_boost_ratio: 3.0
            },
            characteristics: [
                "최대 이동거리",
                "팀원 부활 최우선",
                "안전한 포지셔닝",
                "서포터 역할"
            ]
        },
        {
            id: 5,
            name: "Explorer (Type 4)",
            type: "explorer",
            count: 4046,
            percentage: 5.1,
            color: "#FF8F00",
            icon: "fas fa-route",
            description: "무기 수집과 이동을 중시하는 플레이어",
            features: {
                walkDistance_log: 4139.77,
                walkDistance: 1544.91,
                weaponsAcquired: 451.79,
                kills: 1.1,
                heal_boost_ratio: 2.2
            },
            characteristics: [
                "높은 이동거리",
                "무기 수집 적극적",
                "장비 최적화",
                "준비성 높음"
            ]
        },
        {
            id: 6,
            name: "Explorer (Type 5)",
            type: "explorer",
            count: 5391,
            percentage: 6.7,
            color: "#FFA726",
            icon: "fas fa-route",
            description: "지구력과 이동을 중시하는 플레이어",
            features: {
                walkDistance_log: 3995.30,
                matchDuration: 1400.69,
                walkDistance: 1327.73,
                kills: 1.0,
                heal_boost_ratio: 2.1
            },
            characteristics: [
                "긴 게임 지속시간",
                "꾸준한 이동",
                "인내심 강함",
                "후반 생존력"
            ]
        },
        {
            id: 7,
            name: "Aggressive",
            type: "aggressive",
            count: 89,
            percentage: 0.1,
            color: "#F44336",
            icon: "fas fa-crosshairs",
            description: "극도로 공격적인 플레이를 하는 희귀한 플레이어",
            features: {
                kill_efficiency: 23396.88,
                damage_per_kill: 1435.73,
                assists: 920.03,
                kills: 8.5,
                heal_boost_ratio: 0.5
            },
            characteristics: [
                "킬 효율성 극상위 (23,000배 이상)",
                "공격적 플레이 스타일",
                "매우 희귀한 플레이어 유형",
                "높은 데미지 산출 능력"
            ]
        }
    ]
};

// ===============================
// DOM 요소 및 차트 변수
// ===============================
let clusterDistributionChart = null;
let comparisonChart = null;
let correlationChart = null;
let radarCharts = {};

// ===============================
// 페이지 초기화
// ===============================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 클러스터 분석 페이지 초기화 시작');
    
    try {
        initializeClusterAnalysis();
        console.log('✅ 클러스터 분석 페이지 초기화 완료');
    } catch (error) {
        console.error('❌ 초기화 오류:', error);
        showErrorMessage('페이지 초기화 중 오류가 발생했습니다.');
    }
});

/**
 * 클러스터 분석 페이지 초기화
 */
function initializeClusterAnalysis() {
    // 데이터 검증
    if (!validateClusterData()) {
        showErrorMessage('클러스터 데이터를 로딩할 수 없습니다.');
        return;
    }
    
    // 고급 기능 초기화
    if (typeof initializeAdvancedFeatures === 'function') {
        initializeAdvancedFeatures();
    }
    
    // 차트 초기화
    initializeCharts();
    
    // 이벤트 리스너 설정
    setupEventListeners();
    
    // 탭 기능 초기화
    initializeTabs();
    
    // 통계 테이블 생성
    generateStatisticsTable();
    
    // 애니메이션 시작
    startAnimations();
    
    console.log('✅ 클러스터 분석 페이지 초기화 완료 - 모든 기능 활성화');
}

/**
 * 모든 차트 초기화
 */
function initializeCharts() {
    try {
        createClusterDistributionChart();
        createRadarCharts();
        createComparisonChart();
        createCorrelationChart();
        console.log('✅ 모든 차트 초기화 완료');
    } catch (error) {
        console.error('❌ 차트 초기화 오류:', error);
    }
}

/**
 * 클러스터 분포 도넛 차트 생성
 */
function createClusterDistributionChart() {
    const ctx = document.getElementById('clusterDistributionChart');
    if (!ctx) return;

    const data = {
        labels: CLUSTER_DATA.clusters.map(cluster => cluster.name),
        datasets: [{
            data: CLUSTER_DATA.clusters.map(cluster => cluster.percentage),
            backgroundColor: CLUSTER_DATA.clusters.map(cluster => cluster.color),
            borderColor: '#ffffff',
            borderWidth: 2,
            hoverOffset: 10
        }]
    };

    clusterDistributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 12,
                            family: "'Noto Sans KR', sans-serif"
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const cluster = CLUSTER_DATA.clusters[context.dataIndex];
                            return `${cluster.name}: ${cluster.percentage}% (${cluster.count.toLocaleString()}명)`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 2000
            }
        }
    });
}

/**
 * 레이더 차트들 생성
 */
function createRadarCharts() {
    CLUSTER_DATA.clusters.forEach(cluster => {
        const canvasId = `radarChart${cluster.id}`;
        const ctx = document.getElementById(canvasId);
        
        if (ctx) {
            createSingleRadarChart(ctx, cluster);
        }
    });

    // Explorer 통합 레이더 차트
    const explorerCtx = document.getElementById('radarChartExplorer');
    if (explorerCtx) {
        createExplorerRadarChart(explorerCtx);
    }
}

/**
 * 개별 레이더 차트 생성
 */
function createSingleRadarChart(ctx, cluster) {
    const features = Object.keys(cluster.features);
    const values = Object.values(cluster.features).map(val => Math.log10(val + 1)); // 로그 스케일

    radarCharts[cluster.id] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: features.map(feature => feature.replace(/_/g, ' ')),
            datasets: [{
                label: cluster.name,
                data: values,
                backgroundColor: cluster.color + '30',
                borderColor: cluster.color,
                borderWidth: 2,
                pointBackgroundColor: cluster.color,
                pointBorderColor: '#fff',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 5,
                    ticks: {
                        stepSize: 1,
                        display: false
                    },
                    grid: {
                        color: '#e0e0e0'
                    },
                    pointLabels: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

/**
 * Explorer 통합 레이더 차트 생성
 */
function createExplorerRadarChart(ctx) {
    const explorerClusters = CLUSTER_DATA.clusters.filter(c => c.type === 'explorer');
    const features = ['walkDistance', 'longestKill', 'revives', 'weaponsAcquired', 'kills'];
    
    const datasets = explorerClusters.map(cluster => ({
        label: cluster.name,
        data: features.map(feature => {
            const value = cluster.features[feature] || cluster.features[feature + '_log'] || 1;
            return Math.log10(value + 1);
        }),
        backgroundColor: cluster.color + '20',
        borderColor: cluster.color,
        borderWidth: 1,
        pointRadius: 3
    }));

    radarCharts['explorer'] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: features.map(f => f.replace(/([A-Z])/g, ' $1').toLowerCase()),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 10
                        }
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 5,
                    ticks: {
                        stepSize: 1,
                        display: false
                    }
                }
            }
        }
    });
}

/**
 * 비교 바 차트 생성
 */
function createComparisonChart() {
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;

    const features = ['kills', 'walkDistance', 'heal_boost_ratio', 'assists'];
    const datasets = features.map((feature, index) => ({
        label: feature.replace(/_/g, ' '),
        data: CLUSTER_DATA.clusters.map(cluster => {
            const value = cluster.features[feature] || 1;
            return Math.log10(value + 1);
        }),
        backgroundColor: `hsl(${index * 60}, 70%, 60%)`,
        borderWidth: 1
    }));

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: CLUSTER_DATA.clusters.map(c => c.name.split(' ')[0]),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Log Scale Values'
                    }
                }
            }
        }
    });
}

/**
 * 상관관계 매트릭스 차트 생성
 */
function createCorrelationChart() {
    const ctx = document.getElementById('correlationChart');
    if (!ctx) return;

    // 간단한 상관관계 히트맵 시뮬레이션
    const clusterNames = CLUSTER_DATA.clusters.map(c => c.name.split(' ')[0]);
    const correlationData = [];

    for (let i = 0; i < clusterNames.length; i++) {
        for (let j = 0; j < clusterNames.length; j++) {
            const correlation = i === j ? 1 : Math.random() * 0.8;
            correlationData.push({
                x: j,
                y: i,
                v: correlation
            });
        }
    }

    correlationChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Correlation',
                data: correlationData,
                backgroundColor: function(context) {
                    const value = context.parsed.v;
                    const alpha = Math.abs(value);
                    return value > 0 ? `rgba(76, 175, 80, ${alpha})` : `rgba(244, 67, 54, ${alpha})`;
                },
                pointRadius: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: -0.5,
                    max: clusterNames.length - 0.5,
                    ticks: {
                        callback: function(value) {
                            return clusterNames[Math.round(value)] || '';
                        }
                    }
                },
                y: {
                    type: 'linear',
                    min: -0.5,
                    max: clusterNames.length - 0.5,
                    ticks: {
                        callback: function(value) {
                            return clusterNames[Math.round(value)] || '';
                        }
                    }
                }
            }
        }
    });
}
