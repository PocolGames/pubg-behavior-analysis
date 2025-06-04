/**
 * PUBG Player Prediction System - Core Module
 * 플레이어 예측 시스템 핵심 모듈
 * 클래스 정의, 초기화, 데이터 로딩 담당
 */

class PlayerPredictor {
    constructor() {
        this.charts = new Map();
        this.currentPrediction = null;
        this.predictionData = null;
        this.isDataLoaded = false;
        
        // 백업 클러스터 정보 (JSON 로딩 실패 시 사용)
        this.fallbackClusterInfo = {
            0: { name: "Survivor (보수적)", description: "신중하고 생존 지향적인 플레이어", color: "#56ab2f", icon: "fas fa-shield-alt", percentage: 18.2 },
            1: { name: "Survivor (적극적)", description: "생존력과 공격성을 균형있게 갖춘 플레이어", color: "#4CAF50", icon: "fas fa-user-shield", percentage: 31.2 },
            2: { name: "Explorer (활발한)", description: "맵을 적극적으로 탐험하는 플레이어", color: "#667eea", icon: "fas fa-map-marked-alt", percentage: 13.4 },
            3: { name: "Explorer (균형형)", description: "탐험과 전투를 균형있게 하는 플레이어", color: "#5A67D8", icon: "fas fa-compass", percentage: 19.9 },
            4: { name: "Explorer (극한)", description: "극도로 높은 이동성을 보이는 플레이어", color: "#4299E1", icon: "fas fa-rocket", percentage: 5.4 },
            5: { name: "Explorer (전술적)", description: "전술적 이동과 무기 수집에 능한 플레이어", color: "#3182CE", icon: "fas fa-chess", percentage: 5.1 },
            6: { name: "Explorer (지구력)", description: "높은 지구력과 지속성을 보이는 플레이어", color: "#2B6CB0", icon: "fas fa-mountain", percentage: 6.7 },
            7: { name: "Aggressive (공격형)", description: "극도로 공격적이고 위험을 감수하는 플레이어", color: "#dc3545", icon: "fas fa-fire", percentage: 0.1 }
        };

        // 초기화
        this.initializeWithData();
    }

    /**
     * 데이터와 함께 초기화
     */
    async initializeWithData() {
        try {
            // JSON 데이터 로드
            await this.loadPredictionData();
            
            // 이벤트 리스너 및 차트 초기화
            this.initializeEventListeners();
            this.initializeCharts();
            
            console.log('✅ PlayerPredictor 초기화 완료');
        } catch (error) {
            console.error('❌ PlayerPredictor 초기화 중 오류:', error);
            // 백업 데이터로 초기화
            this.initializeWithFallbackData();
        }
    }

    /**
     * JSON 데이터 로드
     */
    async loadPredictionData() {
        try {
            const response = await fetch('./data/player-prediction.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.predictionData = await response.json();
            this.isDataLoaded = true;
            
            console.log('✅ 플레이어 예측 데이터 로드 완료:', this.predictionData.metadata);
        } catch (error) {
            console.error('❌ JSON 데이터 로드 실패:', error);
            throw error;
        }
    }

    /**
     * 백업 데이터로 초기화
     */
    initializeWithFallbackData() {
        console.warn('⚠️ 백업 데이터로 초기화');
        this.isDataLoaded = false;
        this.initializeEventListeners();
        this.initializeCharts();
    }

    /**
     * 클러스터 정보 가져오기
     */
    getClusterInfo() {
        if (this.isDataLoaded && this.predictionData.clusterCenters) {
            const clusterInfo = {};
            const colors = ["#56ab2f", "#4CAF50", "#667eea", "#5A67D8", "#4299E1", "#3182CE", "#2B6CB0", "#dc3545"];
            const icons = ["fas fa-shield-alt", "fas fa-user-shield", "fas fa-map-marked-alt", "fas fa-compass", 
                          "fas fa-rocket", "fas fa-chess", "fas fa-mountain", "fas fa-fire"];
            const percentages = [18.2, 31.2, 13.4, 19.9, 5.4, 5.1, 6.7, 0.1];
            
            Object.keys(this.predictionData.clusterCenters).forEach((key, index) => {
                const cluster = this.predictionData.clusterCenters[key];
                clusterInfo[key] = {
                    name: cluster.name,
                    description: `${cluster.name} 특성을 가진 플레이어`,
                    color: colors[index] || "#666",
                    icon: icons[index] || "fas fa-user",
                    percentage: percentages[index] || 0,
                    characteristics: ["실제 데이터 기반", "특성 분석 완료"]
                };
            });
            
            return clusterInfo;
        }
        
        return this.fallbackClusterInfo;
    }

    /**
     * 특성 이름 목록 가져오기
     */
    getFeatureNames() {
        if (this.isDataLoaded && this.predictionData.featureDefinitions) {
            return this.predictionData.featureDefinitions.map(feature => feature.name);
        }
        
        return ['walkDistance', 'killPlace', 'boosts', 'weaponsAcquired', 'damageDealt', 
                'kills', 'heals', 'longestKill', 'rideDistance', 'assists'];
    }

    /**
     * 이벤트 리스너 초기화
     */
    initializeEventListeners() {
        // 폼 제출
        const form = document.getElementById('playerForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.predictPlayerType();
            });
        }

        // 초기화 버튼
        const resetBtn = document.getElementById('resetForm');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetForm());
        }

        // 샘플 로드 버튼
        const loadSampleBtn = document.getElementById('loadSample');
        if (loadSampleBtn) {
            loadSampleBtn.addEventListener('click', () => this.showSampleMenu());
        }

        // 샘플 플레이어 버튼들
        const sampleButtons = document.querySelectorAll('.sample-player button');
        sampleButtons.forEach((btn, index) => {
            btn.addEventListener('click', (e) => {
                const playerType = e.target.closest('.sample-player').dataset.type;
                this.loadSamplePlayer(playerType);
            });
        });

        // 실시간 입력 검증
        const inputs = document.querySelectorAll('#playerForm input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('input', () => this.validateInput(input));
            input.addEventListener('blur', () => this.validateInput(input));
        });

        // 윈도우 리사이즈
        window.addEventListener('resize', () => {
            this.resizeCharts();
        });
    }

    /**
     * 차트 초기화
     */
    initializeCharts() {
        this.createPlaceholderCharts();
    }

    /**
     * 플레이스홀더 차트 생성
     */
    createPlaceholderCharts() {
        // 확률 분포 차트
        const probabilityCanvas = document.getElementById('probabilityChart');
        if (probabilityCanvas && window.ChartUtils) {
            const placeholderData = {
                labels: Object.values(this.getClusterInfo()).map(info => info.name),
                datasets: [{
                    label: '예측 확률',
                    data: Array(8).fill(12.5), // 균등 분포
                    backgroundColor: Object.values(this.getClusterInfo()).map(info => info.color + '80'),
                    borderColor: Object.values(this.getClusterInfo()).map(info => info.color),
                    borderWidth: 1
                }]
            };

            this.charts.set('probability', window.ChartUtils.createBarChart(
                probabilityCanvas,
                placeholderData,
                {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        title: {
                            display: false
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            ));
        }

        // 플레이어 레이더 차트
        const radarCanvas = document.getElementById('playerRadarChart');
        if (radarCanvas && window.ChartUtils) {
            const placeholderRadarData = {
                labels: ['킬 수', '데미지', '이동거리', '생존력', '어시스트', '무기획득', '부스트', '치료'],
                datasets: [{
                    label: '플레이어 특성',
                    data: Array(8).fill(50), // 중간값
                    backgroundColor: '#667eea20',
                    borderColor: '#667eea',
                    borderWidth: 2,
                    pointBackgroundColor: '#667eea'
                }]
            };

            this.charts.set('radar', window.ChartUtils.createRadarChart(
                radarCanvas,
                placeholderRadarData,
                {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                stepSize: 20
                            }
                        }
                    }
                }
            ));
        }
    }

    /**
     * 차트 리사이즈
     */
    resizeCharts() {
        this.charts.forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }

    /**
     * 메모리 정리
     */
    destroy() {
        this.charts.forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts.clear();
    }
}

// 전역 접근을 위한 export
if (typeof window !== 'undefined') {
    window.PlayerPredictor = PlayerPredictor;
}
