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
            this.updateFormLabels();
            
            console.log('✅ PlayerPredictor 초기화 완료');
            this.showLoadingComplete();
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
            this.showLoading('플레이어 예측 데이터 로딩 중...');
            
            const response = await fetch('./data/player-prediction.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.predictionData = await response.json();
            this.isDataLoaded = true;
            
            console.log('✅ 플레이어 예측 데이터 로드 완료:', this.predictionData.metadata);
            
            // 메타데이터 업데이트
            this.updateMetadataDisplay();
            
        } catch (error) {
            console.error('❌ JSON 데이터 로드 실패:', error);
            this.showError('데이터 로딩에 실패했습니다. 백업 데이터를 사용합니다.');
            throw error;
        }
    }

    /**
     * 메타데이터 표시 업데이트
     */
    updateMetadataDisplay() {
        if (!this.isDataLoaded) return;

        const metadata = this.predictionData.metadata;
        
        // 모델 정보 카드 업데이트
        const accuracyElement = document.querySelector('.model-accuracy');
        if (accuracyElement) {
            accuracyElement.textContent = `${(metadata.accuracy * 100).toFixed(2)}%`;
        }

        const featuresElement = document.querySelector('.model-features');
        if (featuresElement) {
            featuresElement.textContent = `${metadata.features}개`;
        }

        const classesElement = document.querySelector('.model-classes');
        if (classesElement) {
            classesElement.textContent = `${metadata.classes}개`;
        }

        const modelNameElement = document.querySelector('.model-name');
        if (modelNameElement) {
            modelNameElement.textContent = metadata.modelName;
        }
    }

    /**
     * 폼 라벨 업데이트 (JSON 데이터 기반)
     */
    updateFormLabels() {
        if (!this.isDataLoaded) return;

        const features = this.predictionData.featureDefinitions;
        
        features.forEach(feature => {
            const input = document.querySelector(`input[name="${feature.name}"]`);
            if (input) {
                const label = input.closest('.form-group')?.querySelector('label');
                if (label) {
                    label.innerHTML = `
                        ${feature.displayName} 
                        <span class="feature-info">(${feature.min}-${feature.max}${feature.unit})</span>
                    `;
                }
                
                // 입력 제한 설정
                input.min = feature.min;
                input.max = feature.max;
                input.title = feature.description;
            }
        });
    }

    /**
     * 백업 데이터로 초기화
     */
    initializeWithFallbackData() {
        console.warn('⚠️ 백업 데이터로 초기화');
        this.isDataLoaded = false;
        this.initializeEventListeners();
        this.initializeCharts();
        this.showError('JSON 데이터를 로드할 수 없어 기본 데이터를 사용합니다.');
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
                    features: cluster.features,
                    characteristics: ["실제 PUBG 데이터 기반", "머신러닝 분석 완료"]
                };
            });
            
            return clusterInfo;
        }
        
        return this.fallbackClusterInfo;
    }

    /**
     * 특성 정의 가져오기
     */
    getFeatureDefinitions() {
        if (this.isDataLoaded && this.predictionData.featureDefinitions) {
            return this.predictionData.featureDefinitions;
        }
        
        // 백업 특성 정의
        return [
            { name: 'walkDistance', displayName: '보행 거리', min: 0, max: 10000, unit: 'm', weight: 0.075 },
            { name: 'killPlace', displayName: '킬 순위', min: 1, max: 100, unit: '', weight: 0.057 },
            { name: 'boosts', displayName: '부스트 아이템', min: 0, max: 20, unit: '개', weight: 0.040 },
            { name: 'weaponsAcquired', displayName: '무기 획득', min: 0, max: 20, unit: '개', weight: 0.059 },
            { name: 'damageDealt', displayName: '총 데미지', min: 0, max: 2000, unit: '', weight: 0.052 },
            { name: 'kills', displayName: '킬 수', min: 0, max: 50, unit: '명', weight: 0.045 },
            { name: 'heals', displayName: '치료 아이템', min: 0, max: 30, unit: '개', weight: 0.038 },
            { name: 'longestKill', displayName: '최장 킬 거리', min: 0, max: 1000, unit: 'm', weight: 0.035 },
            { name: 'rideDistance', displayName: '탑승 거리', min: 0, max: 20000, unit: 'm', weight: 0.051 },
            { name: 'assists', displayName: '어시스트', min: 0, max: 20, unit: '회', weight: 0.032 }
        ];
    }

    /**
     * 샘플 플레이어 데이터 가져오기
     */
    getSamplePlayers() {
        if (this.isDataLoaded && this.predictionData.samplePlayers) {
            return this.predictionData.samplePlayers;
        }
        
        // 백업 샘플 데이터
        return {
            conservative: { name: "보수적 생존형", data: [150, 60, 1, 2, 50, 0, 2, 0, 0, 1] },
            active: { name: "적극적 생존형", data: [600, 40, 2, 3, 120, 2, 3, 15, 300, 2] },
            explorer: { name: "탐험형", data: [2000, 25, 4, 5, 180, 3, 5, 25, 1000, 3] },
            aggressive: { name: "공격형", data: [1500, 10, 6, 8, 400, 8, 8, 50, 800, 5] }
        };
    }

    /**
     * 로딩 상태 표시
     */
    showLoading(message = '로딩 중...') {
        const loadingElement = document.querySelector('.loading-indicator');
        if (loadingElement) {
            loadingElement.textContent = message;
            loadingElement.style.display = 'block';
        }
    }

    /**
     * 로딩 완료 표시
     */
    showLoadingComplete() {
        const loadingElement = document.querySelector('.loading-indicator');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
    }

    /**
     * 에러 메시지 표시
     */
    showError(message) {
        if (window.App && window.App.showNotification) {
            window.App.showNotification(message, 'warning');
        } else {
            console.warn('⚠️', message);
        }
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
        const clusterInfo = this.getClusterInfo();
        
        // 확률 분포 차트
        const probabilityCanvas = document.getElementById('probabilityChart');
        if (probabilityCanvas && window.ChartUtils) {
            const placeholderData = {
                labels: Object.values(clusterInfo).map(info => info.name),
                datasets: [{
                    label: '예측 확률',
                    data: Array(8).fill(12.5), // 균등 분포
                    backgroundColor: Object.values(clusterInfo).map(info => info.color + '80'),
                    borderColor: Object.values(clusterInfo).map(info => info.color),
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
