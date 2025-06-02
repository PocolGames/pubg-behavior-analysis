/**
 * PUBG Player Prediction System
 * 플레이어 특성 기반 유형 예측 시스템
 */

class PlayerPredictor {
    constructor() {
        this.charts = new Map();
        this.currentPrediction = null;
        
        // 클러스터 정보 (실제 분석 결과 기반)
        this.clusterInfo = {
            0: {
                name: "Survivor (보수적)",
                description: "신중하고 생존 지향적인 플레이어",
                color: "#56ab2f",
                icon: "fas fa-shield-alt",
                percentage: 18.2,
                characteristics: ["높은 생존력", "신중한 플레이", "안전 우선"]
            },
            1: {
                name: "Survivor (적극적)",
                description: "생존력과 공격성을 균형있게 갖춘 플레이어",
                color: "#4CAF50",
                icon: "fas fa-user-shield",
                percentage: 31.2,
                characteristics: ["균형잡힌 플레이", "팀워크 중시", "전략적 사고"]
            },
            2: {
                name: "Explorer (활발한)",
                description: "맵을 적극적으로 탐험하는 플레이어",
                color: "#667eea",
                icon: "fas fa-map-marked-alt",
                percentage: 13.4,
                characteristics: ["높은 이동성", "맵 탐험", "위험 감수"]
            },
            3: {
                name: "Explorer (균형형)",
                description: "탐험과 전투를 균형있게 하는 플레이어",
                color: "#5A67D8",
                icon: "fas fa-compass",
                percentage: 19.9,
                characteristics: ["균형적 이동", "상황 판단", "유연한 전략"]
            },
            4: {
                name: "Explorer (극한)",
                description: "극도로 높은 이동성을 보이는 플레이어",
                color: "#4299E1",
                icon: "fas fa-rocket",
                percentage: 5.4,
                characteristics: ["극한 이동", "고위험 탐험", "독특한 플레이"]
            },
            5: {
                name: "Explorer (전술적)",
                description: "전술적 이동과 무기 수집에 능한 플레이어",
                color: "#3182CE",
                icon: "fas fa-chess",
                percentage: 5.1,
                characteristics: ["전략적 이동", "무기 수집", "계획적 플레이"]
            },
            6: {
                name: "Explorer (지구력)",
                description: "높은 지구력과 지속성을 보이는 플레이어",
                color: "#2B6CB0",
                icon: "fas fa-mountain",
                percentage: 6.7,
                characteristics: ["높은 지구력", "장시간 플레이", "끈기있는 탐험"]
            },
            7: {
                name: "Aggressive (공격형)",
                description: "극도로 공격적이고 위험을 감수하는 플레이어",
                color: "#dc3545",
                icon: "fas fa-fire",
                percentage: 0.1,
                characteristics: ["극도의 공격성", "고위험 고수익", "빠른 결정"]
            }
        };

        // 샘플 플레이어 데이터
        this.samplePlayers = {
            'survivor-conservative': {
                kills: 1, damageDealt: 150, longestKill: 50, headshotKills: 0, assists: 0, weaponsAcquired: 2,
                walkDistance: 800, rideDistance: 0, swimDistance: 0,
                heals: 3, boosts: 1, revives: 0, DBNOs: 1,
                killPlace: 70, matchDuration: 2000, maxPlace: 100, numGroups: 50
            },
            'survivor-active': {
                kills: 3, damageDealt: 400, longestKill: 120, headshotKills: 1, assists: 1, weaponsAcquired: 4,
                walkDistance: 1500, rideDistance: 200, swimDistance: 0,
                heals: 2, boosts: 3, revives: 1, DBNOs: 3,
                killPlace: 30, matchDuration: 1800, maxPlace: 100, numGroups: 50
            },
            'explorer': {
                kills: 2, damageDealt: 250, longestKill: 80, headshotKills: 0, assists: 0, weaponsAcquired: 5,
                walkDistance: 3000, rideDistance: 1500, swimDistance: 50,
                heals: 1, boosts: 2, revives: 0, DBNOs: 2,
                killPlace: 45, matchDuration: 1900, maxPlace: 100, numGroups: 50
            },
            'aggressive': {
                kills: 8, damageDealt: 800, longestKill: 200, headshotKills: 3, assists: 2, weaponsAcquired: 6,
                walkDistance: 1200, rideDistance: 300, swimDistance: 0,
                heals: 1, boosts: 1, revives: 0, DBNOs: 8,
                killPlace: 5, matchDuration: 1200, maxPlace: 100, numGroups: 50
            }
        };

        this.initializeEventListeners();
        this.initializeCharts();
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
                labels: Object.values(this.clusterInfo).map(info => info.name),
                datasets: [{
                    label: '예측 확률',
                    data: Array(8).fill(12.5), // 균등 분포
                    backgroundColor: Object.values(this.clusterInfo).map(info => info.color + '80'),
                    borderColor: Object.values(this.clusterInfo).map(info => info.color),
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
     * 플레이어 유형 예측 (메인 함수)
     */
    async predictPlayerType() {
        const formData = this.getFormData();
        
        // 입력 검증
        if (!this.validateAllInputs(formData)) {
            App.showNotification('입력값을 확인해주세요.', 'error');
            return;
        }

        // 예측 시작 상태
        this.updatePredictionStatus('분석중', 'processing');
        
        const startTime = performance.now();

        try {
            // 예측 알고리즘 실행
            const prediction = await this.runPredictionAlgorithm(formData);
            
            const endTime = performance.now();
            const predictionTime = ((endTime - startTime) / 1000).toFixed(2);

            // 결과 업데이트
            this.currentPrediction = prediction;
            this.updatePredictionResults(prediction, predictionTime);
            this.updateCharts(formData, prediction);
            
            // 완료 상태
            this.updatePredictionStatus('분석 완료', 'success');
            
            App.showNotification('플레이어 유형 예측이 완료되었습니다!', 'success');

        } catch (error) {
            console.error('예측 중 오류 발생:', error);
            this.updatePredictionStatus('오류 발생', 'error');
            App.showNotification('예측 중 오류가 발생했습니다.', 'error');
        }
    }

    /**
     * 예측 알고리즘 실행 (시뮬레이션)
     */
    async runPredictionAlgorithm(formData) {
        // 시뮬레이션 지연
        await new Promise(resolve => setTimeout(resolve, 800));

        // 특성 정규화 및 가중치 계산
        const features = this.normalizeFeatures(formData);
        const weights = this.calculateFeatureWeights();
        
        // 각 클러스터에 대한 유사도 계산
        const similarities = this.calculateClusterSimilarities(features, weights);
        
        // 소프트맥스로 확률 변환
        const probabilities = this.softmax(similarities);
        
        // 가장 높은 확률의 클러스터 선택
        const predictedCluster = probabilities.indexOf(Math.max(...probabilities));
        const confidence = probabilities[predictedCluster] * 100;

        return {
            predictedCluster,
            confidence,
            probabilities,
            features,
            analysis: this.generateAnalysis(features, predictedCluster)
        };
    }

    /**
     * 특성 정규화
     */
    normalizeFeatures(formData) {
        const normalized = {};
        
        // 각 특성별 정규화 (0-100 스케일)
        normalized.combat = Math.min(100, (
            formData.kills * 10 + 
            formData.damageDealt * 0.05 + 
            formData.headshotKills * 15 + 
            formData.assists * 8
        ) / 4);
        
        normalized.movement = Math.min(100, (
            formData.walkDistance * 0.01 + 
            formData.rideDistance * 0.005 + 
            formData.swimDistance * 0.1
        ) / 3);
        
        normalized.survival = Math.min(100, (
            formData.heals * 8 + 
            formData.boosts * 10 + 
            formData.revives * 12 +
            (100 - formData.killPlace)
        ) / 4);
        
        normalized.teamplay = Math.min(100, (
            formData.assists * 15 + 
            formData.revives * 20
        ) / 2);
        
        normalized.weapons = Math.min(100, formData.weaponsAcquired * 10);
        normalized.longevity = Math.min(100, formData.matchDuration * 0.05);
        normalized.efficiency = Math.min(100, formData.longestKill * 0.1);
        normalized.aggression = Math.min(100, formData.DBNOs * 8);

        return normalized;
    }

    /**
     * 특성 가중치 계산 (실제 모델 중요도 기반)
     */
    calculateFeatureWeights() {
        return {
            combat: 0.32,      // has_kills 기반
            movement: 0.22,    // walkDistance 기반
            survival: 0.15,    // 생존 관련
            teamplay: 0.10,    // assists 기반
            weapons: 0.08,     // weaponsAcquired 기반
            longevity: 0.05,   // 지속성
            efficiency: 0.04,  // 효율성
            aggression: 0.04   // 공격성
        };
    }

    /**
     * 클러스터 유사도 계산
     */
    calculateClusterSimilarities(features, weights) {
        // 각 클러스터의 대표 특성 (실제 데이터 기반)
        const clusterCentroids = {
            0: { combat: 20, movement: 30, survival: 80, teamplay: 25, weapons: 40, longevity: 85, efficiency: 25, aggression: 15 },
            1: { combat: 45, movement: 50, survival: 70, teamplay: 65, weapons: 60, longevity: 75, efficiency: 40, aggression: 35 },
            2: { combat: 35, movement: 85, survival: 45, teamplay: 40, weapons: 65, longevity: 65, efficiency: 60, aggression: 30 },
            3: { combat: 40, movement: 70, survival: 55, teamplay: 50, weapons: 70, longevity: 70, efficiency: 55, aggression: 40 },
            4: { combat: 30, movement: 95, survival: 35, teamplay: 30, weapons: 75, longevity: 55, efficiency: 70, aggression: 25 },
            5: { combat: 35, movement: 80, survival: 40, teamplay: 35, weapons: 85, longevity: 60, efficiency: 65, aggression: 30 },
            6: { combat: 25, movement: 75, survival: 60, teamplay: 45, weapons: 70, longevity: 90, efficiency: 50, aggression: 20 },
            7: { combat: 95, movement: 45, survival: 15, teamplay: 20, weapons: 80, longevity: 35, efficiency: 85, aggression: 95 }
        };

        const similarities = [];
        
        for (let i = 0; i < 8; i++) {
            let similarity = 0;
            const centroid = clusterCentroids[i];
            
            for (const [feature, weight] of Object.entries(weights)) {
                const distance = Math.abs(features[feature] - centroid[feature]);
                similarity += (100 - distance) * weight;
            }
            
            similarities.push(similarity);
        }
        
        return similarities;
    }

    /**
     * 소프트맥스 함수
     */
    softmax(similarities) {
        const max = Math.max(...similarities);
        const exponentials = similarities.map(x => Math.exp(x - max));
        const sum = exponentials.reduce((a, b) => a + b, 0);
        return exponentials.map(x => x / sum);
    }

    /**
     * 분석 결과 생성
     */
    generateAnalysis(features, predictedCluster) {
        const sortedFeatures = Object.entries(features)
            .sort(([,a], [,b]) => b - a)
            .map(([key]) => key);

        const featureNames = {
            combat: '전투력',
            movement: '이동성',
            survival: '생존력',
            teamplay: '팀플레이',
            weapons: '무기 활용',
            longevity: '지속성',
            efficiency: '효율성',
            aggression: '공격성'
        };

        const playStyles = {
            0: '매우 보수적', 1: '균형잡힌', 2: '탐험적', 3: '전략적',
            4: '모험적', 5: '계획적', 6: '인내심 있는', 7: '극도로 공격적'
        };

        return {
            strongestFeature: featureNames[sortedFeatures[0]],
            playStyle: playStyles[predictedCluster],
            similarPlayers: this.clusterInfo[predictedCluster].percentage.toFixed(1) + '%'
        };
    }

    /**
     * 예측 결과 업데이트
     */
    updatePredictionResults(prediction, predictionTime) {
        const clusterInfo = this.clusterInfo[prediction.predictedCluster];
        
        // 예측 요약 업데이트
        const summary = document.getElementById('predictionSummary');
        const typeName = document.getElementById('predictedTypeName');
        const typeDescription = document.getElementById('predictedTypeDescription');
        const confidenceScore = document.getElementById('confidenceScore');
        
        if (summary && typeName && typeDescription && confidenceScore) {
            summary.style.display = 'block';
            typeName.textContent = clusterInfo.name;
            typeDescription.textContent = clusterInfo.description;
            confidenceScore.textContent = prediction.confidence.toFixed(1) + '%';
            
            // 아이콘 색상 변경
            const typeIcon = summary.querySelector('.type-icon i');
            if (typeIcon) {
                typeIcon.style.color = clusterInfo.color;
                typeIcon.className = clusterInfo.icon;
            }
        }

        // 분석 세부사항 업데이트
        const details = document.getElementById('analysisDetails');
        if (details) {
            details.style.display = 'block';
            
            const strongestFeature = document.getElementById('strongestFeature');
            const playStyle = document.getElementById('playStyle');
            const similarPlayers = document.getElementById('similarPlayers');
            const predictionTimeEl = document.getElementById('predictionTime');
            
            if (strongestFeature) strongestFeature.textContent = prediction.analysis.strongestFeature;
            if (playStyle) playStyle.textContent = prediction.analysis.playStyle;
            if (similarPlayers) similarPlayers.textContent = prediction.analysis.similarPlayers;
            if (predictionTimeEl) predictionTimeEl.textContent = predictionTime + '초';
        }
    }

    /**
     * 차트 업데이트
     */
    updateCharts(formData, prediction) {
        this.updateProbabilityChart(prediction.probabilities);
        this.updateRadarChart(formData, prediction.features);
    }

    /**
     * 확률 분포 차트 업데이트
     */
    updateProbabilityChart(probabilities) {
        const chart = this.charts.get('probability');
        if (chart) {
            const percentages = probabilities.map(p => (p * 100).toFixed(1));
            chart.data.datasets[0].data = percentages;
            chart.update('active');
        }
    }

    /**
     * 레이더 차트 업데이트
     */
    updateRadarChart(formData, features) {
        const chart = this.charts.get('radar');
        if (chart) {
            const radarData = [
                Math.min(100, formData.kills * 10),
                Math.min(100, formData.damageDealt * 0.1),
                Math.min(100, formData.walkDistance * 0.01),
                features.survival,
                Math.min(100, formData.assists * 15),
                Math.min(100, formData.weaponsAcquired * 12),
                Math.min(100, formData.boosts * 15),
                Math.min(100, formData.heals * 12)
            ];
            
            chart.data.datasets[0].data = radarData;
            chart.update('active');
        }
    }

    /**
     * 예측 상태 업데이트
     */
    updatePredictionStatus(status, type) {
        const statusEl = document.getElementById('predictionStatus');
        if (statusEl) {
            const indicator = statusEl.querySelector('.status-indicator');
            if (indicator) {
                indicator.textContent = status;
                indicator.className = `status-indicator ${type}`;
            }
        }
    }

    /**
     * 폼 데이터 가져오기
     */
    getFormData() {
        const form = document.getElementById('playerForm');
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value) || 0;
        }
        
        return data;
    }

    /**
     * 모든 입력 검증
     */
    validateAllInputs(formData) {
        const inputs = document.querySelectorAll('#playerForm input[type="number"]');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateInput(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }

    /**
     * 개별 입력 검증
     */
    validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        let isValid = true;
        let message = '';
        
        if (isNaN(value)) {
            isValid = false;
            message = '숫자를 입력해주세요.';
        } else if (value < min) {
            isValid = false;
            message = `최소값: ${min}`;
        } else if (value > max) {
            isValid = false;
            message = `최대값: ${max}`;
        }
        
        // 시각적 피드백
        if (isValid) {
            input.classList.remove('error');
            this.removeErrorMessage(input);
        } else {
            input.classList.add('error');
            this.showErrorMessage(input, message);
        }
        
        return isValid;
    }

    /**
     * 에러 메시지 표시
     */
    showErrorMessage(input, message) {
        this.removeErrorMessage(input);
        
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message;
        
        input.parentNode.appendChild(errorEl);
    }

    /**
     * 에러 메시지 제거
     */
    removeErrorMessage(input) {
        const existingError = input.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
    }

    /**
     * 폼 초기화
     */
    resetForm() {
        const form = document.getElementById('playerForm');
        if (form) {
            form.reset();
            
            // 기본값 설정
            const defaults = {
                kills: 0, damageDealt: 0, longestKill: 0, headshotKills: 0, assists: 0, weaponsAcquired: 3,
                walkDistance: 1000, rideDistance: 0, swimDistance: 0,
                heals: 1, boosts: 1, revives: 0, DBNOs: 0,
                killPlace: 50, matchDuration: 1800, maxPlace: 100, numGroups: 50
            };
            
            for (const [key, value] of Object.entries(defaults)) {
                const input = document.getElementById(key);
                if (input) input.value = value;
            }
        }
        
        // 결과 패널 숨기기
        this.hidePredictionResults();
        
        // 차트 초기화
        this.createPlaceholderCharts();
        
        App.showNotification('폼이 초기화되었습니다.', 'info');
    }

    /**
     * 샘플 플레이어 로드
     */
    loadSamplePlayer(playerType) {
        const sampleData = this.samplePlayers[playerType];
        if (!sampleData) return;
        
        // 폼에 데이터 입력
        for (const [key, value] of Object.entries(sampleData)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;
            }
        }
        
        App.showNotification(`${playerType} 샘플 데이터가 로드되었습니다.`, 'success');
        
        // 자동 예측 실행 (선택사항)
        setTimeout(() => {
            this.predictPlayerType();
        }, 500);
    }

    /**
     * 샘플 메뉴 표시
     */
    showSampleMenu() {
        // 샘플 플레이어 섹션으로 스크롤
        const sampleSection = document.querySelector('.sample-players');
        if (sampleSection) {
            sampleSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // 하이라이트 효과
            sampleSection.style.animation = 'pulse 1s ease-in-out';
            setTimeout(() => {
                sampleSection.style.animation = '';
            }, 1000);
        }
    }

    /**
     * 예측 결과 숨기기
     */
    hidePredictionResults() {
        const summary = document.getElementById('predictionSummary');
        const details = document.getElementById('analysisDetails');
        
        if (summary) summary.style.display = 'none';
        if (details) details.style.display = 'none';
        
        this.updatePredictionStatus('대기중', '');
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

// 전역 변수로 인스턴스 생성
let playerPredictor;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    // Player Prediction 페이지에서만 실행
    if (window.location.pathname.includes('player-prediction.html')) {
        playerPredictor = new PlayerPredictor();
        
        // 페이지 언로드 시 정리
        window.addEventListener('beforeunload', () => {
            if (playerPredictor) {
                playerPredictor.destroy();
            }
        });
    }
});

// 전역 접근을 위한 export
if (typeof window !== 'undefined') {
    window.PlayerPredictor = PlayerPredictor;
}