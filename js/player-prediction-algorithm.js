/**
 * PUBG Player Prediction System - Algorithm Module
 * 플레이어 예측 시스템 알고리즘 모듈
 * 예측 로직, 특성 정규화, 클러스터 유사도 계산 담당
 */

/**
 * PlayerPredictor 클래스에 알고리즘 관련 메서드 추가
 */
Object.assign(PlayerPredictor.prototype, {
    /**
     * 플레이어 유형 예측 (메인 함수)
     */
    async predictPlayerType() {
        const formData = this.getFormData();
        
        // 입력 검증
        if (!this.validateAllInputs(formData)) {
            if (window.App) {
                App.showNotification('입력값을 확인해주세요.', 'error');
            }
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
            
            if (window.App) {
                App.showNotification('플레이어 유형 예측이 완료되었습니다!', 'success');
            }

        } catch (error) {
            console.error('예측 중 오류 발생:', error);
            this.updatePredictionStatus('오류 발생', 'error');
            if (window.App) {
                App.showNotification('예측 중 오류가 발생했습니다.', 'error');
            }
        }
    },

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
    },

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
    },

    /**
     * 특성 가중치 계산 (실제 모델 중요도 기반)
     */
    calculateFeatureWeights() {
        if (this.isDataLoaded && this.predictionData.featureWeights) {
            return this.predictionData.featureWeights;
        }
        
        // 백업 가중치 (실제 PUBG 분석 결과 기반)
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
    },

    /**
     * 클러스터 유사도 계산
     */
    calculateClusterSimilarities(features, weights) {
        // 각 클러스터의 대표 특성 (실제 데이터 기반)
        const clusterCentroids = this.getClusterCentroids();
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
    },

    /**
     * 클러스터 중심점 가져오기
     */
    getClusterCentroids() {
        if (this.isDataLoaded && this.predictionData.clusterCentroids) {
            return this.predictionData.clusterCentroids;
        }
        
        // 백업 클러스터 중심점 (실제 PUBG 분석 결과 기반)
        return {
            0: { combat: 20, movement: 30, survival: 80, teamplay: 25, weapons: 40, longevity: 85, efficiency: 25, aggression: 15 },
            1: { combat: 45, movement: 50, survival: 70, teamplay: 65, weapons: 60, longevity: 75, efficiency: 40, aggression: 35 },
            2: { combat: 35, movement: 85, survival: 45, teamplay: 40, weapons: 65, longevity: 65, efficiency: 60, aggression: 30 },
            3: { combat: 40, movement: 70, survival: 55, teamplay: 50, weapons: 70, longevity: 70, efficiency: 55, aggression: 40 },
            4: { combat: 30, movement: 95, survival: 35, teamplay: 30, weapons: 75, longevity: 55, efficiency: 70, aggression: 25 },
            5: { combat: 35, movement: 80, survival: 40, teamplay: 35, weapons: 85, longevity: 60, efficiency: 65, aggression: 30 },
            6: { combat: 25, movement: 75, survival: 60, teamplay: 45, weapons: 70, longevity: 90, efficiency: 50, aggression: 20 },
            7: { combat: 95, movement: 45, survival: 15, teamplay: 20, weapons: 80, longevity: 35, efficiency: 85, aggression: 95 }
        };
    },

    /**
     * 소프트맥스 함수
     */
    softmax(similarities) {
        const max = Math.max(...similarities);
        const exponentials = similarities.map(x => Math.exp(x - max));
        const sum = exponentials.reduce((a, b) => a + b, 0);
        return exponentials.map(x => x / sum);
    },

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

        const clusterInfo = this.getClusterInfo();

        return {
            strongestFeature: featureNames[sortedFeatures[0]],
            playStyle: playStyles[predictedCluster],
            similarPlayers: clusterInfo[predictedCluster].percentage.toFixed(1) + '%'
        };
    },

    /**
     * 폼 데이터 가져오기
     */
    getFormData() {
        const form = document.getElementById('playerForm');
        if (!form) return {};
        
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value) || 0;
        }
        
        return data;
    },

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
    },

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
    },

    /**
     * 에러 메시지 표시
     */
    showErrorMessage(input, message) {
        this.removeErrorMessage(input);
        
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message;
        
        input.parentNode.appendChild(errorEl);
    },

    /**
     * 에러 메시지 제거
     */
    removeErrorMessage(input) {
        const existingError = input.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
    }
});
