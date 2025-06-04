/**
 * PUBG Player Prediction System - Algorithm Module
 * 플레이어 예측 시스템 알고리즘 모듈
 * 예측 로직, 특성 정규화, 클러스터 유사도 계산 담당
 */

// PlayerPredictor 클래스가 로드될 때까지 대기
(function() {
    'use strict';
    
    /**
     * PlayerPredictor 클래스에 알고리즘 관련 메서드 추가
     */
    function extendPlayerPredictor() {
        if (typeof PlayerPredictor === 'undefined') {
            // PlayerPredictor가 아직 로드되지 않았으면 100ms 후 재시도
            setTimeout(extendPlayerPredictor, 100);
            return;
        }

        // PlayerPredictor 프로토타입 확장
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
             * 예측 알고리즘 실행 (실제 JSON 데이터 기반)
             */
            async runPredictionAlgorithm(formData) {
                // 시뮬레이션 지연
                await new Promise(resolve => setTimeout(resolve, 800));

                // JSON 데이터 기반 특성 정규화
                const normalizedFeatures = this.normalizeFeatures(formData);
                
                // 클러스터 중심점과의 거리 계산
                const distances = this.calculateClusterDistances(normalizedFeatures);
                
                // 거리를 유사도로 변환
                const similarities = this.distancesToSimilarities(distances);
                
                // 소프트맥스로 확률 변환
                const probabilities = this.softmax(similarities);
                
                // 가장 높은 확률의 클러스터 선택
                const predictedCluster = probabilities.indexOf(Math.max(...probabilities));
                const confidence = probabilities[predictedCluster] * 100;

                return {
                    predictedCluster,
                    confidence,
                    probabilities: probabilities.map(p => p * 100), // 퍼센트로 변환
                    normalizedFeatures,
                    distances,
                    analysis: this.generateAnalysis(normalizedFeatures, predictedCluster, confidence)
                };
            },

            /**
             * 특성 정규화 (JSON 데이터 기반)
             */
            normalizeFeatures(formData) {
                const features = this.getFeatureDefinitions();
                const normalized = [];
                
                features.forEach(feature => {
                    if (formData.hasOwnProperty(feature.name)) {
                        const value = formData[feature.name];
                        const range = feature.max - feature.min;
                        const normalizedValue = range > 0 ? (value - feature.min) / range : 0;
                        normalized.push(Math.max(0, Math.min(1, normalizedValue)) * 100);
                    }
                });
                
                return normalized;
            },

            /**
             * 클러스터 거리 계산 (실제 클러스터 중심점 사용)
             */
            calculateClusterDistances(normalizedFeatures) {
                const clusterInfo = this.getClusterInfo();
                const distances = [];
                
                Object.keys(clusterInfo).forEach(clusterId => {
                    const cluster = clusterInfo[clusterId];
                    if (cluster.features) {
                        // 유클리드 거리 계산
                        let distance = 0;
                        for (let i = 0; i < normalizedFeatures.length && i < cluster.features.length; i++) {
                            const diff = normalizedFeatures[i] - this.normalizeClusterFeature(cluster.features[i], i);
                            distance += diff * diff;
                        }
                        distances.push(Math.sqrt(distance));
                    } else {
                        // 백업 거리 계산
                        distances.push(this.calculateFallbackDistance(normalizedFeatures, parseInt(clusterId)));
                    }
                });
                
                return distances;
            },

            /**
             * 클러스터 특성 정규화
             */
            normalizeClusterFeature(value, featureIndex) {
                const features = this.getFeatureDefinitions();
                if (featureIndex < features.length) {
                    const feature = features[featureIndex];
                    const range = feature.max - feature.min;
                    return range > 0 ? ((value - feature.min) / range) * 100 : 0;
                }
                return value;
            },

            /**
             * 백업 거리 계산 (JSON 로드 실패 시)
             */
            calculateFallbackDistance(normalizedFeatures, clusterId) {
                const clusterCentroids = this.getFallbackClusterCentroids();
                const centroid = clusterCentroids[clusterId];
                
                if (!centroid) {
                    return 100; // 최대 거리
                }
                
                let distance = 0;
                const centroidValues = Object.values(centroid);
                
                for (let i = 0; i < normalizedFeatures.length && i < centroidValues.length; i++) {
                    const diff = normalizedFeatures[i] - centroidValues[i];
                    distance += diff * diff;
                }
                
                return Math.sqrt(distance);
            },

            /**
             * 백업 클러스터 중심점
             */
            getFallbackClusterCentroids() {
                return {
                    0: [14.3, 40, 0.7, 15, 5.7, 3, 1.3, 5, 0, 8],      // Survivor 보수적
                    1: [55.0, 35, 4.0, 30, 9.5, 5.9, 6.9, 12, 20, 12], // Survivor 적극적
                    2: [165.9, 25, 14.0, 45, 14.2, 10, 19.6, 18, 80, 15], // Explorer 활발한
                    3: [165.9, 30, 14.0, 45, 14.2, 10, 19.6, 20, 60, 13], // Explorer 균형형
                    4: [165.9, 20, 14.0, 50, 14.2, 10, 19.6, 25, 120, 18], // Explorer 극한
                    5: [165.9, 22, 14.0, 60, 14.2, 10, 19.6, 22, 90, 14], // Explorer 전술적
                    6: [165.9, 28, 14.0, 48, 14.2, 10, 19.6, 19, 70, 16], // Explorer 지구력
                    7: [264.5, 15, 30.6, 80, 25.9, 20.6, 31.7, 35, 150, 25] // Aggressive
                };
            },

            /**
             * 거리를 유사도로 변환
             */
            distancesToSimilarities(distances) {
                const maxDistance = Math.max(...distances);
                if (maxDistance === 0) {
                    return Array(distances.length).fill(1);
                }
                
                // 거리가 작을수록 유사도가 높음
                return distances.map(distance => 1 / (1 + distance));
            },

            /**
             * 소프트맥스 함수 (확률 분포 생성)
             */
            softmax(similarities, temperature = 2.0) {
                const adjustedSimilarities = similarities.map(s => s * temperature);
                const max = Math.max(...adjustedSimilarities);
                const exponentials = adjustedSimilarities.map(x => Math.exp(x - max));
                const sum = exponentials.reduce((a, b) => a + b, 0);
                return exponentials.map(x => x / sum);
            },

            /**
             * 분석 결과 생성 (JSON 데이터 기반)
             */
            generateAnalysis(normalizedFeatures, predictedCluster, confidence) {
                const features = this.getFeatureDefinitions();
                const clusterInfo = this.getClusterInfo();
                
                // 가장 강한 특성 찾기
                const maxFeatureIndex = normalizedFeatures.indexOf(Math.max(...normalizedFeatures));
                const strongestFeature = features[maxFeatureIndex] ? features[maxFeatureIndex].displayName : '알 수 없음';
                
                // 플레이 스타일 결정
                const playStyles = {
                    0: '매우 신중하고 안전 지향적',
                    1: '균형잡힌 생존 중심',
                    2: '활발한 탐험형',
                    3: '전략적 탐험형',
                    4: '극도로 모험적',
                    5: '계획적 전술형',
                    6: '지구력 기반 지속형',
                    7: '극도로 공격적'
                };

                // 신뢰도 레벨 결정
                const confidenceLevel = this.getConfidenceLevel(confidence);
                
                return {
                    strongestFeature,
                    playStyle: playStyles[predictedCluster] || '분석 중',
                    similarPlayers: clusterInfo[predictedCluster] ? clusterInfo[predictedCluster].percentage.toFixed(1) + '%' : '알 수 없음',
                    confidenceLevel,
                    recommendedActions: this.getRecommendedActions(predictedCluster),
                    technicalInsights: this.getTechnicalInsights(normalizedFeatures, predictedCluster)
                };
            },

            /**
             * 신뢰도 레벨 결정
             */
            getConfidenceLevel(confidence) {
                if (this.isDataLoaded && this.predictionData.predictionThresholds) {
                    const thresholds = this.predictionData.predictionThresholds;
                    if (confidence >= thresholds.highConfidence * 100) {
                        return 'high';
                    }
                    if (confidence >= thresholds.mediumConfidence * 100) {
                        return 'medium';
                    }
                    return 'low';
                }
                
                if (confidence >= 80) {
                    return 'high';
                }
                if (confidence >= 60) {
                    return 'medium';
                }
                return 'low';
            },

            /**
             * 추천 액션 생성
             */
            getRecommendedActions(clusterId) {
                const recommendations = {
                    0: ['안전한 지역 선택', '생존 아이템 우선 수집', '후반 플레이 집중'],
                    1: ['균형잡힌 루팅', '적절한 교전 참여', '팀원과의 협력'],
                    2: ['맵 전체 탐험', '빠른 이동', '다양한 지역 경험'],
                    3: ['전략적 위치 선점', '계획적 이동', '상황 판단력 활용'],
                    4: ['고위험 지역 도전', '빠른 결정', '극한 상황 대응'],
                    5: ['무기 수집 최적화', '전술적 교전', '계획적 플레이'],
                    6: ['장기전 준비', '지속적 생존', '인내심 있는 플레이'],
                    7: ['적극적 교전', '킬 수 증대', '고위험 고수익 플레이']
                };
                
                return recommendations[clusterId] || ['일반적인 플레이'];
            },

            /**
             * 기술적 인사이트 생성
             */
            getTechnicalInsights(normalizedFeatures, clusterId) {
                const features = this.getFeatureDefinitions();
                const insights = [];
                
                // 특성별 강도 분석
                normalizedFeatures.forEach((value, index) => {
                    if (features[index] && value > 70) {
                        insights.push(`${features[index].displayName} 특성이 매우 강함 (${value.toFixed(1)}%)`);
                    }
                });
                
                // 클러스터 특성 분석
                const clusterInfo = this.getClusterInfo();
                if (clusterInfo[clusterId]) {
                    insights.push(`${clusterInfo[clusterId].name} 유형과 매칭`);
                    insights.push(`전체 플레이어 중 ${clusterInfo[clusterId].percentage}% 비율`);
                }
                
                return insights.length > 0 ? insights : ['표준적인 플레이어 특성'];
            },

            /**
             * 폼 데이터 가져오기 (JSON 특성 정의 기반)
             */
            getFormData() {
                const form = document.getElementById('playerForm');
                if (!form) {
                    return {};
                }
                
                const features = this.getFeatureDefinitions();
                const data = {};
                
                features.forEach(feature => {
                    const input = form.querySelector(`input[name="${feature.name}"]`);
                    if (input) {
                        data[feature.name] = parseFloat(input.value) || 0;
                    }
                });
                
                return data;
            },

            /**
             * 모든 입력 검증 (JSON 검증 규칙 적용)
             */
            validateAllInputs(formData) {
                const features = this.getFeatureDefinitions();
                let isValid = true;
                
                features.forEach(feature => {
                    const input = document.querySelector(`input[name="${feature.name}"]`);
                    if (input && !this.validateInput(input)) {
                        isValid = false;
                    }
                });
                
                // JSON 논리적 검증 규칙 적용
                if (this.isDataLoaded && this.predictionData.validationRules) {
                    isValid = isValid && this.validateLogicalRules(formData);
                }
                
                return isValid;
            },

            /**
             * 논리적 검증 규칙 적용
             */
            validateLogicalRules(formData) {
                if (!this.isDataLoaded) {
                    return true;
                }
                
                let isValid = true;
                
                // 킬 수 > 0이면 데미지 > 0
                if (formData.kills > 0 && formData.damageDealt <= 0) {
                    this.showValidationError('킬이 있으면 데미지도 있어야 합니다.');
                    isValid = false;
                }
                
                // 극도로 높은 이동 거리는 생존 시간과 연관
                if (formData.walkDistance > 5000 && formData.matchDuration && formData.matchDuration < 1000) {
                    this.showValidationError('높은 이동 거리는 충분한 게임 시간이 필요합니다.');
                    isValid = false;
                }
                
                // 많은 무기 획득은 최소한의 활동 필요
                if (formData.weaponsAcquired > 10 && formData.walkDistance < 500) {
                    this.showValidationError('많은 무기 획득은 충분한 활동이 필요합니다.');
                    isValid = false;
                }
                
                return isValid;
            },

            /**
             * 검증 오류 표시
             */
            showValidationError(message) {
                if (window.App && window.App.showNotification) {
                    window.App.showNotification(message, 'warning');
                } else {
                    console.warn('검증 오류:', message);
                }
            },

            /**
             * 개별 입력 검증 (JSON 제약 조건 적용)
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

        console.log('✅ PlayerPredictor 알고리즘 확장 완료');
    }

    // 페이지 로드 완료 후 실행
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', extendPlayerPredictor);
    } else {
        extendPlayerPredictor();
    }

})();