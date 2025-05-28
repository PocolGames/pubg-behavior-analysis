// 플레이어 분류기 클래스
class PlayerClassifier {
    constructor() {
        this.features = Object.keys(FEATURE_WEIGHTS);
    }

    // 입력 데이터 정규화
    normalizeData(data) {
        const normalized = {};
        
        for (const [key, value] of Object.entries(data)) {
            if (STAT_RANGES[key]) {
                const range = STAT_RANGES[key];
                // 0-1 범위로 정규화
                normalized[key] = Math.min(Math.max((value - range.min) / (range.max - range.min), 0), 1);
            } else {
                normalized[key] = value;
            }
        }
        
        return normalized;
    }

    // 파생 특성 계산
    calculateDerivedFeatures(data) {
        const derived = { ...data };
        
        // 총 이동거리
        derived.totalDistance = data.walkDistance + data.rideDistance;
        
        // 총 치료 아이템
        derived.totalHeals = data.heals + data.boosts;
        
        // 킬 효율성 (킬/데미지)
        derived.killEfficiency = data.damageDealt > 0 ? data.kills / data.damageDealt : 0;
        
        // 치료 비율 (치료/총행동)
        const totalActions = data.kills + data.heals + data.boosts + data.assists;
        derived.healRatio = totalActions > 0 ? derived.totalHeals / totalActions : 0;
        
        // 킬 비율
        derived.killRatio = totalActions > 0 ? data.kills / totalActions : 0;
        
        // 이동 비율
        derived.distanceRatio = derived.totalDistance > 0 ? data.walkDistance / derived.totalDistance : 0;
        
        return derived;
    }

    // 각 유형별 점수 계산
    calculateTypeScores(data) {
        const scores = {
            aggressive: 0,
            survivor: 0,
            explorer: 0
        };

        // Aggressive 점수 계산
        scores.aggressive += data.kills * 0.3;
        scores.aggressive += (data.damageDealt / 100) * 0.2;
        scores.aggressive += data.killRatio * 0.2;
        scores.aggressive += data.killEfficiency * 100 * 0.15;
        scores.aggressive += (data.weaponsAcquired / 10) * 0.1;
        scores.aggressive -= data.healRatio * 0.05; // 치료 많이 하면 감점

        // Survivor 점수 계산
        scores.survivor += data.healRatio * 0.4;
        scores.survivor += (data.heals / 10) * 0.2;
        scores.survivor += (data.boosts / 10) * 0.15;
        scores.survivor += (data.assists / 5) * 0.1;
        scores.survivor += (data.totalDistance / 5000) * 0.1;
        scores.survivor -= data.killRatio * 0.05; // 킬 많이 하면 감점

        // Explorer 점수 계산
        scores.explorer += (data.walkDistance / 3000) * 0.4;
        scores.explorer += (data.totalDistance / 5000) * 0.3;
        scores.explorer += data.distanceRatio * 0.15;
        scores.explorer += (data.weaponsAcquired / 10) * 0.1;
        scores.explorer += Math.min(data.kills / 10, 0.3) * 0.05; // 적당한 킬은 가점

        return scores;
    }

    // 확률 분포 계산
    calculateProbabilities(scores) {
        const total = Object.values(scores).reduce((sum, score) => sum + Math.max(score, 0), 0);
        
        if (total === 0) {
            // 모든 점수가 0이면 균등 분배
            return {
                aggressive: 0.33,
                survivor: 0.33,
                explorer: 0.34
            };
        }

        const probabilities = {};
        for (const [type, score] of Object.entries(scores)) {
            probabilities[type] = Math.max(score, 0) / total;
        }

        return probabilities;
    }

    // 신뢰도 계산
    calculateConfidence(probabilities) {
        const values = Object.values(probabilities);
        const maxProb = Math.max(...values);
        const secondMaxProb = values.sort((a, b) => b - a)[1];
        
        // 최고 확률과 두 번째 확률의 차이로 신뢰도 계산
        const confidence = Math.min((maxProb - secondMaxProb) + 0.5, 1.0);
        return Math.max(confidence, 0.6); // 최소 60% 신뢰도 보장
    }

    // 이상치 탐지
    detectAnomaly(data) {
        const thresholds = {
            kills: 20,
            damageDealt: 1500,
            walkDistance: 7000,
            heals: 12,
            boosts: 8
        };

        for (const [key, threshold] of Object.entries(thresholds)) {
            if (data[key] > threshold) {
                return true;
            }
        }

        return false;
    }

    // 메인 분류 함수
    classify(inputData) {
        // 1. 파생 특성 계산
        const enrichedData = this.calculateDerivedFeatures(inputData);
        
        // 2. 각 유형별 점수 계산
        const scores = this.calculateTypeScores(enrichedData);
        
        // 3. 확률 분포 계산
        const probabilities = this.calculateProbabilities(scores);
        
        // 4. 최고 확률 유형 선택
        const predictedType = Object.keys(probabilities).reduce((a, b) => 
            probabilities[a] > probabilities[b] ? a : b
        );
        
        // 5. 신뢰도 계산
        const confidence = this.calculateConfidence(probabilities);
        
        // 6. 이상치 탐지
        const isAnomaly = this.detectAnomaly(inputData);
        
        return {
            predictedType,
            probabilities,
            confidence,
            isAnomaly,
            scores,
            processedData: enrichedData
        };
    }

    // 인사이트 생성
    generateInsights(inputData, result) {
        const insights = [];
        const data = result.processedData;

        // 킬 관련 인사이트
        if (inputData.kills >= 5) {
            insights.push({
                icon: '⚔️',
                text: `높은 킬 수 (${inputData.kills}개)로 공격적인 플레이 스타일을 보입니다.`
            });
        } else if (inputData.kills <= 1) {
            insights.push({
                icon: '🛡️',
                text: '낮은 킬 수로 신중하고 생존 중심적인 플레이를 합니다.'
            });
        }

        // 이동 관련 인사이트
        if (data.totalDistance >= 3000) {
            insights.push({
                icon: '🗺️',
                text: `높은 이동거리 (${Math.round(data.totalDistance)}m)로 맵 탐험을 즐기는 스타일입니다.`
            });
        }

        // 치료 관련 인사이트
        if (data.totalHeals >= 4) {
            insights.push({
                icon: '💊',
                text: '치료 아이템을 자주 사용하여 생존력이 뛰어납니다.'
            });
        }

        // 팀플레이 관련 인사이트
        if (inputData.assists >= 3) {
            insights.push({
                icon: '🤝',
                text: '높은 어시스트로 팀워크를 중시하는 플레이어입니다.'
            });
        }

        // 효율성 관련 인사이트
        if (data.killEfficiency > 0.01) {
            insights.push({
                icon: '🎯',
                text: '데미지 대비 킬 효율성이 높아 정확한 플레이를 합니다.'
            });
        }

        return insights;
    }
}

// 전역 분류기 인스턴스
const classifier = new PlayerClassifier();