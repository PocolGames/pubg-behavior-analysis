/**
 * PUBG 플레이어 행동 분석 - 메인 스크립트
 * 사용자 입력을 받아 플레이어 유형을 예측하는 로직
 */

// 전역 변수
let currentPrediction = null;

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * 앱 초기화
 */
function initializeApp() {
    console.log('PUBG 플레이어 분석 앱 초기화 완료');
    
    // 폼 이벤트 리스너 등록
    const form = document.getElementById('statsForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // 입력 필드 검증 리스너 등록
    setupInputValidation();
}

/**
 * 폼 제출 처리
 */
function handleFormSubmit(event) {
    event.preventDefault();
    
    try {
        // 로딩 표시
        showLoadingState();
        
        // 사용자 입력 수집
        const userStats = collectUserInput();
        
        // 입력 검증
        if (!validateInput(userStats)) {
            hideLoadingState();
            return;
        }
        
        // 30개 특성 계산
        const features = calculateFeatures(userStats);
        
        // 플레이어 유형 예측
        const prediction = predictPlayerType(features);
        
        // 결과 표시
        setTimeout(() => {
            hideLoadingState();
            displayResults(prediction, userStats);
        }, 1000);
        
    } catch (error) {
        console.error('분석 중 오류 발생:', error);
        hideLoadingState();
        // 고급 에러 처리
        if (window.ErrorHandler) {
            window.ErrorHandler.handleAnalysisError('CALCULATION_FAILED', userStats);
            window.ErrorHandler.logError('form_submission', error, 'user_initiated_analysis');
        } else {
            showErrorMessage('분석 중 오류가 발생했습니다. 다시 시도해주세요.');
        }
    }
}

/**
 * 사용자 입력 수집
 */
function collectUserInput() {
    return {
        // 전투 스탯
        kills: parseInt(document.getElementById('kills').value) || 0,
        assists: parseInt(document.getElementById('assists').value) || 0,
        damageDealt: parseFloat(document.getElementById('damageDealt').value) || 0,
        headshotKills: parseInt(document.getElementById('headshotKills').value) || 0,
        
        // 이동 스탯
        walkDistance: parseFloat(document.getElementById('walkDistance').value) || 0,
        rideDistance: parseFloat(document.getElementById('rideDistance').value) || 0,
        swimDistance: parseFloat(document.getElementById('swimDistance').value) || 0,
        longestKill: parseFloat(document.getElementById('longestKill').value) || 0,
        
        // 생존 스탯
        heals: parseInt(document.getElementById('heals').value) || 0,
        boosts: parseInt(document.getElementById('boosts').value) || 0,
        revives: parseInt(document.getElementById('revives').value) || 0,
        weaponsAcquired: parseInt(document.getElementById('weaponsAcquired').value) || 0,
        
        // 게임 정보
        killPlace: parseInt(document.getElementById('killPlace').value) || 50,
        matchDuration: parseInt(document.getElementById('matchDuration').value) || 1800
    };
}

/**
 * 입력 검증
 */
function validateInput(stats) {
    const errors = [];
    
    // 기본 검증
    if (stats.kills < 0 || stats.kills > 50) {
        errors.push('킬 수는 0~50 사이여야 합니다.');
    }
    
    if (stats.damageDealt < 0 || stats.damageDealt > 5000) {
        errors.push('데미지는 0~5000 사이여야 합니다.');
    }
    
    if (stats.walkDistance < 0 || stats.walkDistance > 10000) {
        errors.push('보행 거리는 0~10000m 사이여야 합니다.');
    }
    
    if (stats.killPlace < 1 || stats.killPlace > 100) {
        errors.push('킬 순위는 1~100 사이여야 합니다.');
    }
    
    if (stats.matchDuration < 0 || stats.matchDuration > 3000) {
        errors.push('게임 시간은 0~3000초 사이여야 합니다.');
    }
    
    // 논리적 검증
    if (stats.headshotKills > stats.kills) {
        errors.push('헤드샷 킬은 총 킬 수보다 클 수 없습니다.');
    }
    
    if (stats.assists > stats.kills * 3) {
        errors.push('어시스트가 비정상적으로 높습니다. (킬의 3배 이하로 입력해주세요)');
    }
    
    if (stats.longestKill > 0 && stats.kills === 0) {
        errors.push('킬이 0인데 최장 킬 거리가 있을 수 없습니다.');
    }
    
    if (stats.revives > 10) {
        errors.push('팀원 부활 횟수가 비정상적으로 높습니다.');
    }
    
    if (errors.length > 0) {
        showErrorMessage(errors.join('\n'));
        return false;
    }
    
    return true;
}

/**
 * 30개 특성 계산
 */
function calculateFeatures(stats) {
    // 파생 특성 계산
    const totalDistance = stats.walkDistance + stats.rideDistance + stats.swimDistance;
    const totalHeals = stats.heals + stats.boosts;
    const killEfficiency = stats.kills / (stats.damageDealt + 1);
    const damagePerKill = stats.damageDealt / (stats.kills + 1);
    const healBoostRatio = stats.heals / (totalHeals + 1);
    const aggressivenessScore = stats.kills * 0.4 + stats.damageDealt * 0.001;
    
    // 로그 변환
    const walkDistanceLog = Math.log1p(stats.walkDistance);
    const damageDealtLog = Math.log1p(stats.damageDealt);
    
    // 이진 특성
    const hasKills = stats.kills > 0 ? 1 : 0;
    const hasSwimDistance = stats.swimDistance > 0 ? 1 : 0;
    
    // 추가 특성 (기본값으로 설정)
    const numGroups = 25; // 기본값
    const maxPlace = 100; // 기본값
    const DBNOs = Math.max(0, Math.floor(stats.kills * 1.1 + stats.assists * 0.5)); // 더 정확한 추정
    const roadKills = 0; // 기본값
    const vehicleDestroys = 0; // 기본값
    const teamKills = 0; // 기본값
    
    // 30개 특성 배열 생성 (FEATURE_NAMES 순서와 일치)
    const features = [
        stats.walkDistance,     // walkDistance
        walkDistanceLog,        // walkDistance_log
        stats.killPlace,        // killPlace
        totalDistance,          // total_distance
        stats.boosts,           // boosts
        stats.damageDealt,      // damageDealt
        stats.heals,            // heals
        stats.weaponsAcquired,  // weaponsAcquired
        stats.kills,            // kills
        stats.assists,          // assists
        stats.rideDistance,     // rideDistance
        stats.longestKill,      // longestKill
        stats.matchDuration,    // matchDuration
        stats.revives,          // revives
        killEfficiency,         // kill_efficiency
        damagePerKill,          // damage_per_kill
        totalHeals,             // total_heals
        healBoostRatio,         // heal_boost_ratio
        aggressivenessScore,    // aggressiveness_score
        damageDealtLog,         // damageDealt_log
        hasKills,               // has_kills
        hasSwimDistance,        // has_swimDistance
        numGroups,              // numGroups
        maxPlace,               // maxPlace
        DBNOs,                  // DBNOs
        stats.headshotKills,    // headshotKills
        stats.swimDistance,     // swimDistance
        roadKills,              // roadKills
        vehicleDestroys,        // vehicleDestroys
        teamKills               // teamKills
    ];
    
    return features;
}

/**
 * 특성 표준화
 */
function standardizeFeatures(features) {
    // 간단한 표준화 (Z-score 정규화)
    // 실제 분석 데이터에서 계산된 평균과 표준편차 사용
    const means = [
        1147.4, 6.8, 47.6, 1748.3, 1.1, 129.2, 1.37, 3.66, 0.91, 0.23,
        596.4, 22.95, 1800, 0.16, 0.008, 144.5, 2.47, 0.55, 1.4, 4.86,
        0.43, 0.044, 25, 100, 1.08, 0.23, 4.5, 0.002, 0.008, 0.027
    ];
    
    const stds = [
        1164.2, 1.18, 27.5, 1580.8, 1.71, 161.1, 2.68, 2.46, 1.46, 0.59,
        1445.1, 35.2, 300, 0.47, 0.015, 185.3, 2.85, 0.42, 1.8, 1.45,
        0.49, 0.20, 4.5, 18.3, 1.15, 0.60, 30.7, 0.048, 0.089, 0.17
    ];
    
    return features.map((feature, index) => {
        return (feature - means[index]) / stds[index];
    });
}

/**
 * 플레이어 유형 예측
 */
function predictPlayerType(features) {
    // 특성 표준화
    const normalizedFeatures = standardizeFeatures(features);
    
    // 각 클러스터와의 거리 계산
    const distances = {};
    const clusterCenters = window.CLUSTER_DATA.CLUSTER_CENTERS;
    
    for (let clusterId in clusterCenters) {
        const center = clusterCenters[clusterId];
        const distance = calculateEuclideanDistance(normalizedFeatures, center);
        distances[clusterId] = distance;
    }
    
    // 가장 가까운 클러스터 찾기
    const closestCluster = Object.keys(distances).reduce((a, b) => 
        distances[a] < distances[b] ? a : b
    );
    
    // 신뢰도 계산 (가장 가까운 거리와 두 번째로 가까운 거리의 비율)
    const sortedDistances = Object.values(distances).sort((a, b) => a - b);
    const confidence = Math.max(0, Math.min(1, 
        1 - (sortedDistances[0] / (sortedDistances[1] + 0.001))
    ));
    
    // 모든 클러스터별 확률 계산
    const maxDistance = Math.max(...Object.values(distances));
    const probabilities = {};
    
    for (let clusterId in distances) {
        // 거리가 가까울수록 높은 확률
        probabilities[clusterId] = Math.max(0, 
            (maxDistance - distances[clusterId]) / maxDistance
        );
    }
    
    // 확률 정규화
    const totalProb = Object.values(probabilities).reduce((sum, prob) => sum + prob, 0);
    for (let clusterId in probabilities) {
        probabilities[clusterId] = probabilities[clusterId] / totalProb;
    }
    
    return {
        predictedCluster: parseInt(closestCluster),
        confidence: confidence,
        probabilities: probabilities,
        distances: distances,
        clusterInfo: window.CLUSTER_DATA.CLUSTER_INFO[closestCluster]
    };
}

/**
 * 유클리드 거리 계산
 */
function calculateEuclideanDistance(point1, point2) {
    if (point1.length !== point2.length) {
        throw new Error('점들의 차원이 일치하지 않습니다.');
    }
    
    const squaredSum = point1.reduce((sum, value, index) => {
        const diff = value - point2[index];
        return sum + (diff * diff);
    }, 0);
    
    return Math.sqrt(squaredSum);
}

/**
 * 결과 표시
 */
function displayResults(prediction, userStats) {
    currentPrediction = prediction;
    
    // 결과 섹션 표시
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    
    // 메인 결과 카드 업데이트
    updateMainResultCard(prediction);
    
    // 스타일 분석 업데이트
    updateStyleAnalysis(prediction, userStats);
    
    // 개선 제안 업데이트
    updateImprovementSuggestions(prediction, userStats);
    
    // 결과로 스크롤
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * 메인 결과 카드 업데이트
 */
function updateMainResultCard(prediction) {
    const clusterInfo = prediction.clusterInfo;
    const confidence = (prediction.confidence * 100).toFixed(1);
    
    const resultContent = document.getElementById('result-content');
    const mainResultCard = document.getElementById('main-result-card');
    
    // 카드에 클러스터 타입에 따른 테두리 색상 적용
    mainResultCard.className = 'card shadow-lg';
    if (clusterInfo.type === 'Survivor') {
        mainResultCard.classList.add('border-survivor');
    } else if (clusterInfo.type === 'Explorer') {
        mainResultCard.classList.add('border-explorer');
    } else if (clusterInfo.type === 'Aggressive') {
        mainResultCard.classList.add('border-aggressive');
    }
    
    resultContent.innerHTML = `
        <div class="player-type-result">
            <div class="type-icon mb-3" style="font-size: 4rem;">
                ${clusterInfo.icon}
            </div>
            <h2 class="mb-3" style="color: ${clusterInfo.color};">
                ${clusterInfo.name}
            </h2>
            <div class="badge mb-3" style="background-color: ${clusterInfo.color}20; color: ${clusterInfo.color}; font-size: 0.9rem;">
                ${clusterInfo.type} 타입 • ${clusterInfo.percentage}% 비율
            </div>
            <p class="lead mb-4">${clusterInfo.description}</p>
            
            <div class="confidence-meter mb-4">
                <h6>분석 신뢰도</h6>
                <div class="progress mb-2" style="height: 20px;">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${confidence}%; background-color: ${clusterInfo.color};">
                        ${confidence}%
                    </div>
                </div>
                <small class="text-muted">
                    ${confidence >= 80 ? '매우 높은 신뢰도' : 
                      confidence >= 60 ? '높은 신뢰도' : 
                      confidence >= 40 ? '보통 신뢰도' : '낮은 신뢰도'}
                </small>
            </div>
            
            <div class="row text-start">
                <div class="col-md-6">
                    <h6 class="text-success">
                        <i class="fas fa-plus-circle me-2"></i>강점
                    </h6>
                    <ul class="list-unstyled">
                        ${clusterInfo.strengths.map(strength => 
                            `<li><i class="fas fa-check text-success me-2"></i>${strength}</li>`
                        ).join('')}
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6 class="text-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>개선점
                    </h6>
                    <ul class="list-unstyled">
                        ${clusterInfo.weaknesses.map(weakness => 
                            `<li><i class="fas fa-arrow-up text-warning me-2"></i>${weakness}</li>`
                        ).join('')}
                    </ul>
                </div>
            </div>
        </div>
    `;
}

/**
 * 스타일 분석 업데이트
 */
function updateStyleAnalysis(prediction, userStats) {
    const clusterInfo = prediction.clusterInfo;
    const styleAnalysis = document.getElementById('style-analysis');
    
    styleAnalysis.innerHTML = `
        <div class="style-details">
            <div class="mb-4">
                <h6 class="text-primary">
                    <i class="fas fa-chess me-2"></i>플레이 전략
                </h6>
                <p class="mb-3">${clusterInfo.strategy}</p>
            </div>
            
            <div class="mb-4">
                <h6 class="text-primary">
                    <i class="fas fa-star me-2"></i>주요 특징
                </h6>
                <ul class="list-unstyled">
                    ${clusterInfo.characteristics.map(char => 
                        `<li class="mb-2"><i class="fas fa-chevron-right text-primary me-2"></i>${char}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="stats-summary">
                <h6 class="text-primary">
                    <i class="fas fa-chart-bar me-2"></i>입력한 스탯 요약
                </h6>
                <div class="row">
                    <div class="col-4">
                        <small class="text-muted">전투력</small><br>
                        <strong>${userStats.kills}킬 / ${userStats.damageDealt}dmg</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted">이동력</small><br>
                        <strong>${(userStats.walkDistance + userStats.rideDistance).toLocaleString()}m</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted">생존력</small><br>
                        <strong>${userStats.heals + userStats.boosts}회</strong>
                    </div>
                </div>
                
                <div class="mt-3">
                    <h6 class="text-primary mb-2">
                        <i class="fas fa-users me-2"></i>플레이어 유형 분포
                    </h6>
                    <div id="playerDistributionBars">
                        ${createPlayerDistributionBars(prediction)}
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 개선 제안 업데이트
 */
function updateImprovementSuggestions(prediction, userStats) {
    const clusterInfo = prediction.clusterInfo;
    const suggestions = document.getElementById('improvement-suggestions');
    
    // 기본 제안 생성
    let improvementTips = [];
    
    // 킬 관련 제안
    if (userStats.kills < 2) {
        improvementTips.push("더 적극적인 교전을 통해 킬 수를 늘려보세요.");
    }
    
    // 이동 관련 제안
    if (userStats.walkDistance < 1000) {
        improvementTips.push("맵을 더 활발히 탐험하여 좋은 포지션을 찾아보세요.");
    }
    
    // 생존 관련 제안
    if (userStats.heals + userStats.boosts < 3) {
        improvementTips.push("치료 아이템을 더 적극적으로 사용하여 생존률을 높이세요.");
    }
    
    // 팀플레이 관련 제안
    if (userStats.assists + userStats.revives < 2) {
        improvementTips.push("팀원과의 협력을 강화하여 팀플레이를 개선해보세요.");
    }
    
    // 기본 제안이 없다면 일반적인 제안 추가
    if (improvementTips.length === 0) {
        improvementTips = [
            "현재 플레이 스타일을 유지하면서 약점을 보완해보세요.",
            "다양한 전략을 시도하여 플레이의 폭을 넓혀보세요.",
            "팀원과의 소통을 통해 더 나은 성과를 달성해보세요."
        ];
    }
    
    suggestions.innerHTML = `
        <div class="improvement-tips">
            <div class="mb-4">
                <h6 class="text-primary">맞춤형 개선 제안</h6>
                <ul class="list-unstyled">
                    ${improvementTips.map(tip => 
                        `<li class="mb-2"><i class="fas fa-lightbulb text-warning me-2"></i>${tip}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="type-distribution">
                <h6 class="text-primary">전체 플레이어 분포</h6>
                <p class="small text-muted mb-2">
                    ${clusterInfo.name} 유형은 전체 플레이어의 ${clusterInfo.percentage}%입니다.
                </p>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${clusterInfo.percentage}%; background-color: ${clusterInfo.color};">
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 유틸리티 함수들
 */

// 분석 섹션으로 스크롤
function scrollToAnalysis() {
    document.getElementById('analysis-section').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// 분석 초기화
function resetAnalysis() {
    // 폼 초기화
    document.getElementById('statsForm').reset();
    
    // 결과 섹션 숨기기
    document.getElementById('results-section').style.display = 'none';
    
    // 분석 섹션으로 스크롤
    scrollToAnalysis();
    
    currentPrediction = null;
}

// 로딩 상태 표시
function showLoadingState() {
    const submitBtn = document.querySelector('#statsForm button[type="submit"]');
    if (submitBtn) {
        submitBtn.innerHTML = `
            <i class="fas fa-spinner fa-spin me-2"></i>
            분석 중...
        `;
        submitBtn.disabled = true;
    }
}

// 로딩 상태 숨기기
function hideLoadingState() {
    const submitBtn = document.querySelector('#statsForm button[type="submit"]');
    if (submitBtn) {
        submitBtn.innerHTML = `
            <i class="fas fa-magic me-2"></i>
            플레이어 유형 분석하기
        `;
        submitBtn.disabled = false;
    }
}

// 에러 메시지 표시
function showErrorMessage(message) {
    // 기존 에러 메시지 제거
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // 새 에러 메시지 생성
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger error-message mt-3';
    errorDiv.innerHTML = `
        <h6><i class="fas fa-exclamation-triangle me-2"></i>입력 오류</h6>
        <p class="mb-0">${message.replace(/\n/g, '<br>')}</p>
    `;
    
    // 폼 아래에 추가
    const form = document.getElementById('statsForm');
    if (form) {
        form.appendChild(errorDiv);
        
        // 에러 메시지로 스크롤
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // 5초 후 자동 제거
        setTimeout(() => {
            if (errorDiv && errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }
}

// 입력 검증 설정
function setupInputValidation() {
    const inputs = document.querySelectorAll('#statsForm input[type="number"]');
    
    inputs.forEach(input => {
        // 실시간 검증
        input.addEventListener('input', function() {
            validateSingleInput(this);
        });
        
        // 포커스 잃을 때 검증
        input.addEventListener('blur', function() {
            validateSingleInput(this);
        });
    });
    
    console.log('실시간 입력 검증 설정 완료');
}

// 플레이어 분포 막대 그래프 생성 (Chart.js 대신 CSS로)
function createPlayerDistributionBars(prediction) {
    const clusterInfo = window.CLUSTER_DATA.CLUSTER_INFO;
    const currentCluster = prediction.predictedCluster;
    
    let barsHTML = '';
    
    for (let clusterId in clusterInfo) {
        const info = clusterInfo[clusterId];
        const isCurrentPlayer = parseInt(clusterId) === currentCluster;
        const barClass = isCurrentPlayer ? 'bg-primary' : 'bg-secondary';
        const opacity = isCurrentPlayer ? '1' : '0.3';
        
        barsHTML += `
            <div class="mb-2">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <small class="fw-bold ${isCurrentPlayer ? 'text-primary' : 'text-muted'}">
                        ${info.icon} ${info.name}
                        ${isCurrentPlayer ? '<i class="fas fa-arrow-left ms-1"></i>' : ''}
                    </small>
                    <small class="text-muted">${info.percentage}%</small>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar ${barClass}" 
                         style="width: ${info.percentage}%; opacity: ${opacity}; background-color: ${info.color};">
                    </div>
                </div>
            </div>
        `;
    }
    
    return barsHTML;
}

// 개별 입력 필드 검증
function validateSingleInput(input) {
    const value = parseFloat(input.value) || 0;
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || Infinity;
    
    // 기존 에러 표시 제거
    input.classList.remove('is-invalid', 'is-valid');
    const existingFeedback = input.parentNode.querySelector('.invalid-feedback');
    if (existingFeedback) {
        existingFeedback.remove();
    }
    
    let isValid = true;
    let errorMessage = '';
    
    // 범위 검증
    if (value < min) {
        isValid = false;
        errorMessage = `최소값: ${min}`;
    } else if (value > max) {
        isValid = false;
        errorMessage = `최대값: ${max}`;
    }
    
    // 논리적 검증 (특정 필드)
    if (input.id === 'headshotKills') {
        const killsValue = parseInt(document.getElementById('kills').value) || 0;
        if (value > killsValue) {
            isValid = false;
            errorMessage = '헤드샷 킬은 총 킬보다 클 수 없습니다';
        }
    }
    
    if (input.id === 'longestKill') {
        const killsValue = parseInt(document.getElementById('kills').value) || 0;
        if (value > 0 && killsValue === 0) {
            isValid = false;
            errorMessage = '킬이 0이면 최장 킬 거리도 0이어야 합니다';
        }
    }
    
    // 시각적 피드백
    if (input.value && !isValid) {
        input.classList.add('is-invalid');
        
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        feedback.textContent = errorMessage;
        input.parentNode.appendChild(feedback);
    } else if (input.value) {
        input.classList.add('is-valid');
    }
}
