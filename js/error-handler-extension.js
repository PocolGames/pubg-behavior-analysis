/**
 * PUBG 플레이어 분석 - 에러 처리 확장 모듈
 * script.js의 에러 처리 기능을 향상시키는 추가 함수들
 */

// 향상된 입력 검증
function validateInputEnhanced(stats) {
    // 기존 에러 상태 초기화
    if (window.ErrorHandler) {
        window.ErrorHandler.clearAllErrors();
    }

    const errors = [];
    const warnings = [];
    
    // 기본 검증
    if (stats.kills < 0 || stats.kills > 50) {
        errors.push({ field: 'kills', message: '킬 수는 0~50 사이여야 합니다.', type: 'range' });
    }
    if (stats.damageDealt < 0 || stats.damageDealt > 5000) {
        errors.push({ field: 'damageDealt', message: '데미지는 0~5000 사이여야 합니다.', type: 'range' });
    }
    if (stats.walkDistance < 0 || stats.walkDistance > 10000) {
        errors.push({ field: 'walkDistance', message: '보행 거리는 0~10000m 사이여야 합니다.', type: 'range' });
    }
    if (stats.killPlace < 1 || stats.killPlace > 100) {
        errors.push({ field: 'killPlace', message: '킬 순위는 1~100 사이여야 합니다.', type: 'range' });
    }
    if (stats.matchDuration < 0 || stats.matchDuration > 3000) {
        errors.push({ field: 'matchDuration', message: '게임 시간은 0~3000초 사이여야 합니다.', type: 'range' });
    }
    
    // 논리적 검증
    if (stats.headshotKills > stats.kills) {
        errors.push({ field: 'headshotKills', message: '헤드샷 킬은 총 킬 수보다 클 수 없습니다.', type: 'logic' });
    }
    if (stats.assists > stats.kills * 3) {
        errors.push({ field: 'assists', message: '어시스트가 비정상적으로 높습니다.', type: 'logic' });
    }
    if (stats.longestKill > 0 && stats.kills === 0) {
        errors.push({ field: 'longestKill', message: '킬이 0인데 최장 킬 거리가 있을 수 없습니다.', type: 'logic' });
    }
    if (stats.revives > 10) {
        errors.push({ field: 'revives', message: '팀원 부활 횟수가 비정상적으로 높습니다.', type: 'logic' });
    }
    
    // 현실성 검사 (경고)
    if (stats.kills > 20) {
        warnings.push({ field: 'kills', message: '킬 수가 매우 높습니다. 정확한 값인지 확인해주세요.', type: 'realistic' });
    }
    if (stats.damageDealt > 0 && stats.kills === 0) {
        warnings.push({ field: 'damageDealt', message: '데미지는 있지만 킬이 0입니다. 어시스트만 있었던 게임인가요?', type: 'realistic' });
    }
    if (stats.walkDistance > 5000) {
        warnings.push({ field: 'walkDistance', message: '보행 거리가 매우 깁니다. 정확한 값인지 확인해주세요.', type: 'realistic' });
    }
    
    // 고급 에러 처리
    if (errors.length > 0) {
        if (window.ErrorHandler) {
            // 토스트로 전체 에러 요약
            window.ErrorHandler.showToast('error', '입력 검증 실패', 
                `${errors.length}개의 오류가 발견되었습니다. 입력값을 확인해주세요.`);
            
            // 각 에러에 대해 개별 필드 처리
            errors.forEach(error => {
                window.ErrorHandler.handleValidationError(error.field, error.type, error.message);
            });
            
            // 에러 로깅
            window.ErrorHandler.logError('input_validation', errors.map(e => e.message), 'form_validation');
        } else {
            // 폴백: 기존 방식
            showErrorMessage(errors.map(e => e.message).join('\n'));
        }
        return false;
    }
    
    // 경고 처리
    if (warnings.length > 0 && window.ErrorHandler) {
        warnings.forEach(warning => {
            window.ErrorHandler.showToast('warning', '입력값 확인', warning.message);
        });
    }
    
    return true;
}

// 향상된 폼 제출 처리
function handleFormSubmitEnhanced(event) {
    event.preventDefault();
    
    try {
        // 에러 상태 초기화
        if (window.ErrorHandler) {
            window.ErrorHandler.clearAllErrors();
        }
        
        // 로딩 표시
        showLoadingState();
        
        // 네트워크 상태 확인
        if (!navigator.onLine) {
            hideLoadingState();
            if (window.ErrorHandler) {
                window.ErrorHandler.handleNetworkError('플레이어 분석');
            } else {
                showErrorMessage('네트워크 연결을 확인해주세요.');
            }
            return;
        }
        
        // 필수 데이터 확인
        if (!window.CLUSTER_DATA || !window.CLUSTER_DATA.CLUSTER_CENTERS) {
            hideLoadingState();
            if (window.ErrorHandler) {
                window.ErrorHandler.handleAnalysisError('DATA_MISSING', null);
            } else {
                showErrorMessage('분석에 필요한 데이터를 불러올 수 없습니다.');
            }
            return;
        }
        
        // 사용자 입력 수집
        const userStats = collectUserInput();
        
        // 향상된 입력 검증
        if (!validateInputEnhanced(userStats)) {
            hideLoadingState();
            return;
        }
        
        // 30개 특성 계산 (에러 처리 포함)
        let features;
        try {
            features = calculateFeatures(userStats);
            
            // 특성 유효성 검사
            if (!features || features.length !== 30) {
                throw new Error('특성 계산 결과가 올바르지 않습니다.');
            }
            
            // NaN 또는 무한값 검사
            if (features.some(f => !isFinite(f))) {
                throw new Error('계산된 특성에 유효하지 않은 값이 포함되어 있습니다.');
            }
            
        } catch (calcError) {
            hideLoadingState();
            console.error('특성 계산 오류:', calcError);
            
            if (window.ErrorHandler) {
                window.ErrorHandler.handleAnalysisError('CALCULATION_FAILED', userStats);
                window.ErrorHandler.logError('feature_calculation', calcError, userStats);
            } else {
                showErrorMessage('특성 계산 중 오류가 발생했습니다. 입력값을 확인해주세요.');
            }
            return;
        }
        
        // 플레이어 유형 예측 (에러 처리 포함)
        let prediction;
        try {
            prediction = predictPlayerType(features);
            
            // 예측 결과 유효성 검사
            if (!prediction || 
                prediction.predictedCluster === undefined || 
                prediction.predictedCluster === null || 
                !prediction.clusterInfo) {
                throw new Error('예측 결과가 올바르지 않습니다.');
            }
            
        } catch (predError) {
            hideLoadingState();
            console.error('예측 오류:', predError);
            
            if (window.ErrorHandler) {
                window.ErrorHandler.handleAnalysisError('PREDICTION_FAILED', userStats);
                window.ErrorHandler.logError('prediction', predError, { userStats, features });
            } else {
                showErrorMessage('플레이어 유형 예측에 실패했습니다. 다시 시도해주세요.');
            }
            return;
        }
        
        // 결과 표시
        setTimeout(() => {
            try {
                hideLoadingState();
                displayResults(prediction, userStats);
                
                // 성공 알림
                if (window.ErrorHandler) {
                    window.ErrorHandler.showToast('success', '분석 완료', 
                        `${prediction.clusterInfo.name} 유형으로 분석되었습니다 (신뢰도: ${Math.round(prediction.confidence * 100)}%)`);
                }
                
            } catch (displayError) {
                console.error('결과 표시 오류:', displayError);
                if (window.ErrorHandler) {
                    window.ErrorHandler.showToast('error', '결과 표시 오류', 
                        '분석은 완료되었지만 결과 표시 중 오류가 발생했습니다.');
                }
            }
        }, 1000);
        
    } catch (error) {
        console.error('분석 중 예상치 못한 오류:', error);
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

// 향상된 개별 입력 검증
function validateSingleInputEnhanced(input) {
    if (!input) return;
    
    const value = parseFloat(input.value) || 0;
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || Infinity;
    
    // 기존 상태 초기화
    if (window.ErrorHandler) {
        window.ErrorHandler.clearFieldError(input.id);
    } else {
        input.classList.remove('is-invalid', 'is-valid');
        const existingFeedback = input.parentNode.querySelector('.invalid-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }
    }
    
    let isValid = true;
    let errorType = 'range';
    let customMessage = null;
    
    // 범위 검증
    if (value < min) {
        isValid = false;
        errorType = 'min';
        customMessage = `최소값은 ${min}입니다`;
    } else if (value > max) {
        isValid = false;
        errorType = 'max';
        customMessage = `최대값은 ${max}입니다`;
    }
    
    // 논리적 검증
    if (input.id === 'headshotKills') {
        const killsInput = document.getElementById('kills');
        const killsValue = killsInput ? parseInt(killsInput.value) || 0 : 0;
        if (value > killsValue) {
            isValid = false;
            errorType = 'logic';
            customMessage = `헤드샷 킬(${value})은 총 킬 수(${killsValue})보다 클 수 없습니다`;
        }
    }
    
    if (input.id === 'longestKill') {
        const killsInput = document.getElementById('kills');
        const killsValue = killsInput ? parseInt(killsInput.value) || 0 : 0;
        if (value > 0 && killsValue === 0) {
            isValid = false;
            errorType = 'logic';
            customMessage = '킬이 0이면 최장 킬 거리도 0이어야 합니다';
        }
    }
    
    // 결과 처리
    if (input.value && !isValid) {
        if (window.ErrorHandler) {
            window.ErrorHandler.handleValidationError(input.id, errorType, customMessage);
        } else {
            // 폴백: 기존 방식
            input.classList.add('is-invalid');
            const feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            feedback.textContent = customMessage || '입력값이 올바르지 않습니다';
            input.parentNode.appendChild(feedback);
        }
    } else if (input.value) {
        if (window.ErrorHandler) {
            window.ErrorHandler.showFieldSuccess(input.id);
        } else {
            input.classList.add('is-valid');
        }
    }
}

// 에러 복구 함수
function recoverFromError() {
    console.log('🔄 에러 복구 시도...');
    
    // 모든 에러 상태 초기화
    if (window.ErrorHandler) {
        window.ErrorHandler.clearAllErrors();
    }
    
    // 폼 초기화
    const form = document.getElementById('statsForm');
    if (form) {
        form.reset();
    }
    
    // 결과 섹션 숨기기
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
    
    // 로딩 상태 해제
    hideLoadingState();
    
    // 성공 메시지
    if (window.ErrorHandler) {
        window.ErrorHandler.showToast('info', '시스템 초기화', '모든 입력이 초기화되었습니다');
    }
}

// 향상된 초기화 함수
function initializeAppEnhanced() {
    console.log('🛡️ 고급 에러 처리와 함께 PUBG 플레이어 분석 앱 초기화');
    
    // 필수 데이터 검증
    if (!window.CLUSTER_DATA || !window.CLUSTER_DATA.CLUSTER_CENTERS) {
        console.error('필수 클러스터 데이터가 없습니다');
        if (window.ErrorHandler) {
            window.ErrorHandler.handleAnalysisError('DATA_MISSING', null);
        }
        return;
    }
    
    // 폼 이벤트 리스너 등록 (향상된 버전)
    const form = document.getElementById('statsForm');
    if (form) {
        // 기존 이벤트 리스너 제거 후 새로운 것 추가
        const newForm = form.cloneNode(true);
        form.parentNode.replaceChild(newForm, form);
        newForm.addEventListener('submit', handleFormSubmitEnhanced);
    }
    
    // 향상된 입력 검증 리스너 등록
    setupInputValidationEnhanced();
    
    // 성공 알림
    if (window.ErrorHandler) {
        window.ErrorHandler.showToast('success', '시스템 준비 완료', 
            '이제 게임 스탯을 입력하여 분석을 시작할 수 있습니다');
    }
}

// 향상된 입력 검증 설정
function setupInputValidationEnhanced() {
    const inputs = document.querySelectorAll('#statsForm input[type="number"]');
    
    inputs.forEach(input => {
        // 실시간 검증
        input.addEventListener('input', function() {
            clearTimeout(this.validationTimeout);
            this.validationTimeout = setTimeout(() => {
                validateSingleInputEnhanced(this);
            }, 300);
        });
        
        // 포커스 잃을 때 즉시 검증
        input.addEventListener('blur', function() {
            clearTimeout(this.validationTimeout);
            validateSingleInputEnhanced(this);
        });
        
        // 포커스 얻을 때 에러 초기화
        input.addEventListener('focus', function() {
            if (window.ErrorHandler) {
                window.ErrorHandler.clearFieldError(this.id);
            }
        });
    });
    
    console.log('✅ 향상된 실시간 입력 검증 설정 완료');
}

// DOM 로드 시 향상된 초기화 실행
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        initializeAppEnhanced();
    }, 500); // 다른 스크립트들이 로드된 후 실행
});

// 전역 함수로 노출
window.validateInputEnhanced = validateInputEnhanced;
window.handleFormSubmitEnhanced = handleFormSubmitEnhanced;
window.validateSingleInputEnhanced = validateSingleInputEnhanced;
window.recoverFromError = recoverFromError;

console.log('🛡️ 에러 처리 확장 모듈 로드 완료');
