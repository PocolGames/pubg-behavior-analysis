/**
 * PUBG 플레이어 분석 - 고급 에러 처리 모듈
 * 사용자 친화적인 에러 메시지, 토스트 알림, 복구 기능 제공
 */

class ErrorHandler {
    constructor() {
        this.toastContainer = null;
        this.init();
    }

    /**
     * 에러 핸들러 초기화
     */
    init() {
        this.createToastContainer();
        this.setupGlobalErrorHandlers();
        console.log('🛡️ 고급 에러 처리 시스템 초기화 완료');
    }

    /**
     * 토스트 컨테이너 생성
     */
    createToastContainer() {
        if (this.toastContainer) return;

        this.toastContainer = document.createElement('div');
        this.toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        this.toastContainer.style.zIndex = '9999';
        document.body.appendChild(this.toastContainer);
    }

    /**
     * 전역 에러 핸들러 설정
     */
    setupGlobalErrorHandlers() {
        // JavaScript 에러 처리
        window.addEventListener('error', (event) => {
            console.error('전역 JavaScript 에러:', event.error);
            this.showToast('error', '예상치 못한 오류가 발생했습니다', event.error.message);
        });

        // Promise 거부 처리
        window.addEventListener('unhandledrejection', (event) => {
            console.error('처리되지 않은 Promise 거부:', event.reason);
            this.showToast('error', '데이터 처리 중 오류가 발생했습니다', '다시 시도해주세요');
        });

        // 네트워크 상태 변화 감지
        window.addEventListener('online', () => {
            this.showToast('success', '네트워크 연결이 복구되었습니다');
        });

        window.addEventListener('offline', () => {
            this.showToast('warning', '네트워크 연결이 끊어졌습니다', '일부 기능이 제한될 수 있습니다');
        });
    }

    /**
     * 토스트 알림 표시
     * @param {string} type - success, error, warning, info
     * @param {string} title - 제목
     * @param {string} message - 메시지 (선택사항)
     * @param {number} duration - 표시 시간(ms, 기본 5000)
     */
    showToast(type = 'info', title, message = '', duration = 5000) {
        const toastId = 'toast-' + Date.now();
        
        const iconMap = {
            success: 'fas fa-check-circle text-success',
            error: 'fas fa-exclamation-circle text-danger',
            warning: 'fas fa-exclamation-triangle text-warning',
            info: 'fas fa-info-circle text-info'
        };

        const bgMap = {
            success: 'bg-success',
            error: 'bg-danger',
            warning: 'bg-warning',
            info: 'bg-info'
        };

        const toast = document.createElement('div');
        toast.id = toastId;
        toast.className = 'toast align-items-center text-white border-0';
        toast.classList.add(bgMap[type]);
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');

        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <div class="d-flex align-items-center">
                        <i class="${iconMap[type]} me-2"></i>
                        <div>
                            <strong>${title}</strong>
                            ${message ? `<div class="small">${message}</div>` : ''}
                        </div>
                    </div>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;

        this.toastContainer.appendChild(toast);

        // Bootstrap 토스트 초기화 및 표시
        const bsToast = new bootstrap.Toast(toast, {
            delay: duration
        });
        bsToast.show();

        // 토스트 제거 후 DOM에서 삭제
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });

        return toastId;
    }

    /**
     * 입력 검증 에러 처리
     */
    handleValidationError(fieldId, errorType, customMessage = null) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        // 기존 에러 상태 제거
        this.clearFieldError(fieldId);

        // 필드에 에러 스타일 적용
        field.classList.add('is-invalid');

        // 에러 메시지 생성
        let errorMessage = customMessage || this.getValidationErrorMessage(fieldId, errorType);

        // 에러 피드백 요소 생성
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        feedback.textContent = errorMessage;

        // 필드 바로 아래에 삽입
        field.parentNode.appendChild(feedback);

        // 필드로 포커스 이동
        field.focus();

        return errorMessage;
    }

    /**
     * 필드 에러 상태 제거
     */
    clearFieldError(fieldId) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        field.classList.remove('is-invalid', 'is-valid');
        
        const existingFeedback = field.parentNode.querySelector('.invalid-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }
    }

    /**
     * 필드 성공 상태 표시
     */
    showFieldSuccess(fieldId) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        this.clearFieldError(fieldId);
        field.classList.add('is-valid');
    }

    /**
     * 검증 에러 메시지 생성
     */
    getValidationErrorMessage(fieldId, errorType) {
        const fieldNames = {
            'kills': '킬 수',
            'assists': '어시스트',
            'damageDealt': '데미지',
            'headshotKills': '헤드샷 킬',
            'walkDistance': '보행 거리',
            'rideDistance': '차량 이동 거리',
            'swimDistance': '수영 거리',
            'longestKill': '최장 킬 거리',
            'heals': '힐 사용',
            'boosts': '부스트 사용',
            'revives': '팀원 부활',
            'weaponsAcquired': '무기 획득',
            'killPlace': '킬 순위',
            'matchDuration': '게임 시간'
        };

        const fieldName = fieldNames[fieldId] || fieldId;

        const errorMessages = {
            'required': `${fieldName}는 필수 입력 항목입니다`,
            'min': `${fieldName}가 너무 작습니다`,
            'max': `${fieldName}가 너무 큽니다`,
            'range': `${fieldName}가 유효 범위를 벗어났습니다`,
            'logic': `${fieldName}가 논리적으로 맞지 않습니다`,
            'format': `${fieldName} 형식이 올바르지 않습니다`,
            'realistic': `${fieldName}가 비현실적입니다`
        };

        return errorMessages[errorType] || `${fieldName} 입력에 오류가 있습니다`;
    }

    /**
     * 네트워크 에러 처리
     */
    handleNetworkError(operation = '데이터 처리') {
        if (!navigator.onLine) {
            this.showToast('error', '네트워크 연결 오류', 
                '인터넷 연결을 확인하고 다시 시도해주세요');
            return 'offline';
        }

        this.showToast('error', `${operation} 실패`, 
            '서버 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요');
        return 'network';
    }

    /**
     * 분석 실패 에러 처리
     */
    handleAnalysisError(errorCode, userStats) {
        const errorMessages = {
            'INVALID_INPUT': {
                title: '입력 데이터 오류',
                message: '입력한 데이터에 문제가 있습니다. 모든 값을 다시 확인해주세요.'
            },
            'CALCULATION_FAILED': {
                title: '계산 처리 오류',
                message: '특성 계산 중 오류가 발생했습니다. 입력값을 확인해주세요.'
            },
            'PREDICTION_FAILED': {
                title: '예측 모델 오류',
                message: '플레이어 유형 예측에 실패했습니다. 다시 시도해주세요.'
            },
            'DATA_MISSING': {
                title: '데이터 로딩 오류',
                message: '분석에 필요한 데이터를 불러올 수 없습니다.'
            }
        };

        const error = errorMessages[errorCode] || {
            title: '알 수 없는 오류',
            message: '예상치 못한 오류가 발생했습니다.'
        };

        this.showToast('error', error.title, error.message);

        // 상세 로그 기록
        console.error('분석 에러:', {
            code: errorCode,
            userStats: userStats,
            timestamp: new Date().toISOString()
        });

        return error;
    }

    /**
     * 복구 가능한 에러 처리
     */
    handleRecoverableError(errorType, recoveryCallback) {
        const toastId = this.showToast('warning', '일시적 오류 발생', 
            '자동으로 복구를 시도합니다...', 3000);

        setTimeout(() => {
            if (typeof recoveryCallback === 'function') {
                try {
                    recoveryCallback();
                    this.showToast('success', '복구 완료', '작업이 성공적으로 재시도되었습니다');
                } catch (retryError) {
                    this.showToast('error', '복구 실패', '수동으로 다시 시도해주세요');
                }
            }
        }, 1000);
    }

    /**
     * 진행 상황 에러 처리
     */
    handleProgressError(step, totalSteps, errorMessage) {
        this.showToast('error', `${step}/${totalSteps} 단계에서 오류`, errorMessage);
    }

    /**
     * 사용자 액션 에러 처리
     */
    handleUserActionError(action, suggestion) {
        this.showToast('warning', `${action} 실패`, suggestion);
    }

    /**
     * 모든 에러 상태 초기화
     */
    clearAllErrors() {
        // 모든 필드 에러 제거
        const invalidFields = document.querySelectorAll('.is-invalid');
        invalidFields.forEach(field => {
            this.clearFieldError(field.id);
        });

        // 기존 에러 메시지 제거
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(msg => msg.remove());

        // 토스트 제거
        const activeToasts = document.querySelectorAll('.toast.show');
        activeToasts.forEach(toast => {
            const bsToast = bootstrap.Toast.getInstance(toast);
            if (bsToast) bsToast.hide();
        });
    }

    /**
     * 에러 통계 수집
     */
    logError(category, error, userAction = null) {
        const errorLog = {
            category: category,
            error: error,
            userAction: userAction,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        console.group('🔍 에러 로그');
        console.error('카테고리:', category);
        console.error('에러:', error);
        console.error('사용자 액션:', userAction);
        console.error('상세 정보:', errorLog);
        console.groupEnd();

        // 실제 서비스에서는 여기서 에러 추적 서비스로 전송
        // 예: analytics.track('error', errorLog);
    }
}

// 전역 에러 핸들러 인스턴스 생성
window.ErrorHandler = new ErrorHandler();

// 기존 script.js와의 호환성을 위한 래퍼 함수들
window.showErrorToast = (title, message) => window.ErrorHandler.showToast('error', title, message);
window.showSuccessToast = (title, message) => window.ErrorHandler.showToast('success', title, message);
window.showWarningToast = (title, message) => window.ErrorHandler.showToast('warning', title, message);
window.clearAllErrors = () => window.ErrorHandler.clearAllErrors();
