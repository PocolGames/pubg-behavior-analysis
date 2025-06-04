/**
 * PUBG Player Prediction System - Utils Module
 * 플레이어 예측 시스템 유틸리티 모듈
 * 초기화, 헬퍼 함수, 전역 관리 담당
 */

/**
 * 전역 변수 및 초기화 관련 코드
 */
let playerPredictor;

/**
 * 페이지 로드 시 초기화
 */
document.addEventListener('DOMContentLoaded', function() {
    // Player Prediction 페이지에서만 실행
    if (window.location.pathname.includes('player-prediction.html') || 
        document.getElementById('playerForm')) {
        
        console.log('🎮 플레이어 예측 페이지 감지됨');
        initializePlayerPrediction();
        
        // 페이지 언로드 시 정리
        window.addEventListener('beforeunload', () => {
            if (playerPredictor) {
                playerPredictor.destroy();
            }
        });
    }
});

/**
 * 플레이어 예측 시스템 초기화
 */
async function initializePlayerPrediction() {
    try {
        // Chart.js 로딩 확인
        if (!window.ChartUtils) {
            console.warn('⚠️ ChartUtils가 로드되지 않았습니다. 차트 기능이 제한될 수 있습니다.');
        }
        
        // PlayerPredictor 인스턴스 생성
        playerPredictor = new PlayerPredictor();
        
        // 추가 초기화 작업
        setupAdvancedFeatures();
        
        console.log('✅ 플레이어 예측 시스템 초기화 완료');
    } catch (error) {
        console.error('❌ 플레이어 예측 시스템 초기화 실패:', error);
    }
}

/**
 * 고급 기능 설정
 */
function setupAdvancedFeatures() {
    // 키보드 단축키 설정
    setupKeyboardShortcuts();
    
    // 자동 저장 기능
    setupAutoSave();
    
    // 도움말 툴팁
    setupHelpTooltips();
    
    // 성능 모니터링
    setupPerformanceMonitoring();
}

/**
 * 키보드 단축키 설정
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl + Enter: 예측 실행
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            if (playerPredictor) {
                playerPredictor.predictPlayerType();
            }
        }
        
        // Ctrl + R: 폼 초기화
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            if (playerPredictor) {
                playerPredictor.resetForm();
            }
        }
        
        // Escape: 결과 숨기기
        if (e.key === 'Escape') {
            if (playerPredictor) {
                playerPredictor.hidePredictionResults();
            }
        }
    });
}

/**
 * 자동 저장 기능
 */
function setupAutoSave() {
    const form = document.getElementById('playerForm');
    if (!form) return;
    
    // 입력값 자동 저장 (로컬 스토리지)
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        // 저장된 값 복원
        const savedValue = localStorage.getItem(`pubg_prediction_${input.id}`);
        if (savedValue !== null) {
            input.value = savedValue;
        }
        
        // 값 변경 시 자동 저장
        input.addEventListener('input', function() {
            localStorage.setItem(`pubg_prediction_${input.id}`, input.value);
        });
    });
}

/**
 * 도움말 툴팁 설정
 */
function setupHelpTooltips() {
    const tooltips = {
        'kills': '게임에서 처치한 적 플레이어 수',
        'damageDealt': '적에게 가한 총 데미지',
        'walkDistance': '도보로 이동한 거리 (미터)',
        'heals': '사용한 치료 아이템 개수',
        'assists': '팀원을 도와 처치한 횟수',
        'weaponsAcquired': '획득한 무기 개수',
        'killPlace': '킬 수 기준 순위 (낮을수록 좋음)'
    };
    
    Object.entries(tooltips).forEach(([id, text]) => {
        const input = document.getElementById(id);
        if (input) {
            input.title = text;
            input.setAttribute('data-tooltip', text);
        }
    });
}

/**
 * 성능 모니터링
 */
function setupPerformanceMonitoring() {
    let predictionCount = 0;
    let totalPredictionTime = 0;
    
    // 예측 성능 추적
    const originalPredict = PlayerPredictor.prototype.predictPlayerType;
    PlayerPredictor.prototype.predictPlayerType = async function() {
        const startTime = performance.now();
        
        try {
            await originalPredict.call(this);
            predictionCount++;
            const endTime = performance.now();
            totalPredictionTime += (endTime - startTime);
            
            const averageTime = totalPredictionTime / predictionCount;
            console.log(`📊 예측 성능: ${predictionCount}회, 평균 ${averageTime.toFixed(2)}ms`);
            
        } catch (error) {
            console.error('성능 모니터링 중 오류:', error);
        }
    };
}

/**
 * 유틸리티 함수들
 */
const PredictionUtils = {
    /**
     * 숫자 포맷팅
     */
    formatNumber(num, decimals = 1) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(decimals) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(decimals) + 'K';
        }
        return num.toFixed(decimals);
    },

    /**
     * 퍼센트 포맷팅
     */
    formatPercent(value, decimals = 1) {
        return value.toFixed(decimals) + '%';
    },

    /**
     * 플레이어 특성 검증
     */
    validatePlayerStats(stats) {
        const rules = {
            kills: { min: 0, max: 50 },
            damageDealt: { min: 0, max: 2000 },
            walkDistance: { min: 0, max: 10000 },
            heals: { min: 0, max: 30 },
            assists: { min: 0, max: 20 }
        };

        for (const [key, rule] of Object.entries(rules)) {
            if (stats[key] < rule.min || stats[key] > rule.max) {
                return { valid: false, field: key, message: `${key}는 ${rule.min}-${rule.max} 범위여야 합니다.` };
            }
        }

        return { valid: true };
    },

    /**
     * 색상 유틸리티
     */
    getClusterColor(clusterId, alpha = 1) {
        const colors = [
            '#56ab2f', '#4CAF50', '#667eea', '#5A67D8',
            '#4299E1', '#3182CE', '#2B6CB0', '#dc3545'
        ];
        
        const baseColor = colors[clusterId] || '#666666';
        if (alpha < 1) {
            // RGB로 변환 후 알파 추가
            const hex = baseColor.replace('#', '');
            const r = parseInt(hex.substr(0, 2), 16);
            const g = parseInt(hex.substr(2, 2), 16);
            const b = parseInt(hex.substr(4, 2), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        
        return baseColor;
    },

    /**
     * 로컬 스토리지 관리
     */
    saveUserPreferences(preferences) {
        localStorage.setItem('pubg_prediction_preferences', JSON.stringify(preferences));
    },

    loadUserPreferences() {
        const saved = localStorage.getItem('pubg_prediction_preferences');
        return saved ? JSON.parse(saved) : {};
    },

    clearSavedData() {
        const keys = Object.keys(localStorage).filter(key => key.startsWith('pubg_prediction_'));
        keys.forEach(key => localStorage.removeItem(key));
    }
};

/**
 * 데이터 내보내기 기능
 */
const ExportUtils = {
    /**
     * 예측 결과를 JSON으로 내보내기
     */
    exportPredictionAsJSON(prediction, formData) {
        const exportData = {
            timestamp: new Date().toISOString(),
            formData: formData,
            prediction: prediction,
            metadata: {
                version: '1.0',
                model: 'PUBG Player Type Classifier'
            }
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        this.downloadFile(dataStr, 'prediction-result.json', 'application/json');
    },

    /**
     * 예측 결과를 CSV로 내보내기
     */
    exportPredictionAsCSV(prediction, formData) {
        const headers = ['특성', '값'];
        let csvContent = headers.join(',') + '\n';

        // 폼 데이터 추가
        Object.entries(formData).forEach(([key, value]) => {
            csvContent += `${key},${value}\n`;
        });

        // 예측 결과 추가
        csvContent += '\n예측 결과\n';
        csvContent += `예측된 클러스터,${prediction.predictedCluster}\n`;
        csvContent += `신뢰도,${prediction.confidence.toFixed(2)}%\n`;

        this.downloadFile(csvContent, 'prediction-result.csv', 'text/csv');
    },

    /**
     * 파일 다운로드 헬퍼
     */
    downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = window.URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.URL.revokeObjectURL(url);
    }
};

// 전역 접근을 위한 export
if (typeof window !== 'undefined') {
    window.PredictionUtils = PredictionUtils;
    window.ExportUtils = ExportUtils;
    window.playerPredictor = null; // 초기화 후 설정됨
}
