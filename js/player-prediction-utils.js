/**
 * PUBG Player Prediction System - Utils Module
 * 플레이어 예측 시스템 유틸리티 모듈
 * 초기화, 헬퍼 함수, 전역 관리 담당
 */

/**
 * 전역 변수 및 초기화 관련 코드
 */
let playerPredictor;
let predictionConfig = null;

/**
 * 페이지 로드 시 초기화
 */
document.addEventListener('DOMContentLoaded', function() {
    // Player Prediction 페이지에서만 실행
    if (window.location.pathname.includes('player-prediction.html') && 
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
        // 설정 데이터 로드
        await loadPredictionConfig();
        
        // Chart.js 로딩 확인
        if (!window.ChartUtils) {
            console.warn('⚠️ ChartUtils가 로드되지 않았습니다. 차트 기능이 제한될 수 있습니다.');
        }
        
        // PlayerPredictor 인스턴스 생성
        playerPredictor = new PlayerPredictor();
        
        // 추가 초기화 작업
        setupAdvancedFeatures();
        
        // JSON 기반 UI 초기화
        setupJSONBasedFeatures();
        
        console.log('✅ 플레이어 예측 시스템 초기화 완료');
        
        // 전역 접근 설정
        window.playerPredictor = playerPredictor;
        
    } catch (error) {
        console.error('❌ 플레이어 예측 시스템 초기화 실패:', error);
        showInitializationError(error);
    }
}

/**
 * 예측 설정 로드
 */
async function loadPredictionConfig() {
    try {
        // 현재 경로에 따라 적절한 데이터 경로 결정
        let dataPath = './data/player-prediction.json';
        
        // pages 폴더에서 실행되는 경우 상위 디렉토리의 data 폴더 참조
        if (window.location.pathname.includes('/pages/')) {
            dataPath = '../data/player-prediction.json';
        }
        
        const response = await fetch(dataPath);
        if (response.ok) {
            predictionConfig = await response.json();
            console.log('✅ 예측 설정 로드 완료:', predictionConfig.metadata);
        } else {
            throw new Error('설정 파일 로드 실패');
        }
    } catch (error) {
        console.warn('⚠️ 설정 파일 로드 실패, 기본 설정 사용:', error);
        predictionConfig = null;
    }
}

/**
 * JSON 기반 기능 설정
 */
function setupJSONBasedFeatures() {
    if (!predictionConfig) return;
    
    // 동적 툴팁 설정
    setupDynamicTooltips();
    
    // 검증 규칙 적용
    setupValidationRules();
    
    // 메타데이터 표시
    displayMetadata();
    
    // 샘플 플레이어 버튼 업데이트
    updateSamplePlayerButtons();
}

/**
 * 동적 툴팁 설정 (JSON 기반)
 */
function setupDynamicTooltips() {
    if (!predictionConfig.featureDefinitions) return;
    
    predictionConfig.featureDefinitions.forEach(feature => {
        const input = document.querySelector(`input[name="${feature.name}"]`);
        if (input) {
            const tooltipText = `${feature.description} (${feature.min}-${feature.max}${feature.unit})`;
            input.title = tooltipText;
            input.setAttribute('data-tooltip', tooltipText);
            
            // 카테고리 정보 추가
            const formGroup = input.closest('.form-group');
            if (formGroup) {
                formGroup.setAttribute('data-category', feature.category);
            }
        }
    });
}

/**
 * 검증 규칙 설정 (JSON 기반)
 */
function setupValidationRules() {
    if (!predictionConfig.validationRules) return;
    
    const form = document.getElementById('playerForm');
    if (!form) return;
    
    // 논리적 검증 규칙 UI 표시
    const rulesContainer = document.getElementById('validationRules');
    if (rulesContainer && predictionConfig.validationRules.logicalChecks) {
        rulesContainer.innerHTML = `
            <h5>게임 로직 검증 규칙:</h5>
            <ul>
                ${predictionConfig.validationRules.logicalChecks.map(rule => 
                    `<li>${rule}</li>`
                ).join('')}
            </ul>
        `;
    }
}

/**
 * 메타데이터 표시
 */
function displayMetadata() {
    if (!predictionConfig.metadata) return;
    
    const metadata = predictionConfig.metadata;
    
    // 페이지 제목 업데이트
    const pageTitle = document.querySelector('.page-title');
    if (pageTitle) {
        pageTitle.innerHTML += ` <small>v${metadata.version}</small>`;
    }
    
    // 모델 정보 카드 업데이트
    const modelInfoElements = {
        '.model-name': metadata.modelName,
        '.model-accuracy': `${(metadata.accuracy * 100).toFixed(2)}%`,
        '.model-features': `${metadata.features}개 특성`,
        '.model-classes': `${metadata.classes}개 유형`,
        '.model-version': `v${metadata.version}`,
        '.model-created': metadata.created
    };
    
    Object.entries(modelInfoElements).forEach(([selector, value]) => {
        const element = document.querySelector(selector);
        if (element) {
            element.textContent = value;
        }
    });
}

/**
 * 샘플 플레이어 버튼 업데이트
 */
function updateSamplePlayerButtons() {
    if (!predictionConfig.samplePlayers) return;
    
    Object.entries(predictionConfig.samplePlayers).forEach(([key, sample]) => {
        const button = document.querySelector(`[data-type="${key}"] button`);
        const card = document.querySelector(`[data-type="${key}"]`);
        
        if (button && sample.name) {
            button.textContent = sample.name;
        }
        
        if (card && sample.description) {
            const description = card.querySelector('.sample-description');
            if (description) {
                description.textContent = sample.description;
            }
        }
    });
}

/**
 * 초기화 오류 표시
 */
function showInitializationError(error) {
    const errorContainer = document.getElementById('initializationError');
    if (errorContainer) {
        errorContainer.innerHTML = `
            <div class="alert alert-error">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>초기화 오류:</strong> ${error.message}
                <br><small>페이지를 새로고침하거나 관리자에게 문의하세요.</small>
            </div>
        `;
        errorContainer.style.display = 'block';
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
    
    // 접근성 기능
    setupAccessibilityFeatures();
}

/**
 * 키보드 단축키 설정 (확장된 버전)
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl + Enter: 예측 실행
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            if (playerPredictor) {
                playerPredictor.predictPlayerType();
            }
            showShortcutNotification('예측 실행');
        }
        
        // Ctrl + R: 폼 초기화
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            if (playerPredictor) {
                playerPredictor.resetForm();
            }
            showShortcutNotification('폼 초기화');
        }
        
        // Escape: 결과 숨기기
        if (e.key === 'Escape') {
            if (playerPredictor) {
                playerPredictor.hidePredictionResults();
            }
            showShortcutNotification('결과 숨기기');
        }
        
        // Ctrl + S: 결과 저장
        if (e.ctrlKey && e.key === 's' && playerPredictor && playerPredictor.currentPrediction) {
            e.preventDefault();
            ExportUtils.exportPredictionAsJSON(
                playerPredictor.currentPrediction,
                playerPredictor.getFormData()
            );
            showShortcutNotification('결과 저장');
        }
        
        // F1: 도움말
        if (e.key === 'F1') {
            e.preventDefault();
            showHelpModal();
        }
    });
}

/**
 * 단축키 알림 표시
 */
function showShortcutNotification(action) {
    if (window.App && window.App.showNotification) {
        window.App.showNotification(`키보드 단축키: ${action}`, 'info');
    }
}

/**
 * 도움말 모달 표시
 */
function showHelpModal() {
    const shortcuts = [
        { key: 'Ctrl + Enter', action: '예측 실행' },
        { key: 'Ctrl + R', action: '폼 초기화' },
        { key: 'Escape', action: '결과 숨기기' },
        { key: 'Ctrl + S', action: '결과 저장' },
        { key: 'F1', action: '도움말 표시' }
    ];
    
    const helpContent = `
        <div class="help-modal">
            <h3>키보드 단축키</h3>
            <table>
                ${shortcuts.map(s => 
                    `<tr><td><kbd>${s.key}</kbd></td><td>${s.action}</td></tr>`
                ).join('')}
            </table>
        </div>
    `;
    
    if (window.App && window.App.showModal) {
        window.App.showModal('도움말', helpContent);
    } else {
        alert('도움말: F1 키를 눌러 단축키를 확인할 수 있습니다.');
    }
}

/**
 * 자동 저장 기능 (JSON 설정 기반)
 */
function setupAutoSave() {
    const form = document.getElementById('playerForm');
    if (!form) return;
    
    // 사용자 설정 확인
    const preferences = PredictionUtils.loadUserPreferences();
    const autoSaveEnabled = preferences.autoSave !== false; // 기본값: true
    
    if (!autoSaveEnabled) return;
    
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        // 저장된 값 복원
        const savedValue = localStorage.getItem(`pubg_prediction_${input.name}`);
        if (savedValue !== null && input.value === '') {
            input.value = savedValue;
        }
        
        // 값 변경 시 자동 저장 (디바운스 적용)
        let saveTimeout;
        input.addEventListener('input', function() {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                localStorage.setItem(`pubg_prediction_${input.name}`, input.value);
            }, 500);
        });
    });
    
    // 자동 저장 상태 표시
    const statusEl = document.getElementById('autoSaveStatus');
    if (statusEl) {
        statusEl.textContent = '자동 저장 활성화';
        statusEl.className = 'auto-save-status active';
    }
}

/**
 * 도움말 툴팁 설정
 */
function setupHelpTooltips() {
    // JSON 데이터가 있으면 해당 함수에서 처리됨
    if (predictionConfig) return;
    
    // 백업 툴팁 (JSON 로드 실패 시)
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
        const input = document.querySelector(`input[name="${id}"]`);
        if (input) {
            input.title = text;
            input.setAttribute('data-tooltip', text);
        }
    });
}

/**
 * 접근성 기능 설정
 */
function setupAccessibilityFeatures() {
    // 포커스 관리
    const form = document.getElementById('playerForm');
    if (form) {
        const inputs = form.querySelectorAll('input[type="number"]');
        
        inputs.forEach((input, index) => {
            // 탭 순서 설정
            input.tabIndex = index + 1;
            
            // ARIA 레이블 설정
            const label = input.closest('.form-group')?.querySelector('label');
            if (label) {
                input.setAttribute('aria-labelledby', label.id || `label-${input.name}`);
                if (!label.id) {
                    label.id = `label-${input.name}`;
                }
            }
        });
    }
    
    // 스크린 리더를 위한 상태 알림
    const statusRegion = document.getElementById('srOnlyStatus');
    if (statusRegion) {
        statusRegion.setAttribute('aria-live', 'polite');
        statusRegion.setAttribute('aria-atomic', 'true');
    }
}

/**
 * 성능 모니터링 (향상된 버전)
 */
function setupPerformanceMonitoring() {
    let predictionCount = 0;
    let totalPredictionTime = 0;
    let errorCount = 0;
    
    // 예측 성능 추적
    const originalPredict = PlayerPredictor.prototype.predictPlayerType;
    PlayerPredictor.prototype.predictPlayerType = async function() {
        const startTime = performance.now();
        
        try {
            await originalPredict.call(this);
            predictionCount++;
            const endTime = performance.now();
            const predictionTime = endTime - startTime;
            totalPredictionTime += predictionTime;
            
            const averageTime = totalPredictionTime / predictionCount;
            
            // 성능 로그
            console.log(`📊 예측 성능 통계:`, {
                총_예측_수: predictionCount,
                평균_시간: `${averageTime.toFixed(2)}ms`,
                이번_예측: `${predictionTime.toFixed(2)}ms`,
                오류_수: errorCount,
                성공률: `${((predictionCount / (predictionCount + errorCount)) * 100).toFixed(1)}%`
            });
            
            // 성능 경고
            if (averageTime > 2000) {
                console.warn('⚠️ 예측 성능이 저하되었습니다. 평균 시간:', averageTime.toFixed(2) + 'ms');
            }
            
        } catch (error) {
            errorCount++;
            console.error('❌ 예측 중 오류 발생:', error);
            throw error;
        }
    };
}

/**
 * 유틸리티 함수들 (JSON 데이터 기반 확장)
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
     * 플레이어 특성 검증 (JSON 기반)
     */
    validatePlayerStats(stats) {
        if (predictionConfig && predictionConfig.featureDefinitions) {
            // JSON 기반 검증
            for (const feature of predictionConfig.featureDefinitions) {
                const value = stats[feature.name];
                if (value !== undefined) {
                    if (value < feature.min || value > feature.max) {
                        return {
                            valid: false,
                            field: feature.name,
                            message: `${feature.displayName}는 ${feature.min}-${feature.max}${feature.unit} 범위여야 합니다.`
                        };
                    }
                }
            }
        } else {
            // 백업 검증 규칙
            const rules = {
                kills: { min: 0, max: 50 },
                damageDealt: { min: 0, max: 2000 },
                walkDistance: { min: 0, max: 10000 },
                heals: { min: 0, max: 30 },
                assists: { min: 0, max: 20 }
            };

            for (const [key, rule] of Object.entries(rules)) {
                if (stats[key] < rule.min || stats[key] > rule.max) {
                    return {
                        valid: false,
                        field: key,
                        message: `${key}는 ${rule.min}-${rule.max} 범위여야 합니다.`
                    };
                }
            }
        }

        return { valid: true };
    },

    /**
     * 특성 중요도 가져오기
     */
    getFeatureImportance(featureName) {
        if (predictionConfig && predictionConfig.featureDefinitions) {
            const feature = predictionConfig.featureDefinitions.find(f => f.name === featureName);
            return feature ? feature.weight : 0;
        }
        return 0;
    },

    /**
     * 색상 유틸리티 (JSON 기반)
     */
    getClusterColor(clusterId, alpha = 1) {
        const colors = [
            '#56ab2f', '#4CAF50', '#667eea', '#5A67D8',
            '#4299E1', '#3182CE', '#2B6CB0', '#dc3545'
        ];
        
        const baseColor = colors[clusterId] || '#666666';
        if (alpha < 1) {
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
        const merged = { ...this.loadUserPreferences(), ...preferences };
        localStorage.setItem('pubg_prediction_preferences', JSON.stringify(merged));
    },

    loadUserPreferences() {
        const saved = localStorage.getItem('pubg_prediction_preferences');
        return saved ? JSON.parse(saved) : {
            autoSave: true,
            showTooltips: true,
            theme: 'dark',
            language: 'ko'
        };
    },

    clearSavedData() {
        const keys = Object.keys(localStorage).filter(key => key.startsWith('pubg_prediction_'));
        keys.forEach(key => localStorage.removeItem(key));
        
        if (window.App && window.App.showNotification) {
            window.App.showNotification('저장된 데이터가 모두 삭제되었습니다.', 'info');
        }
    },

    /**
     * 설정 정보 가져오기
     */
    getConfig() {
        return predictionConfig;
    }
};

/**
 * 데이터 내보내기 기능 (JSON 기반 확장)
 */
const ExportUtils = {
    /**
     * 예측 결과를 JSON으로 내보내기 (메타데이터 포함)
     */
    exportPredictionAsJSON(prediction, formData) {
        const exportData = {
            timestamp: new Date().toISOString(),
            metadata: predictionConfig ? predictionConfig.metadata : null,
            formData: formData,
            prediction: prediction,
            userAgent: navigator.userAgent,
            version: '2.0'
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        this.downloadFile(dataStr, `prediction-result-${this.getTimestamp()}.json`, 'application/json');
    },

    /**
     * 예측 결과를 CSV로 내보내기 (확장된 버전)
     */
    exportPredictionAsCSV(prediction, formData) {
        let csvContent = '';
        
        // 헤더
        csvContent += '플레이어 예측 결과 보고서\n';
        csvContent += `생성 시간: ${new Date().toISOString()}\n`;
        csvContent += `모델: ${predictionConfig ? predictionConfig.metadata.modelName : 'Unknown'}\n\n`;
        
        // 입력 데이터
        csvContent += '입력 특성,값\n';
        Object.entries(formData).forEach(([key, value]) => {
            const feature = predictionConfig ? 
                predictionConfig.featureDefinitions.find(f => f.name === key) : null;
            const displayName = feature ? feature.displayName : key;
            const unit = feature ? feature.unit : '';
            csvContent += `${displayName},${value}${unit}\n`;
        });

        // 예측 결과
        csvContent += '\n예측 결과\n';
        csvContent += `예측된 클러스터,${prediction.predictedCluster}\n`;
        csvContent += `신뢰도,${prediction.confidence.toFixed(2)}%\n`;
        
        // 클러스터별 확률
        if (prediction.probabilities) {
            csvContent += '\n클러스터별 확률\n';
            prediction.probabilities.forEach((prob, index) => {
                csvContent += `클러스터 ${index},${prob.toFixed(2)}%\n`;
            });
        }

        this.downloadFile(csvContent, `prediction-result-${this.getTimestamp()}.csv`, 'text/csv');
    },

    /**
     * 타임스탬프 생성
     */
    getTimestamp() {
        const now = new Date();
        return now.toISOString().replace(/[:.]/g, '-').split('T')[0] + '_' + 
               now.toTimeString().split(' ')[0].replace(/:/g, '');
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
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.URL.revokeObjectURL(url);
        
        // 다운로드 알림
        if (window.App && window.App.showNotification) {
            window.App.showNotification(`파일이 다운로드되었습니다: ${filename}`, 'success');
        }
    }
};

// 전역 접근을 위한 export
if (typeof window !== 'undefined') {
    window.PredictionUtils = PredictionUtils;
    window.ExportUtils = ExportUtils;
    window.playerPredictor = null; // 초기화 후 설정됨
    window.predictionConfig = null; // 설정 로드 후 설정됨
}
