// UI 관리 클래스
class UIManager {
    constructor() {
        this.form = document.getElementById('classifierForm');
        this.resultPanel = document.getElementById('resultPanel');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.resultContent = document.getElementById('resultContent');
        this.classifyBtn = document.getElementById('classifyBtn');
        
        this.initializeEventListeners();
        this.initializeSampleButtons();
    }

    // 이벤트 리스너 초기화
    initializeEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        
        // 입력 필드 실시간 검증
        const inputs = this.form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('input', () => this.validateInput(input));
        });
    }

    // 샘플 버튼 초기화
    initializeSampleButtons() {
        const sampleButtonsHtml = `
            <div class="sample-buttons">
                <h3>샘플 데이터로 테스트</h3>
                <div class="sample-btn-group">
                    <button type="button" class="sample-btn" data-type="aggressive">
                        ⚔️ 공격형
                    </button>
                    <button type="button" class="sample-btn" data-type="survivor">
                        🛡️ 생존형
                    </button>
                    <button type="button" class="sample-btn" data-type="explorer">
                        🗺️ 탐험형
                    </button>
                </div>
            </div>
        `;
        
        this.form.insertAdjacentHTML('beforeend', sampleButtonsHtml);
        
        // 샘플 버튼 이벤트 리스너
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.loadSampleData(e.target.dataset.type));
        });
    }

    // 입력 검증
    validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        if (value < min || value > max) {
            input.style.borderColor = 'var(--danger-color)';
        } else {
            input.style.borderColor = 'var(--border-color)';
        }
    }

    // 샘플 데이터 로드
    loadSampleData(type) {
        const sampleData = SAMPLE_PROFILES[type];
        if (!sampleData) return;

        Object.entries(sampleData).forEach(([key, value]) => {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;
                this.validateInput(input);
            }
        });
    }

    // 폼 데이터 수집
    collectFormData() {
        const formData = new FormData(this.form);
        const data = {};
        
        for (const [key, value] of formData.entries()) {
            data[key] = parseFloat(value) || 0;
        }
        
        return data;
    }

    // 폼 제출 처리
    async handleFormSubmit(e) {
        e.preventDefault();
        
        const inputData = this.collectFormData();
        
        // 로딩 상태 표시
        this.showLoading();
        
        try {
            // 분류 실행 (약간의 지연으로 로딩 효과)
            await new Promise(resolve => setTimeout(resolve, 1500));
            const result = classifier.classify(inputData);
            
            // 결과 표시
            this.displayResult(inputData, result);
        } catch (error) {
            console.error('Classification error:', error);
            this.showError('분류 중 오류가 발생했습니다.');
        }
    }

    // 로딩 상태 표시
    showLoading() {
        this.classifyBtn.disabled = true;
        this.loadingIndicator.classList.add('active');
        this.resultContent.classList.remove('active');
        
        // 스피너 회전 애니메이션
        const spinner = this.loadingIndicator.querySelector('.loading-spinner');
        spinner.style.animation = 'spin 1s linear infinite';
    }

    // 결과 표시
    displayResult(inputData, result) {
        // 로딩 숨기기
        this.loadingIndicator.classList.remove('active');
        this.classifyBtn.disabled = false;
        
        // 결과 컨텐츠 업데이트
        this.updatePlayerTypeCard(result);
        this.updateProbabilityChart(result.probabilities);
        this.updateInsights(inputData, result);
        
        // 결과 패널 표시
        this.resultContent.classList.add('active');
        
        // 결과 패널로 스크롤
        this.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // 플레이어 유형 카드 업데이트
    updatePlayerTypeCard(result) {
        const typeData = PLAYER_TYPES[result.predictedType];
        const card = document.getElementById('playerTypeCard');
        
        // 유형별 클래스 적용
        card.className = `player-type-card type-${result.predictedType}`;
        
        // 아이콘 업데이트
        document.getElementById('typeIcon').textContent = typeData.icon;
        
        // 이름 및 설명 업데이트
        document.getElementById('typeName').textContent = typeData.name;
        document.getElementById('typeDescription').textContent = typeData.description;
        
        // 신뢰도 업데이트
        const confidenceValue = document.getElementById('confidenceValue');
        const confidencePercent = Math.round(result.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // 신뢰도에 따른 색상 변경
        if (confidencePercent >= 80) {
            confidenceValue.className = 'confidence-value text-success';
        } else if (confidencePercent >= 60) {
            confidenceValue.className = 'confidence-value text-warning';
        } else {
            confidenceValue.className = 'confidence-value text-danger';
        }
    }

    // 확률 차트 업데이트
    updateProbabilityChart(probabilities) {
        const chartBars = document.getElementById('chartBars');
        chartBars.innerHTML = '';
        
        // 확률 내림차순 정렬
        const sortedProbs = Object.entries(probabilities)
            .sort(([,a], [,b]) => b - a);
        
        sortedProbs.forEach(([type, probability]) => {
            const typeData = PLAYER_TYPES[type];
            const percentage = Math.round(probability * 100);
            
            const barHtml = `
                <div class="chart-bar">
                    <div class="bar-label">${typeData.icon} ${typeData.name}</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${percentage}%; background-color: ${typeData.color};">
                            <span class="bar-value">${percentage}%</span>
                        </div>
                    </div>
                </div>
            `;
            
            chartBars.insertAdjacentHTML('beforeend', barHtml);
        });
    }

    // 인사이트 업데이트
    updateInsights(inputData, result) {
        const insights = classifier.generateInsights(inputData, result);
        const insightsContent = document.getElementById('insightsContent');
        insightsContent.innerHTML = '';
        
        if (insights.length === 0) {
            insightsContent.innerHTML = '<p class="text-muted">특별한 특성이 발견되지 않았습니다.</p>';
            return;
        }
        
        insights.forEach(insight => {
            const insightHtml = `
                <div class="insight-item">
                    <div class="insight-icon">${insight.icon}</div>
                    <div class="insight-text">${insight.text}</div>
                </div>
            `;
            
            insightsContent.insertAdjacentHTML('beforeend', insightHtml);
        });
        
        // 이상치 표시
        if (result.isAnomaly) {
            const anomalyHtml = `
                <div class="insight-item">
                    <div class="insight-icon">🚨</div>
                    <div class="insight-text">
                        <strong>이상치 감지:</strong> 일반적이지 않은 극단적인 수치가 발견되었습니다.
                    </div>
                </div>
            `;
            
            insightsContent.insertAdjacentHTML('beforeend', anomalyHtml);
        }
    }

    // 오류 표시
    showError(message) {
        this.loadingIndicator.classList.remove('active');
        this.classifyBtn.disabled = false;
        
        const errorHtml = `
            <div class="error-message">
                <h3>오류 발생</h3>
                <p>${message}</p>
                <button type="button" onclick="location.reload()">다시 시도</button>
            </div>
        `;
        
        this.resultContent.innerHTML = errorHtml;
        this.resultContent.classList.add('active');
    }
}

// CSS 애니메이션 추가
const additionalStyles = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .sample-buttons {
        margin-top: 24px;
        padding-top: 24px;
        border-top: 1px solid var(--border-color);
    }
    
    .sample-buttons h3 {
        color: var(--text-primary);
        font-size: 1.1rem;
        margin-bottom: 12px;
    }
    
    .sample-btn-group {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    
    .sample-btn {
        padding: 8px 16px;
        background: var(--bg-secondary);
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    .sample-btn:hover {
        background: var(--primary-color);
        color: var(--text-primary);
        border-color: var(--primary-color);
    }
    
    .error-message {
        text-align: center;
        padding: 40px;
        color: var(--danger-color);
    }
    
    .error-message button {
        margin-top: 16px;
        padding: 10px 20px;
        background: var(--primary-color);
        color: var(--text-primary);
        border: none;
        border-radius: 6px;
        cursor: pointer;
    }
`;

// 스타일 주입
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// DOM 로드 완료 후 UI 초기화
document.addEventListener('DOMContentLoaded', () => {
    const uiManager = new UIManager();
    
    // 개발자 도구용 전역 변수
    window.classifier = classifier;
    window.uiManager = uiManager;
    
    console.log('🎮 PUBG Player Classifier 초기화 완료');
    console.log('샘플 데이터로 테스트해보세요!');
});