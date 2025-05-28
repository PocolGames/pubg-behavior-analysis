// 전역 변수
let globalCharts = {};
let globalModelData = {};

// DOM 로드 완료 시 실행
document.addEventListener('DOMContentLoaded', async function() {
    console.log('Dashboard initializing...');
    
    try {
        // 로딩 상태 표시
        showLoadingState();
        
        // 모델 데이터 가져오기
        const modelData = await fetchModelData();
        globalModelData = modelData;
        
        // 메트릭 업데이트
        updateMetrics(modelData);
        
        // 차트 초기화
        globalCharts = initializeCharts(modelData);
        
        if (globalCharts) {
            // 리사이즈 핸들러 등록
            handleChartResize(globalCharts);
            
            console.log('Dashboard initialized successfully');
            hideLoadingState();
        } else {
            throw new Error('Failed to initialize charts');
        }
        
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showErrorState(error.message);
    }
});

// 로딩 상태 표시
function showLoadingState() {
    const chartSections = document.querySelectorAll('.chart-container');
    chartSections.forEach(section => {
        section.innerHTML = '<div class="loading">Loading chart data...</div>';
    });
}

// 로딩 상태 숨기기
function hideLoadingState() {
    const loadingElements = document.querySelectorAll('.loading');
    loadingElements.forEach(element => {
        element.style.display = 'none';
    });
}

// 에러 상태 표시
function showErrorState(errorMessage) {
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.innerHTML = `
            <h3>Dashboard Loading Error</h3>
            <p>${errorMessage}</p>
            <button class="btn" onclick="location.reload()">Retry</button>
        `;
        mainContent.insertBefore(errorDiv, mainContent.firstChild);
    }
}

// 메트릭 업데이트
function updateMetrics(modelData) {
    try {
        // 정확도 업데이트
        const accuracyElement = document.getElementById('accuracy-value');
        if (accuracyElement && modelData.accuracy) {
            accuracyElement.textContent = `${(modelData.accuracy * 100).toFixed(2)}%`;
        }
        
        // F1 스코어 업데이트
        const f1Element = document.getElementById('f1-score-value');
        if (f1Element && modelData.f1_score) {
            f1Element.textContent = `${(modelData.f1_score * 100).toFixed(2)}%`;
        }
        
        // 클래스 수 업데이트
        const classesElement = document.getElementById('classes-value');
        if (classesElement && modelData.cluster_names) {
            classesElement.textContent = Object.keys(modelData.cluster_names).length;
        }
        
        // 특성 수 업데이트
        const featuresElement = document.getElementById('features-value');
        if (featuresElement && modelData.feature_count) {
            featuresElement.textContent = modelData.feature_count;
        }
        
        console.log('Metrics updated successfully');
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

// 데이터 새로고침
async function refreshDashboard() {
    try {
        console.log('Refreshing dashboard data...');
        
        // 로딩 상태 표시
        showLoadingState();
        
        // 새 데이터 가져오기
        const newModelData = await fetchModelData();
        globalModelData = newModelData;
        
        // 메트릭 업데이트
        updateMetrics(newModelData);
        
        // 차트 업데이트
        if (globalCharts) {
            updateCharts(globalCharts, newModelData);
        }
        
        hideLoadingState();
        
        // 성공 메시지 표시 (옵션)
        showSuccessMessage('Dashboard refreshed successfully');
        
        console.log('Dashboard refreshed successfully');
        
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
        showErrorMessage('Failed to refresh dashboard data');
    }
}

// 성공 메시지 표시
function showSuccessMessage(message) {
    const notification = createNotification(message, 'success');
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// 에러 메시지 표시
function showErrorMessage(message) {
    const notification = createNotification(message, 'error');
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// 알림 생성
function createNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        z-index: 1000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    
    if (type === 'success') {
        notification.style.backgroundColor = '#51cf66';
    } else if (type === 'error') {
        notification.style.backgroundColor = '#ff6b6b';
    }
    
    notification.textContent = message;
    
    return notification;
}

// 키보드 단축키 처리
document.addEventListener('keydown', function(event) {
    // Ctrl + R 또는 F5로 대시보드 새고고침
    if ((event.ctrlKey && event.key === 'r') || event.key === 'F5') {
        event.preventDefault();
        refreshDashboard();
    }
    
    // ESC로 에러 메시지 닫기
    if (event.key === 'Escape') {
        const notifications = document.querySelectorAll('.notification');
        notifications.forEach(notification => notification.remove());
    }
});

// 대시보드 정보 표시
function showDashboardInfo() {
    const info = `
Dashboard Information:
- Model Accuracy: ${(globalModelData.accuracy * 100).toFixed(2)}%
- Player Types: ${Object.keys(globalModelData.cluster_names).length}
- Features: ${globalModelData.feature_count}
- Data Source: ${globalModelData.isLive ? 'Live API' : 'Mock Data'}
- Last Updated: ${new Date().toLocaleString()}
    `;
    
    alert(info);
}

// 전역 함수로 내보내기 (디버깅용)
window.dashboardFunctions = {
    refresh: refreshDashboard,
    showInfo: showDashboardInfo,
    getModelData: () => globalModelData,
    getCharts: () => globalCharts
};

// 페이지 언로드 시 정리
window.addEventListener('beforeunload', function() {
    // 차트 정리
    if (globalCharts) {
        Object.values(globalCharts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
});

console.log('Dashboard script loaded');
