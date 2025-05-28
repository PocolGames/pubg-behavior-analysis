// API 설정
const API_BASE_URL = 'http://127.0.0.1:8000';

// API 상태 확인
let apiStatus = {
    isConnected: false,
    lastCheck: null
};

// API 헬스체크
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            timeout: 5000
        });
        
        if (response.ok) {
            const data = await response.json();
            apiStatus.isConnected = true;
            apiStatus.lastCheck = new Date();
            return true;
        }
        return false;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        apiStatus.isConnected = false;
        return false;
    }
}

// 모델 정보 가져오기
async function getModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        if (response.ok) {
            return await response.json();
        }
        throw new Error('Failed to fetch model info');
    } catch (error) {
        console.error('Error fetching model info:', error);
        return null;
    }
}

// 플레이어 예측 API 호출
async function predictPlayer(playerData) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(playerData)
        });
        
        if (response.ok) {
            return await response.json();
        }
        throw new Error(`API Error: ${response.status}`);
    } catch (error) {
        console.error('Error predicting player:', error);
        return null;
    }
}

// Mock 데이터 (API가 연결되지 않을 때 사용)
const mockModelData = {
    model_name: "PUBG Player Classifier",
    accuracy: 0.9925,
    f1_score: 0.9867,
    feature_count: 30,
    cluster_names: {
        "0": "Aggressive Fighter",
        "1": "Cautious Survivor", 
        "2": "Mobile Explorer",
        "3": "Team Supporter",
        "4": "Balanced Player",
        "5": "Strategic Player",
        "6": "Stealth Player",
        "7": "Support Specialist"
    },
    class_accuracies: {
        "Aggressive Fighter": 0.996,
        "Cautious Survivor": 0.999,
        "Mobile Explorer": 0.990,
        "Team Supporter": 0.994,
        "Balanced Player": 0.969,
        "Strategic Player": 0.996,
        "Stealth Player": 0.967,
        "Support Specialist": 1.000
    },
    feature_importance: {
        "has_kills": 0.3232,
        "walkDistance_log": 0.0788,
        "walkDistance": 0.0751,
        "total_distance": 0.0634,
        "has_swimDistance": 0.0609,
        "weaponsAcquired": 0.0588,
        "killPlace": 0.0573,
        "damageDealt": 0.0519,
        "rideDistance": 0.0512,
        "heal_boost_ratio": 0.0501
    },
    player_distribution: {
        "Aggressive Fighter": 89,
        "Cautious Survivor": 39508,
        "Mobile Explorer": 35056,
        "Team Supporter": 5391,
        "Balanced Player": 4312,
        "Strategic Player": 4046,
        "Stealth Player": 10756,
        "Support Specialist": 14527
    }
};

// 모델 데이터 가져오기 (API 또는 Mock)
async function fetchModelData() {
    const isApiHealthy = await checkApiHealth();
    
    if (isApiHealthy) {
        const modelInfo = await getModelInfo();
        if (modelInfo) {
            return {
                ...mockModelData,
                ...modelInfo,
                isLive: true
            };
        }
    }
    
    return {
        ...mockModelData,
        isLive: false
    };
}

// API 상태 표시 업데이트
function updateApiStatus() {
    const statusElements = document.querySelectorAll('.api-status');
    statusElements.forEach(element => {
        if (apiStatus.isConnected) {
            element.className = 'status-indicator status-online';
            element.title = 'API Connected';
        } else {
            element.className = 'status-indicator status-offline';
            element.title = 'API Offline - Using Mock Data';
        }
    });
}

// 에러 처리 함수
function handleApiError(error, context = '') {
    console.error(`API Error ${context}:`, error);
    
    // 사용자에게 에러 표시 (옵션)
    const errorElement = document.getElementById('api-error');
    if (errorElement) {
        errorElement.textContent = `API Error: ${error.message}`;
        errorElement.style.display = 'block';
    }
}

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    checkApiHealth().then(updateApiStatus);
    
    // 주기적으로 API 상태 확인 (30초마다)
    setInterval(() => {
        checkApiHealth().then(updateApiStatus);
    }, 30000);
});
