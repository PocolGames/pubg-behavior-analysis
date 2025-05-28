// 플레이어 유형 정의
const PLAYER_TYPES = {
    survivor: {
        name: 'Survivor',
        icon: '🛡️',
        description: '생존에 집중하며 치료 아이템을 자주 사용하는 신중한 플레이어',
        characteristics: [
            '치료 아이템 사용량이 많음',
            '안전한 플레이 스타일',
            '팀원 지원에 적극적',
            '장거리 전투 선호'
        ],
        color: '#28a745'
    },
    explorer: {
        name: 'Explorer',
        icon: '🗺️',
        description: '맵을 광범위하게 탐험하며 이동거리가 긴 모험적인 플레이어',
        characteristics: [
            '높은 이동거리 기록',
            '맵 탐험을 즐김',
            '다양한 지역 경험',
            '전략적 포지셔닝'
        ],
        color: '#ffc107'
    },
    aggressive: {
        name: 'Aggressive',
        icon: '⚔️',
        description: '적극적인 교전을 즐기며 높은 킬 수와 데미지를 기록하는 공격적인 플레이어',
        characteristics: [
            '높은 킬 수 기록',
            '공격적인 플레이 스타일',
            '높은 데미지 딜링',
            '근거리 전투 선호'
        ],
        color: '#dc3545'
    }
};

// 특성 가중치 (분류 알고리즘용)
const FEATURE_WEIGHTS = {
    kills: 0.25,
    damageDealt: 0.20,
    walkDistance: 0.15,
    rideDistance: 0.10,
    heals: 0.12,
    boosts: 0.08,
    weaponsAcquired: 0.06,
    assists: 0.04
};

// 플레이어 유형별 임계값
const TYPE_THRESHOLDS = {
    aggressive: {
        kills: 5,
        damageDealt: 400,
        killRatio: 0.3
    },
    survivor: {
        heals: 3,
        boosts: 2,
        healRatio: 0.4
    },
    explorer: {
        walkDistance: 2000,
        totalDistance: 3000,
        distanceRatio: 0.6
    }
};

// 통계 정규화 범위
const STAT_RANGES = {
    kills: { min: 0, max: 25, avg: 2.5 },
    damageDealt: { min: 0, max: 2000, avg: 300 },
    walkDistance: { min: 0, max: 8000, avg: 1500 },
    rideDistance: { min: 0, max: 5000, avg: 800 },
    heals: { min: 0, max: 15, avg: 2 },
    boosts: { min: 0, max: 10, avg: 1.5 },
    weaponsAcquired: { min: 0, max: 15, avg: 4 },
    assists: { min: 0, max: 10, avg: 1 }
};

// 샘플 데이터 (테스트용)
const SAMPLE_PROFILES = {
    aggressive: {
        kills: 8,
        damageDealt: 600,
        walkDistance: 1200,
        rideDistance: 400,
        heals: 1,
        boosts: 1,
        weaponsAcquired: 6,
        assists: 0
    },
    survivor: {
        kills: 1,
        damageDealt: 150,
        walkDistance: 2200,
        rideDistance: 1200,
        heals: 6,
        boosts: 4,
        weaponsAcquired: 3,
        assists: 2
    },
    explorer: {
        kills: 3,
        damageDealt: 280,
        walkDistance: 4500,
        rideDistance: 2200,
        heals: 2,
        boosts: 2,
        weaponsAcquired: 4,
        assists: 1
    }
};