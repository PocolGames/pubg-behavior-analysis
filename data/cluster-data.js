/**
 * PUBG 플레이어 행동 분석 - 클러스터 데이터
 * 8개 플레이어 유형과 특성 정보
 */

// 클러스터 중심점 데이터 (30개 특성의 표준화된 값)
const CLUSTER_CENTERS = {
    0: [-0.42, 1.85, -0.31, -0.28, -0.15, -0.21, -0.34, -0.29, -0.18, -0.25, -0.31, -0.19, -0.22, -0.26, -0.33, -0.28, -0.24, -0.31, -0.26, -0.29, -0.35, -0.32, -0.27, -0.23, -0.30, -0.34, -0.28, -0.25, -0.31, -0.27],
    1: [-0.38, 2.12, -0.35, -0.32, -0.18, -0.24, -0.38, -0.33, -0.21, -0.28, -0.35, -0.22, -0.25, -0.29, -0.37, -0.31, -0.27, -0.35, -0.29, -0.32, -0.39, -0.36, -0.30, -0.26, -0.33, -0.38, -0.31, -0.28, -0.35, -0.30],
    2: [1.25, -0.45, 0.85, 0.78, 0.42, 0.68, 0.92, 0.71, 0.56, 0.83, 0.89, 0.65, 0.74, 0.86, 0.95, 0.81, 0.73, 0.91, 0.77, 0.84, 0.98, 0.93, 0.79, 0.66, 0.87, 0.96, 0.82, 0.75, 0.90, 0.78],
    3: [0.95, -0.32, 0.68, 0.61, 0.35, 0.55, 0.74, 0.58, 0.45, 0.67, 0.72, 0.52, 0.59, 0.69, 0.76, 0.65, 0.58, 0.73, 0.62, 0.68, 0.78, 0.75, 0.63, 0.53, 0.70, 0.77, 0.66, 0.60, 0.72, 0.62],
    4: [1.85, -0.28, 1.42, 1.35, 0.78, 1.21, 1.58, 1.28, 0.95, 1.45, 1.52, 1.18, 1.31, 1.48, 1.62, 1.39, 1.25, 1.55, 1.33, 1.43, 1.68, 1.59, 1.36, 1.15, 1.49, 1.65, 1.41, 1.29, 1.54, 1.34],
    5: [1.68, -0.25, 1.28, 1.22, 0.71, 1.09, 1.42, 1.15, 0.86, 1.31, 1.37, 1.06, 1.18, 1.33, 1.46, 1.25, 1.12, 1.40, 1.20, 1.29, 1.51, 1.43, 1.22, 1.03, 1.34, 1.48, 1.27, 1.16, 1.39, 1.21],
    6: [1.55, -0.22, 1.18, 1.12, 0.65, 1.01, 1.31, 1.06, 0.79, 1.21, 1.26, 0.98, 1.09, 1.23, 1.35, 1.15, 1.03, 1.29, 1.11, 1.19, 1.39, 1.32, 1.13, 0.95, 1.24, 1.36, 1.17, 1.07, 1.28, 1.12],
    7: [3.25, -0.15, 2.98, 2.85, 1.95, 2.76, 3.12, 2.71, 2.35, 2.89, 3.05, 2.64, 2.78, 2.94, 3.18, 2.83, 2.69, 3.08, 2.75, 2.87, 3.22, 3.15, 2.81, 2.58, 2.92, 3.19, 2.85, 2.72, 3.06, 2.78]
};

// 특성 이름 매핑 (30개 특성)
const FEATURE_NAMES = [
    'walkDistance', 'walkDistance_log', 'killPlace', 'total_distance', 'boosts',
    'damageDealt', 'heals', 'weaponsAcquired', 'kills', 'assists',
    'rideDistance', 'longestKill', 'matchDuration', 'revives', 'kill_efficiency',
    'damage_per_kill', 'total_heals', 'heal_boost_ratio', 'aggressiveness_score',
    'damageDealt_log', 'has_kills', 'has_swimDistance', 'numGroups', 'maxPlace',
    'DBNOs', 'headshotKills', 'swimDistance', 'roadKills', 'vehicleDestroys', 'teamKills'
];

// 클러스터 정보 (분석 결과 기반)
const CLUSTER_INFO = {
    0: {
        name: "Cautious Survivor",
        type: "Survivor",
        percentage: 18.2,
        description: "생존에 중점을 둔 신중한 플레이어",
        characteristics: [
            "높은 치료 아이템 사용률 (775.29배)",
            "팀원 지원 활동 활발 (479.86배)",
            "수영 거리 활용 (294.02배)",
            "안전한 플레이 스타일"
        ],
        color: "#28a745", // 초록색
        icon: "🛡️",
        strategy: "안전 지대 확보와 생존률 극대화",
        strengths: ["생존력", "팀플레이", "자원 관리"],
        weaknesses: ["공격력", "기동성"]
    },
    1: {
        name: "Support Survivor", 
        type: "Survivor",
        percentage: 31.2,
        description: "팀을 지원하며 생존하는 플레이어",
        characteristics: [
            "매우 높은 치료 아이템 사용 (1861.49배)",
            "뛰어난 팀원 지원 (964.62배)",
            "효율적인 킬당 데미지 (864.65배)",
            "균형잡힌 서포트 플레이"
        ],
        color: "#20c997", // 청록색
        icon: "⚕️",
        strategy: "팀원 생존 지원과 안정적인 플레이",
        strengths: ["지원력", "치료", "팀워크"],
        weaknesses: ["개인 화력", "선제공격"]
    },
    2: {
        name: "Active Explorer",
        type: "Explorer", 
        percentage: 13.4,
        description: "적극적으로 맵을 탐험하는 플레이어",
        characteristics: [
            "매우 높은 이동 거리 (3743.08배)",
            "뛰어난 보행 거리 (1179.09배)",
            "팀원 부활 활동 (626.25배)",
            "공격적인 탐험 스타일"
        ],
        color: "#17a2b8", // 파란색
        icon: "🔍",
        strategy: "맵 장악과 정보 수집 중심",
        strengths: ["기동성", "정보력", "맵 컨트롤"],
        weaknesses: ["생존력", "방어력"]
    },
    3: {
        name: "Combat Explorer",
        type: "Explorer",
        percentage: 19.9, 
        description: "전투와 탐험을 병행하는 플레이어",
        characteristics: [
            "높은 이동성 (2245.80배)",
            "우수한 장거리 킬 (610.84배)",
            "활발한 킬 활동 (501.94배)",
            "공격적 탐험 스타일"
        ],
        color: "#6f42c1", // 보라색
        icon: "⚔️",
        strategy: "이동하며 적극적인 교전",
        strengths: ["공격력", "기동성", "사거리"],
        weaknesses: ["지속력", "팀플레이"]
    },
    4: {
        name: "Marathon Explorer",
        type: "Explorer",
        percentage: 5.4,
        description: "장거리 이동을 즐기는 탐험가",
        characteristics: [
            "최고 수준 이동성 (4451.52배)",
            "극도로 높은 보행 거리 (1845.04배)",
            "높은 팀원 부활률 (1551.02배)",
            "초장거리 플레이 스타일"
        ],
        color: "#fd7e14", // 주황색
        icon: "🏃",
        strategy: "광범위한 맵 커버리지",
        strengths: ["지구력", "맵 장악", "구조"],
        weaknesses: ["전투력", "효율성"]
    },
    5: {
        name: "Tactical Explorer", 
        type: "Explorer",
        percentage: 5.1,
        description: "전술적 아이템 수집과 탐험",
        characteristics: [
            "뛰어난 이동성 (4139.77배)",
            "높은 보행 거리 (1544.91배)",
            "무기 수집 전문 (451.79배)",
            "장비 중심 플레이"
        ],
        color: "#e83e8c", // 핑크색  
        icon: "🎯",
        strategy: "아이템 파밍과 전술적 위치선점",
        strengths: ["장비", "전술", "준비성"],
        weaknesses: ["즉흥성", "근접전"]
    },
    6: {
        name: "Endurance Explorer",
        type: "Explorer", 
        percentage: 6.7,
        description: "오래 버티며 탐험하는 플레이어",
        characteristics: [
            "높은 이동성 (3995.30배)",
            "긴 게임 지속시간 (1400.69배)",
            "지속적인 보행 (1327.73배)",
            "장기전 특화"
        ],
        color: "#6c757d", // 회색
        icon: "⏰",
        strategy: "장기전과 지구력 기반 플레이",
        strengths: ["지구력", "인내심", "후반전"],
        weaknesses: ["초반", "순발력"]
    },
    7: {
        name: "Elite Aggressive",
        type: "Aggressive",
        percentage: 0.1,
        description: "최고 수준의 공격적 플레이어",
        characteristics: [
            "극도의 킬 효율성 (23396.88배)",
            "완벽한 킬당 데미지 (1435.73배)",
            "뛰어난 어시스트 (920.03배)",
            "프로급 공격 실력"
        ],
        color: "#dc3545", // 빨간색
        icon: "💀",
        strategy: "압도적인 개인 실력으로 게임 장악",
        strengths: ["화력", "정확도", "경험"],
        weaknesses: ["희귀성", "팀 의존도"]
    }
};

// 특성 중요도 (분석 결과 기반)
const FEATURE_IMPORTANCE = {
    'has_kills': 0.3232,
    'walkDistance_log': 0.0788,
    'walkDistance': 0.0751,
    'total_distance': 0.0634,
    'has_swimDistance': 0.0609,
    'weaponsAcquired': 0.0588,
    'killPlace': 0.0573,
    'damageDealt': 0.0519,
    'rideDistance': 0.0512,
    'heal_boost_ratio': 0.0501
};

// 게임 밸런스 정보
const GAME_BALANCE = {
    level: "Balanced",
    score: 9.56,
    description: "플레이어 유형 분포가 균형잡혀 있음"
};

// 내보내기
window.CLUSTER_DATA = {
    CLUSTER_CENTERS,
    FEATURE_NAMES, 
    CLUSTER_INFO,
    FEATURE_IMPORTANCE,
    GAME_BALANCE
};
