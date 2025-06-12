/**
 * PUBG 플레이어 행동 분석 - 클러스터 데이터
 * 8개 플레이어 유형과 특성 정보
 */

// 클러스터 중심점 데이터 (30개 특성의 표준화된 값)
// 각 클러스터의 실제 특성을 반영하도록 수정
const CLUSTER_CENTERS = {
    // 0: Cautious Survivor - 낮은 킬, 높은 치료
    0: [-0.8, -0.5, 0.2, -0.7, 1.5, -0.9, 2.1, -0.3, -0.9, 0.8, -0.6, -0.8, 0.1, 1.2, -0.8, 0.5, 1.8, 1.9, -0.7, -0.6, -0.9, 0.7, 0.1, 0.1, -0.7, -0.6, 0.3, -0.1, -0.1, -0.1],
    
    // 1: Support Survivor - 중간 킬, 매우 높은 치료
    1: [-0.6, -0.3, 0.0, -0.5, 2.2, -0.7, 3.1, 0.1, -0.6, 1.5, -0.4, -0.5, 0.2, 1.8, -0.6, 0.8, 2.8, 2.5, -0.5, -0.4, -0.6, 0.5, 0.0, 0.0, -0.5, -0.4, 0.2, -0.1, -0.1, -0.1],
    
    // 2: Active Explorer - 높은 이동, 중간 킬
    2: [1.8, 1.5, -0.3, 2.1, 0.2, 0.3, 0.5, 0.8, 0.2, 0.4, 1.2, 0.1, -0.1, 0.8, 0.1, 0.0, 0.7, 0.2, 0.3, 0.8, 0.2, 0.1, -0.2, -0.1, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
    
    // 3: Combat Explorer - 높은 킬, 높은 이동
    3: [1.2, 1.0, -0.5, 1.5, 0.5, 1.1, 0.3, 1.2, 1.0, 0.6, 0.8, 0.8, 0.0, 0.4, 0.9, 0.2, 0.8, 0.1, 1.2, 0.9, 1.0, 0.1, -0.1, -0.1, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0],
    
    // 4: Marathon Explorer - 최고 이동, 중간 킬
    4: [2.8, 2.2, -0.2, 3.1, 0.3, 0.5, 0.8, 0.9, 0.4, 0.3, 2.1, 0.2, -0.2, 1.5, 0.3, 0.0, 1.2, 0.1, 0.4, 1.1, 0.4, 0.0, -0.2, -0.1, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0],
    
    // 5: Tactical Explorer - 높은 이동, 높은 무기 수집
    5: [2.1, 1.8, -0.1, 2.3, 0.4, 0.7, 0.6, 2.1, 0.6, 0.2, 1.5, 0.3, 0.0, 0.9, 0.5, 0.1, 0.9, 0.0, 0.6, 0.9, 0.6, 0.0, -0.1, -0.1, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
    
    // 6: Endurance Explorer - 높은 이동, 긴 게임시간
    6: [2.3, 1.9, -0.4, 2.5, 0.1, 0.4, 0.4, 1.1, 0.3, 0.1, 1.8, 0.1, 1.2, 0.6, 0.2, 0.0, 0.5, 0.0, 0.3, 0.8, 0.3, 0.0, -0.2, -0.1, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0],
    
    // 7: Elite Aggressive - 최고 킬, 최고 데미지
    7: [0.8, 0.6, -1.2, 1.0, 1.0, 2.8, 0.9, 1.8, 3.2, 1.8, 0.5, 1.5, 0.3, 0.2, 3.5, 0.8, 1.9, 0.2, 3.1, 1.9, 3.2, 0.0, -0.5, -0.3, 2.9, 2.1, 0.0, 0.1, 0.0, 0.0]
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
