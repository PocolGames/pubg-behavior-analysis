// Dashboard Data - 실제 모델 데이터 기반
const DASHBOARD_DATA = {
    // 모델 성능 메트릭
    modelMetrics: {
        accuracy: 0.99245,
        featureCount: 30,
        clusterCount: 8,
        sampleCount: 80000,
        silhouetteScore: 0.1391
    },

    // 실제 특성 목록 (30개)
    features: [
        "walkDistance", "walkDistance_log", "killPlace", "total_distance", "boosts",
        "weaponsAcquired", "total_heals", "damageDealt", "heal_boost_ratio", "kills",
        "heals", "longestKill", "damageDealt_log", "has_kills", "killStreaks",
        "rideDistance", "damage_per_kill", "assists", "headshotKills", "DBNOs",
        "revives", "has_swimDistance", "swimDistance", "vehicleDestroys", "kill_efficiency",
        "maxPlace", "roadKills", "teamKills", "killPoints", "matchDuration"
    ],

    // 특성 중요도 (모델 기반 추정값)
    featureImportance: {
        "has_kills": 0.3232,
        "walkDistance_log": 0.0788,
        "walkDistance": 0.0751,
        "total_distance": 0.0634,
        "has_swimDistance": 0.0609,
        "weaponsAcquired": 0.0588,
        "killPlace": 0.0573,
        "damageDealt": 0.0519,
        "rideDistance": 0.0512,
        "heal_boost_ratio": 0.0501,
        "kills": 0.0485,
        "boosts": 0.0422,
        "damage_per_kill": 0.0398,
        "total_heals": 0.0365,
        "longestKill": 0.0342,
        "heals": 0.0298,
        "damageDealt_log": 0.0276,
        "assists": 0.0251,
        "kill_efficiency": 0.0234,
        "killStreaks": 0.0212
    },

    // 클러스터 정보 (실제 분석 결과)
    clusters: {
        0: {
            name: "Survivor",
            type: "생존형",
            count: 14527,
            percentage: 18.2,
            color: "#4CAF50",
            description: "치료템과 부스터를 적극 활용하는 생존 중심 플레이어",
            keyFeatures: [
                { name: "heal_boost_ratio", ratio: 775.29 },
                { name: "assists", ratio: 479.86 },
                { name: "has_swimDistance", ratio: 294.02 }
            ]
        },
        1: {
            name: "Survivor",
            type: "생존형 (고급)",
            count: 24981,
            percentage: 31.2,
            color: "#66BB6A",
            description: "높은 생존 기술과 팀워크를 보이는 숙련된 생존 플레이어",
            keyFeatures: [
                { name: "heal_boost_ratio", ratio: 1861.49 },
                { name: "assists", ratio: 964.62 },
                { name: "damage_per_kill", ratio: 864.65 }
            ]
        },
        2: {
            name: "Explorer",
            type: "탐험형",
            count: 10756,
            percentage: 13.4,
            color: "#2196F3",
            description: "맵을 광범위하게 탐험하며 이동하는 플레이어",
            keyFeatures: [
                { name: "walkDistance_log", ratio: 3743.08 },
                { name: "walkDistance", ratio: 1179.09 },
                { name: "revives", ratio: 626.25 }
            ]
        },
        3: {
            name: "Explorer",
            type: "탐험형 (전투)",
            count: 15898,
            percentage: 19.9,
            color: "#42A5F5",
            description: "탐험과 전투를 병행하는 균형잡힌 플레이어",
            keyFeatures: [
                { name: "walkDistance_log", ratio: 2245.80 },
                { name: "longestKill", ratio: 610.84 },
                { name: "has_kills", ratio: 501.94 }
            ]
        },
        4: {
            name: "Explorer",
            type: "탐험형 (고급)",
            count: 4312,
            percentage: 5.4,
            color: "#1976D2",
            description: "높은 이동성과 팀 지원 능력을 가진 고급 탐험가",
            keyFeatures: [
                { name: "walkDistance_log", ratio: 4451.52 },
                { name: "walkDistance", ratio: 1845.04 },
                { name: "revives", ratio: 1551.02 }
            ]
        },
        5: {
            name: "Explorer",
            type: "탐험형 (수집)",
            count: 4046,
            percentage: 5.1,
            color: "#1565C0",
            description: "무기 수집과 탐험에 특화된 플레이어",
            keyFeatures: [
                { name: "walkDistance_log", ratio: 4139.77 },
                { name: "walkDistance", ratio: 1544.91 },
                { name: "weaponsAcquired", ratio: 451.79 }
            ]
        },
        6: {
            name: "Explorer",
            type: "탐험형 (지구력)",
            count: 5391,
            percentage: 6.7,
            color: "#0D47A1",
            description: "긴 게임 시간과 높은 이동성을 보이는 지구력형 플레이어",
            keyFeatures: [
                { name: "walkDistance_log", ratio: 3995.30 },
                { name: "matchDuration", ratio: 1400.69 },
                { name: "walkDistance", ratio: 1327.73 }
            ]
        },
        7: {
            name: "Aggressive",
            type: "공격형",
            count: 89,
            percentage: 0.1,
            color: "#F44336",
            description: "극도로 공격적인 플레이 스타일을 보이는 희귀한 플레이어",
            keyFeatures: [
                { name: "kill_efficiency", ratio: 23396.88 },
                { name: "damage_per_kill", ratio: 1435.73 },
                { name: "assists", ratio: 920.03 }
            ]
        }
    }
};

// 성능 메트릭 계산 함수
function calculatePerformanceMetrics() {
    const totalPlayers = Object.values(DASHBOARD_DATA.clusters)
        .reduce((sum, cluster) => sum + cluster.count, 0);
    
    return {
        totalAccuracy: DASHBOARD_DATA.modelMetrics.accuracy,
        avgConfidence: 0.925,
        processingTime: 12.5,
        silhouetteScore: DASHBOARD_DATA.modelMetrics.silhouetteScore
    };
}

// 클러스터 분포 데이터 생성
function getClusterDistributionData() {
    const clusters = DASHBOARD_DATA.clusters;
    return {
        labels: Object.values(clusters).map(c => `${c.name} (${c.type})`),
        data: Object.values(clusters).map(c => c.percentage),
        colors: Object.values(clusters).map(c => c.color),
        counts: Object.values(clusters).map(c => c.count)
    };
}

// 상위 특성 중요도 데이터 (상위 15개)
function getTopFeatureImportance() {
    const importance = DASHBOARD_DATA.featureImportance;
    const sorted = Object.entries(importance)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 15);
    
    return {
        labels: sorted.map(([name]) => name),
        data: sorted.map(([,value]) => value)
    };
}

// 한국어 특성명 매핑
const FEATURE_NAMES_KO = {
    "walkDistance": "도보 이동거리",
    "walkDistance_log": "도보 이동거리 (로그)",
    "killPlace": "킬 순위",
    "total_distance": "총 이동거리",
    "boosts": "부스터 사용",
    "weaponsAcquired": "무기 획득",
    "total_heals": "총 치료템",
    "damageDealt": "총 데미지",
    "heal_boost_ratio": "치료/부스터 비율",
    "kills": "킬 수",
    "heals": "치료템 사용",
    "longestKill": "최장거리 킬",
    "damageDealt_log": "총 데미지 (로그)",
    "has_kills": "킬 여부",
    "killStreaks": "연속킬",
    "rideDistance": "차량 이동거리",
    "damage_per_kill": "킬당 데미지",
    "assists": "어시스트",
    "headshotKills": "헤드샷 킬",
    "DBNOs": "다운 수",
    "revives": "부활",
    "has_swimDistance": "수영 여부",
    "swimDistance": "수영 거리",
    "vehicleDestroys": "차량 파괴",
    "kill_efficiency": "킬 효율성",
    "maxPlace": "최대 순위",
    "roadKills": "차량킬",
    "teamKills": "팀킬",
    "killPoints": "킬 포인트",
    "matchDuration": "게임 시간"
};

// 특성명 번역 함수
function translateFeatureName(englishName) {
    return FEATURE_NAMES_KO[englishName] || englishName;
}
