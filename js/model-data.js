// PUBG 모델의 실제 데이터
const ModelData = {
    // 모델 기본 정보
    model: {
        name: "Basic Neural Network",
        accuracy: 0.99245,
        f1_score: 0.9867,
        silhouette_score: 0.1391,
        feature_count: 30,
        cluster_count: 8,
        total_players: 80000
    },

    // 실제 특성 목록 (중요도 순)
    features: [
        { name: "has_kills", importance: 0.3232, description: "킬 여부" },
        { name: "walkDistance_log", importance: 0.0788, description: "도보 거리 (로그)" },
        { name: "walkDistance", importance: 0.0751, description: "도보 거리" },
        { name: "total_distance", importance: 0.0634, description: "총 이동거리" },
        { name: "has_swimDistance", importance: 0.0609, description: "수영 여부" },
        { name: "weaponsAcquired", importance: 0.0588, description: "무기 획득" },
        { name: "killPlace", importance: 0.0573, description: "킬 순위" },
        { name: "damageDealt", importance: 0.0519, description: "총 데미지" },
        { name: "rideDistance", importance: 0.0512, description: "차량 거리" },
        { name: "heal_boost_ratio", importance: 0.0501, description: "치료/부스터 비율" },
        { name: "total_heals", importance: 0.0456, description: "총 치료템" },
        { name: "boosts", importance: 0.0423, description: "부스터 사용" },
        { name: "heals", importance: 0.0401, description: "치료템 사용" },
        { name: "longestKill", importance: 0.0389, description: "최장 킬 거리" },
        { name: "damageDealt_log", importance: 0.0367, description: "데미지 (로그)" },
        { name: "kills", importance: 0.0345, description: "킬 수" },
        { name: "killStreaks", importance: 0.0323, description: "연속 킬" },
        { name: "damage_per_kill", importance: 0.0301, description: "킬당 데미지" },
        { name: "assists", importance: 0.0289, description: "어시스트" },
        { name: "headshotKills", importance: 0.0267, description: "헤드샷 킬" },
        { name: "DBNOs", importance: 0.0245, description: "다운시킨 적" },
        { name: "revives", importance: 0.0234, description: "소생" },
        { name: "swimDistance", importance: 0.0212, description: "수영 거리" },
        { name: "vehicleDestroys", importance: 0.0201, description: "차량 파괴" },
        { name: "kill_efficiency", importance: 0.0189, description: "킬 효율성" },
        { name: "maxPlace", importance: 0.0178, description: "최대 순위" },
        { name: "roadKills", importance: 0.0167, description: "로드킬" },
        { name: "teamKills", importance: 0.0156, description: "팀킬" },
        { name: "killPoints", importance: 0.0145, description: "킬 포인트" },
        { name: "matchDuration", importance: 0.0134, description: "경기 시간" }
    ],

    // 클러스터 정보 (실제 데이터 기반)
    clusters: [
        {
            id: 0,
            name: "Survivor Type A",
            label: "Survivor",
            count: 14527,
            percentage: 18.2,
            color: "#22c55e",
            characteristics: [
                "heal_boost_ratio: 평균 대비 775.29배",
                "assists: 평균 대비 479.86배", 
                "has_swimDistance: 평균 대비 294.02배"
            ],
            accuracy: 0.996
        },
        {
            id: 1,
            name: "Survivor Type B", 
            label: "Survivor",
            count: 24981,
            percentage: 31.2,
            color: "#16a34a",
            characteristics: [
                "heal_boost_ratio: 평균 대비 1861.49배",
                "assists: 평균 대비 964.62배",
                "damage_per_kill: 평균 대비 864.65배"
            ],
            accuracy: 0.999
        },
        {
            id: 2,
            name: "Explorer Type A",
            label: "Explorer", 
            count: 10756,
            percentage: 13.4,
            color: "#3b82f6",
            characteristics: [
                "walkDistance_log: 평균 대비 3743.08배",
                "walkDistance: 평균 대비 1179.09배",
                "revives: 평균 대비 626.25배"
            ],
            accuracy: 0.990
        },
        {
            id: 3,
            name: "Explorer Type B",
            label: "Explorer",
            count: 15898, 
            percentage: 19.9,
            color: "#2563eb",
            characteristics: [
                "walkDistance_log: 평균 대비 2245.80배",
                "longestKill: 평균 대비 610.84배",
                "has_kills: 평균 대비 501.94배"
            ],
            accuracy: 0.994
        },
        {
            id: 4,
            name: "Explorer Type C",
            label: "Explorer",
            count: 4312,
            percentage: 5.4,
            color: "#1d4ed8",
            characteristics: [
                "walkDistance_log: 평균 대비 4451.52배", 
                "walkDistance: 평균 대비 1845.04배",
                "revives: 평균 대비 1551.02배"
            ],
            accuracy: 0.969
        },
        {
            id: 5,
            name: "Explorer Type D",
            label: "Explorer",
            count: 4046,
            percentage: 5.1,
            color: "#1e40af",
            characteristics: [
                "walkDistance_log: 평균 대비 4139.77배",
                "walkDistance: 평균 대비 1544.91배", 
                "weaponsAcquired: 평균 대비 451.79배"
            ],
            accuracy: 0.996
        },
        {
            id: 6,
            name: "Explorer Type E",
            label: "Explorer",
            count: 5391,
            percentage: 6.7,
            color: "#1e3a8a",
            characteristics: [
                "walkDistance_log: 평균 대비 3995.30배",
                "matchDuration: 평균 대비 1400.69배",
                "walkDistance: 평균 대비 1327.73배"
            ],
            accuracy: 0.967
        },
        {
            id: 7,
            name: "Aggressive Fighter",
            label: "Aggressive",
            count: 89,
            percentage: 0.1,
            color: "#ef4444",
            characteristics: [
                "kill_efficiency: 평균 대비 23396.88배",
                "damage_per_kill: 평균 대비 1435.73배",
                "assists: 평균 대비 920.03배"
            ],
            accuracy: 1.000
        }
    ],

    // Confusion Matrix 데이터 (8x8)
    confusionMatrix: [
        [0.996, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000],
        [0.001, 0.999, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.005, 0.003, 0.990, 0.002, 0.000, 0.000, 0.000, 0.000],
        [0.003, 0.002, 0.001, 0.994, 0.000, 0.000, 0.000, 0.000],
        [0.015, 0.012, 0.004, 0.000, 0.969, 0.000, 0.000, 0.000],
        [0.002, 0.001, 0.001, 0.000, 0.000, 0.996, 0.000, 0.000],
        [0.016, 0.013, 0.004, 0.000, 0.000, 0.000, 0.967, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]
    ],

    // 모델 훈련 히스토리
    trainingHistory: {
        epochs: 50,
        finalTrainAccuracy: 0.9869,
        finalValAccuracy: 0.9927,
        finalTrainLoss: 0.0347,
        finalValLoss: 0.0181,
        bestEpoch: 49
    },

    // 비즈니스 인사이트
    insights: [
        "각 플레이어 유형별 맞춤형 컨텐츠 제공 가능",
        "플레이어 행동 패턴 기반 게임 밸런싱 개선", 
        "신규 플레이어 온보딩 전략 최적화",
        "Survivor 타입이 전체의 49.4%로 가장 높은 비율",
        "Aggressive 타입은 0.1%로 극소수이지만 100% 정확도"
    ]
};

// 데이터 접근 유틸리티 함수들
const ModelDataUtils = {
    // 상위 N개 특성 가져오기
    getTopFeatures: function(n = 10) {
        return ModelData.features.slice(0, n);
    },

    // 클러스터별 데이터 가져오기
    getClusterById: function(id) {
        return ModelData.clusters.find(cluster => cluster.id === id);
    },

    // 라벨별 클러스터들 가져오기
    getClustersByLabel: function(label) {
        return ModelData.clusters.filter(cluster => cluster.label === label);
    },

    // 전체 클러스터 통계
    getClusterStats: function() {
        const stats = {};
        ModelData.clusters.forEach(cluster => {
            if (!stats[cluster.label]) {
                stats[cluster.label] = {
                    count: 0,
                    percentage: 0,
                    clusters: []
                };
            }
            stats[cluster.label].count += cluster.count;
            stats[cluster.label].percentage += cluster.percentage;
            stats[cluster.label].clusters.push(cluster);
        });
        return stats;
    },

    // 평균 정확도 계산
    getAverageAccuracy: function() {
        const totalAccuracy = ModelData.clusters.reduce((sum, cluster) => {
            return sum + (cluster.accuracy * cluster.percentage / 100);
        }, 0);
        return totalAccuracy;
    }
};
