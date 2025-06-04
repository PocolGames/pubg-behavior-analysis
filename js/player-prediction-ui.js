/**
 * PUBG Player Prediction System - UI Module
 * 플레이어 예측 시스템 UI 모듈
 * 차트 업데이트, 결과 표시, 상태 관리 담당
 */

/**
 * PlayerPredictor 클래스에 UI 관련 메서드 추가
 */
Object.assign(PlayerPredictor.prototype, {
    /**
     * 예측 결과 업데이트
     */
    updatePredictionResults(prediction, predictionTime) {
        const clusterInfo = this.getClusterInfo()[prediction.predictedCluster];
        
        // 예측 요약 업데이트
        const summary = document.getElementById('predictionSummary');
        const typeName = document.getElementById('predictedTypeName');
        const typeDescription = document.getElementById('predictedTypeDescription');
        const confidenceScore = document.getElementById('confidenceScore');
        
        if (summary && typeName && typeDescription && confidenceScore) {
            summary.style.display = 'block';
            typeName.textContent = clusterInfo.name;
            typeDescription.textContent = clusterInfo.description;
            confidenceScore.textContent = prediction.confidence.toFixed(1) + '%';
            
            // 아이콘 색상 변경
            const typeIcon = summary.querySelector('.type-icon i');
            if (typeIcon) {
                typeIcon.style.color = clusterInfo.color;
                typeIcon.className = clusterInfo.icon;
            }
        }

        // 분석 세부사항 업데이트
        const details = document.getElementById('analysisDetails');
        if (details) {
            details.style.display = 'block';
            
            const strongestFeature = document.getElementById('strongestFeature');
            const playStyle = document.getElementById('playStyle');
            const similarPlayers = document.getElementById('similarPlayers');
            const predictionTimeEl = document.getElementById('predictionTime');
            
            if (strongestFeature) strongestFeature.textContent = prediction.analysis.strongestFeature;
            if (playStyle) playStyle.textContent = prediction.analysis.playStyle;
            if (similarPlayers) similarPlayers.textContent = prediction.analysis.similarPlayers;
            if (predictionTimeEl) predictionTimeEl.textContent = predictionTime + '초';
        }
    },

    /**
     * 차트 업데이트
     */
    updateCharts(formData, prediction) {
        this.updateProbabilityChart(prediction.probabilities);
        this.updateRadarChart(formData, prediction.features);
    },

    /**
     * 확률 분포 차트 업데이트
     */
    updateProbabilityChart(probabilities) {
        const chart = this.charts.get('probability');
        if (chart) {
            const percentages = probabilities.map(p => (p * 100).toFixed(1));
            chart.data.datasets[0].data = percentages;
            chart.update('active');
        }
    },

    /**
     * 레이더 차트 업데이트
     */
    updateRadarChart(formData, features) {
        const chart = this.charts.get('radar');
        if (chart) {
            const radarData = [
                Math.min(100, formData.kills * 10),
                Math.min(100, formData.damageDealt * 0.1),
                Math.min(100, formData.walkDistance * 0.01),
                features.survival,
                Math.min(100, formData.assists * 15),
                Math.min(100, formData.weaponsAcquired * 12),
                Math.min(100, formData.boosts * 15),
                Math.min(100, formData.heals * 12)
            ];
            
            chart.data.datasets[0].data = radarData;
            chart.update('active');
        }
    },

    /**
     * 예측 상태 업데이트
     */
    updatePredictionStatus(status, type) {
        const statusEl = document.getElementById('predictionStatus');
        if (statusEl) {
            const indicator = statusEl.querySelector('.status-indicator');
            if (indicator) {
                indicator.textContent = status;
                indicator.className = `status-indicator ${type}`;
            }
        }
    },

    /**
     * 폼 초기화
     */
    resetForm() {
        const form = document.getElementById('playerForm');
        if (form) {
            form.reset();
            
            // 기본값 설정
            const defaults = {
                kills: 0, damageDealt: 0, longestKill: 0, headshotKills: 0, assists: 0, weaponsAcquired: 3,
                walkDistance: 1000, rideDistance: 0, swimDistance: 0,
                heals: 1, boosts: 1, revives: 0, DBNOs: 0,
                killPlace: 50, matchDuration: 1800, maxPlace: 100, numGroups: 50
            };
            
            for (const [key, value] of Object.entries(defaults)) {
                const input = document.getElementById(key);
                if (input) input.value = value;
            }
        }
        
        // 결과 패널 숨기기
        this.hidePredictionResults();
        
        // 차트 초기화
        this.createPlaceholderCharts();
        
        if (window.App) {
            App.showNotification('폼이 초기화되었습니다.', 'info');
        }
    },

    /**
     * 샘플 플레이어 로드
     */
    loadSamplePlayer(playerType) {
        const sampleData = this.getSamplePlayers()[playerType];
        if (!sampleData) return;
        
        // 폼에 데이터 입력
        for (const [key, value] of Object.entries(sampleData)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;
            }
        }
        
        if (window.App) {
            App.showNotification(`${playerType} 샘플 데이터가 로드되었습니다.`, 'success');
        }
        
        // 자동 예측 실행 (선택사항)
        setTimeout(() => {
            this.predictPlayerType();
        }, 500);
    },

    /**
     * 샘플 메뉴 표시
     */
    showSampleMenu() {
        // 샘플 플레이어 섹션으로 스크롤
        const sampleSection = document.querySelector('.sample-players');
        if (sampleSection) {
            sampleSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // 하이라이트 효과
            sampleSection.style.animation = 'pulse 1s ease-in-out';
            setTimeout(() => {
                sampleSection.style.animation = '';
            }, 1000);
        }
    },

    /**
     * 예측 결과 숨기기
     */
    hidePredictionResults() {
        const summary = document.getElementById('predictionSummary');
        const details = document.getElementById('analysisDetails');
        
        if (summary) summary.style.display = 'none';
        if (details) details.style.display = 'none';
        
        this.updatePredictionStatus('대기중', '');
    },

    /**
     * 샘플 플레이어 데이터 가져오기
     */
    getSamplePlayers() {
        if (this.isDataLoaded && this.predictionData.samplePlayers) {
            const samplePlayers = {};
            const featureNames = this.getFeatureNames();
            
            Object.keys(this.predictionData.samplePlayers).forEach(key => {
                const sample = this.predictionData.samplePlayers[key];
                const playerData = {};
                
                // 배열 데이터를 객체로 변환
                if (Array.isArray(sample.data)) {
                    featureNames.forEach((name, index) => {
                        playerData[name] = sample.data[index] || 0;
                    });
                } else {
                    // 객체 형태인 경우
                    Object.assign(playerData, sample.data);
                }
                
                // 추가 기본값 설정
                playerData.headshotKills = playerData.headshotKills || 0;
                playerData.swimDistance = playerData.swimDistance || 0;
                playerData.revives = playerData.revives || 0;
                playerData.DBNOs = playerData.DBNOs || playerData.kills || 0;
                playerData.matchDuration = playerData.matchDuration || 1800;
                playerData.maxPlace = playerData.maxPlace || 100;
                playerData.numGroups = playerData.numGroups || 50;
                
                samplePlayers[key] = playerData;
            });
            
            return samplePlayers;
        }
        
        // 백업 샘플 데이터 (4가지 유형)
        return {
            'conservative': {
                kills: 1, damageDealt: 150, longestKill: 50, headshotKills: 0, assists: 0, weaponsAcquired: 2,
                walkDistance: 800, rideDistance: 0, swimDistance: 0, heals: 3, boosts: 1, revives: 0, DBNOs: 1,
                killPlace: 70, matchDuration: 2000, maxPlace: 100, numGroups: 50
            },
            'aggressive': {
                kills: 3, damageDealt: 400, longestKill: 200, headshotKills: 1, assists: 1, weaponsAcquired: 4,
                walkDistance: 1500, rideDistance: 500, swimDistance: 0, heals: 2, boosts: 2, revives: 0, DBNOs: 3,
                killPlace: 30, matchDuration: 1800, maxPlace: 100, numGroups: 50
            },
            'explorer': {
                kills: 2, damageDealt: 250, longestKill: 100, headshotKills: 0, assists: 1, weaponsAcquired: 5,
                walkDistance: 3000, rideDistance: 1000, swimDistance: 100, heals: 1, boosts: 1, revives: 1, DBNOs: 2,
                killPlace: 45, matchDuration: 2100, maxPlace: 100, numGroups: 50
            },
            'balanced': {
                kills: 8, damageDealt: 800, longestKill: 300, headshotKills: 2, assists: 2, weaponsAcquired: 6,
                walkDistance: 1200, rideDistance: 200, swimDistance: 0, heals: 1, boosts: 1, revives: 0, DBNOs: 8,
                killPlace: 5, matchDuration: 1500, maxPlace: 100, numGroups: 50
            }
        };
    },

    /**
     * 입력 필드 스타일 업데이트
     */
    updateInputStyles() {
        const inputs = document.querySelectorAll('#playerForm input[type="number"]');
        inputs.forEach(input => {
            // 값에 따른 스타일링
            const value = parseFloat(input.value) || 0;
            const max = parseFloat(input.max) || 100;
            const percentage = (value / max) * 100;
            
            // 진행률에 따른 색상 변경
            if (percentage > 80) {
                input.style.borderColor = '#dc3545';
            } else if (percentage > 50) {
                input.style.borderColor = '#ff6b35';
            } else if (percentage > 20) {
                input.style.borderColor = '#667eea';
            } else {
                input.style.borderColor = '#56ab2f';
            }
        });
    },

    /**
     * 애니메이션 효과 적용
     */
    applyAnimation(element, animationType = 'fadeIn') {
        if (!element) return;
        
        element.style.animation = 'none';
        setTimeout(() => {
            element.style.animation = `${animationType} 0.5s ease-in-out`;
        }, 10);
    },

    /**
     * 결과 카드 강조 효과
     */
    highlightResultCard(cardId) {
        const card = document.getElementById(cardId);
        if (card) {
            card.classList.add('highlight');
            setTimeout(() => {
                card.classList.remove('highlight');
            }, 2000);
        }
    }
});
