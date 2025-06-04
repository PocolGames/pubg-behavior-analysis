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
     * 예측 결과 업데이트 (JSON 데이터 기반)
     */
    updatePredictionResults(prediction, predictionTime) {
        const clusterInfo = this.getClusterInfo()[prediction.predictedCluster];
        
        // 예측 요약 업데이트
        this.updatePredictionSummary(clusterInfo, prediction);
        
        // 분석 세부사항 업데이트
        this.updateAnalysisDetails(prediction, predictionTime);
        
        // 시각적 효과 적용
        this.applyResultAnimations();
    },

    /**
     * 예측 요약 업데이트
     */
    updatePredictionSummary(clusterInfo, prediction) {
        const summary = document.getElementById('predictionSummary');
        const typeName = document.getElementById('predictedTypeName');
        const typeDescription = document.getElementById('predictedTypeDescription');
        const confidenceScore = document.getElementById('confidenceScore');
        
        if (summary && typeName && typeDescription && confidenceScore) {
            summary.style.display = 'block';
            
            // 타입 정보 업데이트
            typeName.textContent = clusterInfo.name;
            typeDescription.textContent = clusterInfo.description;
            confidenceScore.textContent = prediction.confidence.toFixed(1) + '%';
            
            // 신뢰도에 따른 시각적 피드백
            this.updateConfidenceVisual(confidenceScore, prediction.confidence);
            
            // 아이콘 업데이트
            this.updateTypeIcon(summary, clusterInfo);
            
            // 애니메이션 효과
            this.applyAnimation(summary, 'slideInUp');
        }
    },

    /**
     * 신뢰도 시각적 업데이트
     */
    updateConfidenceVisual(element, confidence) {
        const confidenceLevel = this.getConfidenceLevel(confidence);
        const colors = {
            high: '#56ab2f',
            medium: '#ff6b35', 
            low: '#dc3545'
        };
        
        element.style.color = colors[confidenceLevel] || '#666';
        element.style.fontWeight = confidence > 80 ? 'bold' : 'normal';
        
        // 신뢰도 배지 업데이트
        const badge = element.parentNode.querySelector('.confidence-badge');
        if (badge) {
            badge.textContent = confidenceLevel === 'high' ? '높음' : 
                               confidenceLevel === 'medium' ? '보통' : '낮음';
            badge.className = `confidence-badge ${confidenceLevel}`;
        }
    },

    /**
     * 타입 아이콘 업데이트
     */
    updateTypeIcon(container, clusterInfo) {
        const typeIcon = container.querySelector('.type-icon i');
        if (typeIcon) {
            typeIcon.style.color = clusterInfo.color;
            typeIcon.className = clusterInfo.icon;
            
            // 맥박 효과
            typeIcon.style.animation = 'pulse 2s infinite';
        }
        
        // 배경 색상 업데이트
        const iconContainer = container.querySelector('.type-icon');
        if (iconContainer) {
            iconContainer.style.background = `linear-gradient(135deg, ${clusterInfo.color}20, ${clusterInfo.color}40)`;
        }
    },

    /**
     * 분석 세부사항 업데이트 (JSON 기반 확장 분석)
     */
    updateAnalysisDetails(prediction, predictionTime) {
        const details = document.getElementById('analysisDetails');
        if (!details) return;
        
        details.style.display = 'block';
        
        // 기본 분석 정보
        this.updateBasicAnalysis(prediction.analysis);
        
        // 고급 분석 정보 (JSON 데이터 기반)
        this.updateAdvancedAnalysis(prediction);
        
        // 예측 처리 시간
        const predictionTimeEl = document.getElementById('predictionTime');
        if (predictionTimeEl) {
            predictionTimeEl.textContent = predictionTime + '초';
        }
        
        // 애니메이션 효과
        this.applyAnimation(details, 'fadeInUp');
    },

    /**
     * 기본 분석 정보 업데이트
     */
    updateBasicAnalysis(analysis) {
        const fields = {
            'strongestFeature': analysis.strongestFeature,
            'playStyle': analysis.playStyle,
            'similarPlayers': analysis.similarPlayers
        };
        
        Object.entries(fields).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                
                // 중요 정보는 강조
                if (id === 'strongestFeature') {
                    element.style.fontWeight = 'bold';
                    element.style.color = '#ff6b35';
                }
            }
        });
    },

    /**
     * 고급 분석 정보 업데이트
     */
    updateAdvancedAnalysis(prediction) {
        // 추천 액션 업데이트
        this.updateRecommendedActions(prediction.analysis.recommendedActions);
        
        // 기술적 인사이트 업데이트
        this.updateTechnicalInsights(prediction.analysis.technicalInsights);
        
        // 신뢰도 레벨 표시
        this.updateConfidenceLevelIndicator(prediction.analysis.confidenceLevel);
    },

    /**
     * 추천 액션 업데이트
     */
    updateRecommendedActions(actions) {
        const container = document.getElementById('recommendedActions');
        if (!container || !actions) return;
        
        container.innerHTML = '';
        actions.forEach((action, index) => {
            const actionEl = document.createElement('div');
            actionEl.className = 'recommended-action';
            actionEl.innerHTML = `
                <i class="fas fa-lightbulb"></i>
                <span>${action}</span>
            `;
            
            // 순차적 애니메이션
            setTimeout(() => {
                container.appendChild(actionEl);
                this.applyAnimation(actionEl, 'slideInLeft');
            }, index * 200);
        });
    },

    /**
     * 기술적 인사이트 업데이트
     */
    updateTechnicalInsights(insights) {
        const container = document.getElementById('technicalInsights');
        if (!container || !insights) return;
        
        container.innerHTML = '';
        insights.forEach((insight, index) => {
            const insightEl = document.createElement('div');
            insightEl.className = 'technical-insight';
            insightEl.innerHTML = `
                <i class="fas fa-chart-line"></i>
                <span>${insight}</span>
            `;
            
            // 순차적 애니메이션
            setTimeout(() => {
                container.appendChild(insightEl);
                this.applyAnimation(insightEl, 'slideInRight');
            }, index * 200);
        });
    },

    /**
     * 신뢰도 레벨 인디케이터 업데이트
     */
    updateConfidenceLevelIndicator(confidenceLevel) {
        const indicator = document.getElementById('confidenceLevelIndicator');
        if (!indicator) return;
        
        const levelInfo = {
            high: { text: '높은 신뢰도', color: '#56ab2f', icon: 'fas fa-check-circle' },
            medium: { text: '보통 신뢰도', color: '#ff6b35', icon: 'fas fa-info-circle' },
            low: { text: '낮은 신뢰도', color: '#dc3545', icon: 'fas fa-exclamation-circle' }
        };
        
        const info = levelInfo[confidenceLevel] || levelInfo.medium;
        
        indicator.innerHTML = `
            <i class="${info.icon}" style="color: ${info.color}"></i>
            <span style="color: ${info.color}">${info.text}</span>
        `;
    },

    /**
     * 차트 업데이트 (JSON 데이터 기반)
     */
    updateCharts(formData, prediction) {
        this.updateProbabilityChart(prediction.probabilities);
        this.updateRadarChart(formData, prediction.normalizedFeatures);
    },

    /**
     * 확률 분포 차트 업데이트 (향상된 버전)
     */
    updateProbabilityChart(probabilities) {
        const chart = this.charts.get('probability');
        if (!chart) return;
        
        const clusterInfo = this.getClusterInfo();
        
        // 데이터 업데이트
        chart.data.datasets[0].data = probabilities;
        
        // 색상 업데이트 (예측된 클러스터 강조)
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        chart.data.datasets[0].backgroundColor = Object.values(clusterInfo).map((info, index) => 
            index === maxIndex ? info.color : info.color + '60'
        );
        chart.data.datasets[0].borderColor = Object.values(clusterInfo).map(info => info.color);
        
        // 부드러운 애니메이션으로 업데이트
        chart.update('active');
        
        // 최고 확률 막대 강조
        setTimeout(() => {
            this.highlightMaxProbabilityBar(maxIndex);
        }, 500);
    },

    /**
     * 최고 확률 막대 강조
     */
    highlightMaxProbabilityBar(index) {
        const canvas = document.getElementById('probabilityChart');
        if (canvas) {
            // 시각적 강조 효과 (옵션)
            canvas.style.filter = 'brightness(1.1)';
            setTimeout(() => {
                canvas.style.filter = 'brightness(1)';
            }, 1000);
        }
    },

    /**
     * 레이더 차트 업데이트 (JSON 특성 기반)
     */
    updateRadarChart(formData, normalizedFeatures) {
        const chart = this.charts.get('radar');
        if (!chart || !normalizedFeatures) return;
        
        const features = this.getFeatureDefinitions();
        const radarData = [];
        
        // JSON 특성 정의 기반으로 레이더 데이터 생성
        features.slice(0, 8).forEach((feature, index) => {
            if (normalizedFeatures[index] !== undefined) {
                radarData.push(normalizedFeatures[index]);
            } else {
                // 백업 계산
                const value = formData[feature.name] || 0;
                const range = feature.max - feature.min;
                const normalized = range > 0 ? ((value - feature.min) / range) * 100 : 0;
                radarData.push(Math.max(0, Math.min(100, normalized)));
            }
        });
        
        // 차트 업데이트
        chart.data.datasets[0].data = radarData;
        chart.data.labels = features.slice(0, 8).map(f => f.displayName);
        
        // 동적 색상 (가장 높은 특성에 따라)
        const maxValue = Math.max(...radarData);
        const intensity = maxValue / 100;
        const color = `rgba(102, 126, 234, ${0.3 + intensity * 0.4})`;
        
        chart.data.datasets[0].backgroundColor = color;
        chart.data.datasets[0].borderColor = '#667eea';
        
        chart.update('active');
    },

    /**
     * 예측 상태 업데이트 (향상된 버전)
     */
    updatePredictionStatus(status, type) {
        const statusEl = document.getElementById('predictionStatus');
        if (!statusEl) return;
        
        const indicator = statusEl.querySelector('.status-indicator');
        if (indicator) {
            indicator.textContent = status;
            indicator.className = `status-indicator ${type}`;
            
            // 상태별 아이콘 추가
            const icons = {
                processing: 'fas fa-spinner fa-spin',
                success: 'fas fa-check-circle',
                error: 'fas fa-exclamation-circle',
                '': 'fas fa-clock'
            };
            
            const icon = indicator.querySelector('i') || document.createElement('i');
            icon.className = icons[type] || icons[''];
            
            if (!indicator.querySelector('i')) {
                indicator.prepend(icon);
            }
        }
    },

    /**
     * 폼 초기화 (JSON 데이터 기반)
     */
    resetForm() {
        const form = document.getElementById('playerForm');
        if (!form) return;
        
        form.reset();
        
        // JSON 특성 정의 기반 기본값 설정
        const features = this.getFeatureDefinitions();
        features.forEach(feature => {
            const input = document.querySelector(`input[name="${feature.name}"]`);
            if (input) {
                // 특성별 합리적 기본값 설정
                let defaultValue = feature.min;
                
                if (feature.name === 'walkDistance') defaultValue = 1000;
                else if (feature.name === 'weaponsAcquired') defaultValue = 3;
                else if (feature.name === 'killPlace') defaultValue = 50;
                else if (feature.name === 'boosts') defaultValue = 1;
                else if (feature.name === 'heals') defaultValue = 1;
                
                input.value = defaultValue;
            }
        });
        
        // UI 상태 초기화
        this.hidePredictionResults();
        this.createPlaceholderCharts();
        this.updateInputStyles();
        
        if (window.App) {
            App.showNotification('폼이 초기화되었습니다.', 'info');
        }
    },

    /**
     * 샘플 플레이어 로드 (JSON 데이터 기반)
     */
    loadSamplePlayer(playerType) {
        const samplePlayers = this.getSamplePlayers();
        const sampleData = samplePlayers[playerType];
        
        if (!sampleData) {
            if (window.App) {
                App.showNotification('샘플 데이터를 찾을 수 없습니다.', 'error');
            }
            return;
        }
        
        // 폼에 데이터 입력 (애니메이션 효과와 함께)
        const features = this.getFeatureDefinitions();
        let delay = 0;
        
        features.forEach(feature => {
            if (sampleData.hasOwnProperty(feature.name)) {
                setTimeout(() => {
                    const input = document.querySelector(`input[name="${feature.name}"]`);
                    if (input) {
                        input.value = sampleData[feature.name];
                        input.style.transform = 'scale(1.05)';
                        setTimeout(() => {
                            input.style.transform = 'scale(1)';
                        }, 200);
                    }
                }, delay);
                delay += 50;
            }
        });
        
        // 샘플 정보 표시
        this.showSampleInfo(playerType, sampleData);
        
        if (window.App) {
            const sampleInfo = this.getSamplePlayers()[playerType];
            const sampleName = this.isDataLoaded && this.predictionData.samplePlayers[playerType] ? 
                this.predictionData.samplePlayers[playerType].name : 
                `${playerType} 플레이어`;
            
            App.showNotification(`${sampleName} 샘플 데이터가 로드되었습니다.`, 'success');
        }
        
        // 자동 예측 실행
        setTimeout(() => {
            this.predictPlayerType();
        }, delay + 500);
    },

    /**
     * 샘플 정보 표시
     */
    showSampleInfo(playerType, sampleData) {
        const infoContainer = document.getElementById('sampleInfo');
        if (!infoContainer) return;
        
        let sampleInfo = { name: `${playerType} 플레이어`, description: '샘플 플레이어입니다.' };
        
        if (this.isDataLoaded && this.predictionData.samplePlayers[playerType]) {
            sampleInfo = this.predictionData.samplePlayers[playerType];
        }
        
        infoContainer.innerHTML = `
            <div class="sample-info-card">
                <h4>${sampleInfo.name}</h4>
                <p>${sampleInfo.description}</p>
                <div class="sample-stats">
                    ${Object.entries(sampleData).slice(0, 4).map(([key, value]) => 
                        `<span class="stat">${key}: ${value}</span>`
                    ).join('')}
                </div>
            </div>
        `;
        
        this.applyAnimation(infoContainer, 'slideInDown');
    },

    /**
     * 샘플 메뉴 표시 (향상된 버전)
     */
    showSampleMenu() {
        const sampleSection = document.querySelector('.sample-players');
        if (!sampleSection) return;
        
        // 스크롤 및 강조
        sampleSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // 카드별 순차적 하이라이트
        const sampleCards = sampleSection.querySelectorAll('.sample-player');
        sampleCards.forEach((card, index) => {
            setTimeout(() => {
                card.style.transform = 'scale(1.05)';
                card.style.boxShadow = '0 8px 25px rgba(102, 126, 234, 0.3)';
                
                setTimeout(() => {
                    card.style.transform = 'scale(1)';
                    card.style.boxShadow = '';
                }, 300);
            }, index * 150);
        });
    },

    /**
     * 예측 결과 숨기기
     */
    hidePredictionResults() {
        const elementsToHide = ['predictionSummary', 'analysisDetails', 'sampleInfo'];
        
        elementsToHide.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.style.display = 'none';
            }
        });
        
        this.updatePredictionStatus('대기중', '');
    },

    /**
     * 입력 필드 스타일 업데이트 (JSON 기반)
     */
    updateInputStyles() {
        const features = this.getFeatureDefinitions();
        
        features.forEach(feature => {
            const input = document.querySelector(`input[name="${feature.name}"]`);
            if (!input) return;
            
            const value = parseFloat(input.value) || 0;
            const percentage = ((value - feature.min) / (feature.max - feature.min)) * 100;
            
            // 카테고리별 색상
            const categoryColors = {
                combat: '#dc3545',
                movement: '#667eea',
                survival: '#56ab2f',
                strategy: '#ff6b35',
                teamwork: '#9c27b0'
            };
            
            const color = categoryColors[feature.category] || '#6c757d';
            
            // 진행률에 따른 스타일
            if (percentage > 80) {
                input.style.borderColor = color;
                input.style.borderWidth = '2px';
            } else {
                input.style.borderColor = color + '60';
                input.style.borderWidth = '1px';
            }
            
            // 배경 그라데이션
            const intensity = Math.min(percentage / 100, 1);
            input.style.background = `linear-gradient(90deg, ${color}10 0%, ${color}${Math.floor(intensity * 30 + 10)}% ${percentage}%, transparent ${percentage}%)`;
        });
    },

    /**
     * 결과 애니메이션 적용
     */
    applyResultAnimations() {
        // 순차적 애니메이션
        const elements = [
            document.getElementById('predictionSummary'),
            document.getElementById('analysisDetails')
        ];
        
        elements.forEach((element, index) => {
            if (element) {
                setTimeout(() => {
                    this.applyAnimation(element, 'bounceIn');
                }, index * 300);
            }
        });
    },

    /**
     * 애니메이션 효과 적용 (향상된 버전)
     */
    applyAnimation(element, animationType = 'fadeIn', duration = 0.5) {
        if (!element) return;
        
        element.style.animation = 'none';
        element.style.transform = '';
        
        setTimeout(() => {
            element.style.animation = `${animationType} ${duration}s ease-in-out`;
        }, 10);
        
        // 애니메이션 완료 후 정리
        setTimeout(() => {
            element.style.animation = '';
        }, duration * 1000);
    },

    /**
     * 결과 카드 강조 효과 (향상된 버전)
     */
    highlightResultCard(cardId) {
        const card = document.getElementById(cardId);
        if (!card) return;
        
        card.style.transition = 'all 0.3s ease';
        card.style.transform = 'translateY(-5px)';
        card.style.boxShadow = '0 15px 35px rgba(102, 126, 234, 0.3)';
        
        setTimeout(() => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '';
        }, 2000);
    }
});
