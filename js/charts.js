// 차트 및 시각화 관리 클래스
class ChartManager {
    constructor() {
        this.animationDuration = 1000;
        this.previousResults = this.loadPreviousResults();
        this.globalStats = this.loadGlobalStats();
    }

    // 레이더 차트 생성
    createRadarChart(data, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const abilities = [
            { name: '공격력', value: this.normalizeValue(data.kills, 0, 15) },
            { name: '생존력', value: this.normalizeValue(data.heals + data.boosts, 0, 20) },
            { name: '이동성', value: this.normalizeValue(data.walkDistance, 0, 5000) },
            { name: '효율성', value: this.normalizeValue(data.damageDealt, 0, 1500) },
            { name: '지원력', value: this.normalizeValue(data.assists, 0, 10) },
            { name: '장비력', value: this.normalizeValue(data.weaponsAcquired, 0, 15) }
        ];

        const size = 250;
        const center = size / 2;
        const maxRadius = 100;
        const levels = 5;

        let svg = `<svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">`;
        
        // 그리드 생성
        for (let i = 1; i <= levels; i++) {
            const radius = (maxRadius / levels) * i;
            svg += `<circle cx="${center}" cy="${center}" r="${radius}" class="radar-grid" />`;
        }

        // 축 생성
        const angleStep = (2 * Math.PI) / abilities.length;
        abilities.forEach((ability, index) => {
            const angle = angleStep * index;
            const x = center + maxRadius * Math.cos(angle);
            const y = center + maxRadius * Math.sin(angle);
            
            svg += `<line x1="${center}" y1="${center}" x2="${x}" y2="${y}" class="radar-axis" />`;
            
            // 라벨 추가
            const labelX = center + (maxRadius + 20) * Math.cos(angle);
            const labelY = center + (maxRadius + 20) * Math.sin(angle);
            svg += `<text x="${labelX}" y="${labelY}" class="radar-labels">${ability.name}</text>`;
        });

        // 데이터 영역 생성
        let pathData = '';
        let points = '';
        
        abilities.forEach((ability, index) => {
            const angle = angleStep * index;
            const radius = maxRadius * ability.value;
            const x = center + radius * Math.cos(angle);
            const y = center + radius * Math.sin(angle);
            
            if (index === 0) {
                pathData += `M ${x} ${y}`;
            } else {
                pathData += ` L ${x} ${y}`;
            }
            
            points += `<circle cx="${x}" cy="${y}" r="4" class="radar-points" />`;
        });
        
        pathData += ' Z';
        svg += `<path d="${pathData}" class="radar-area" />`;
        svg += points;
        svg += '</svg>';

        container.innerHTML = svg;
    }

    // 값 정규화 (0-1 범위)
    normalizeValue(value, min, max) {
        return Math.min(Math.max((value - min) / (max - min), 0), 1);
    }

    // 애니메이션 카운터
    animateCounter(element, targetValue, duration = 1000, suffix = '') {
        const startValue = 0;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (easeOutCubic)
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const currentValue = startValue + (targetValue - startValue) * easeOutCubic;
            
            if (typeof targetValue === 'number') {
                element.textContent = Math.round(currentValue) + suffix;
            } else {
                element.textContent = targetValue;
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // 프로그레스 링 생성
    createProgressRing(containerId, percentage, label) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const radius = 50;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (percentage / 100) * circumference;

        const html = `
            <div class="progress-ring">
                <svg>
                    <circle cx="60" cy="60" r="${radius}" class="progress-ring-bg"></circle>
                    <circle cx="60" cy="60" r="${radius}" class="progress-ring-fill" 
                            style="stroke-dasharray: ${circumference}; stroke-dashoffset: ${offset};"></circle>
                </svg>
                <div class="progress-ring-text">
                    <span class="animated-counter">${Math.round(percentage)}%</span>
                    <span class="progress-ring-label">${label}</span>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    // 비교 데이터 생성
    createComparisonData(currentResult) {
        const previous = this.previousResults[this.previousResults.length - 1];
        if (!previous) {
            return null;
        }

        const comparisons = [
            {
                title: '신뢰도',
                current: Math.round(currentResult.confidence * 100),
                previous: Math.round(previous.confidence * 100),
                suffix: '%'
            },
            {
                title: '주요 유형 확률',
                current: Math.round(currentResult.probabilities[currentResult.predictedType] * 100),
                previous: Math.round(previous.probabilities[previous.predictedType] * 100),
                suffix: '%'
            }
        ];

        return comparisons.map(comp => ({
            ...comp,
            change: comp.current - comp.previous,
            changeType: comp.current > comp.previous ? 'positive' : 
                       comp.current < comp.previous ? 'negative' : 'neutral'
        }));
    }

    // 비교 섹션 렌더링
    renderComparison(containerId, currentResult) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const comparisons = this.createComparisonData(currentResult);
        
        if (!comparisons) {
            container.innerHTML = `
                <div class="comparison-section">
                    <h4>이전 결과와 비교</h4>
                    <p class="text-muted">이전 분석 결과가 없습니다. 분석을 더 해보세요!</p>
                </div>
            `;
            return;
        }

        let html = `
            <div class="comparison-section">
                <h4>이전 결과와 비교</h4>
                <div class="comparison-grid">
        `;

        comparisons.forEach(comp => {
            const changeIcon = comp.changeType === 'positive' ? '↗' : 
                             comp.changeType === 'negative' ? '↘' : '→';
            
            html += `
                <div class="comparison-item">
                    <div class="comparison-title">${comp.title}</div>
                    <div class="comparison-value">${comp.current}${comp.suffix}</div>
                    <div class="comparison-change change-${comp.changeType}">
                        ${changeIcon} ${Math.abs(comp.change)}${comp.suffix}
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    // 글로벌 통계 대시보드
    renderGlobalStats(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // 시뮬레이션 데이터 (실제로는 서버나 로컬 스토리지에서 가져와야 함)
        const stats = {
            totalAnalyses: 1247 + Math.floor(Math.random() * 100),
            popularType: 'Explorer',
            averageConfidence: 87 + Math.floor(Math.random() * 10),
            distribution: {
                survivor: 35 + Math.floor(Math.random() * 10),
                explorer: 45 + Math.floor(Math.random() * 10), 
                aggressive: 20 + Math.floor(Math.random() * 10)
            }
        };

        const html = `
            <div class="stats-dashboard">
                <h4>글로벌 통계</h4>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-icon">📊</span>
                        <span class="stat-value" data-counter="${stats.totalAnalyses}">0</span>
                        <span class="stat-label">총 분석 수</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-icon">🏆</span>
                        <span class="stat-value">${stats.popularType}</span>
                        <span class="stat-label">인기 유형</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-icon">🎯</span>
                        <span class="stat-value" data-counter="${stats.averageConfidence}">0</span>
                        <span class="stat-label">평균 신뢰도 (%)</span>
                    </div>
                </div>
                <div class="distribution-chart">
                    <div class="distribution-bars">
                        <div class="distribution-bar survivor" style="height: ${stats.distribution.survivor}%;">
                            <span class="distribution-bar-label">Survivor</span>
                        </div>
                        <div class="distribution-bar explorer" style="height: ${stats.distribution.explorer}%;">
                            <span class="distribution-bar-label">Explorer</span>
                        </div>
                        <div class="distribution-bar aggressive" style="height: ${stats.distribution.aggressive}%;">
                            <span class="distribution-bar-label">Aggressive</span>
                        </div>
                    </div>
                    <div class="distribution-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: var(--success-color);"></div>
                            <span>Survivor (${stats.distribution.survivor}%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: var(--accent-color);"></div>
                            <span>Explorer (${stats.distribution.explorer}%)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: var(--danger-color);"></div>
                            <span>Aggressive (${stats.distribution.aggressive}%)</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;

        // 카운터 애니메이션 실행
        setTimeout(() => {
            const counters = container.querySelectorAll('[data-counter]');
            counters.forEach(counter => {
                const target = parseInt(counter.dataset.counter);
                this.animateCounter(counter, target, 1500, counter.textContent.includes('%') ? '%' : '');
            });
        }, 100);
    }

    // 이전 결과 저장
    savePreviousResult(result) {
        this.previousResults.push({
            timestamp: Date.now(),
            ...result
        });

        // 최대 10개까지만 저장
        if (this.previousResults.length > 10) {
            this.previousResults.shift();
        }

        try {
            localStorage.setItem('pubg_previous_results', JSON.stringify(this.previousResults));
        } catch (e) {
            console.warn('로컬 스토리지 저장 실패:', e);
        }
    }

    // 이전 결과 로드
    loadPreviousResults() {
        try {
            const saved = localStorage.getItem('pubg_previous_results');
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            console.warn('이전 결과 로드 실패:', e);
            return [];
        }
    }

    // 글로벌 통계 로드
    loadGlobalStats() {
        try {
            const saved = localStorage.getItem('pubg_global_stats');
            return saved ? JSON.parse(saved) : {
                totalAnalyses: 0,
                typeDistribution: { survivor: 0, explorer: 0, aggressive: 0 }
            };
        } catch (e) {
            console.warn('글로벌 통계 로드 실패:', e);
            return { totalAnalyses: 0, typeDistribution: { survivor: 0, explorer: 0, aggressive: 0 } };
        }
    }

    // 글로벌 통계 업데이트
    updateGlobalStats(result) {
        this.globalStats.totalAnalyses += 1;
        this.globalStats.typeDistribution[result.predictedType] += 1;

        try {
            localStorage.setItem('pubg_global_stats', JSON.stringify(this.globalStats));
        } catch (e) {
            console.warn('글로벌 통계 저장 실패:', e);
        }
    }

    // 전체 차트 렌더링
    renderAllCharts(inputData, result) {
        // 레이더 차트
        this.createRadarChart(inputData, 'radarChart');
        
        // 신뢰도 프로그레스 링
        this.createProgressRing('confidenceRing', result.confidence * 100, '신뢰도');
        
        // 비교 데이터
        this.renderComparison('comparisonContainer', result);
        
        // 글로벌 통계
        this.renderGlobalStats('globalStatsContainer');
        
        // 결과 저장
        this.savePreviousResult(result);
        this.updateGlobalStats(result);
    }
}

// 전역 차트 매니저 인스턴스
const chartManager = new ChartManager();