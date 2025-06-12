/**
 * CSS 최적화 및 중복 검사 도구
 * PUBG 네비게이션 프로젝트의 CSS 성능 향상을 위한 유틸리티
 */

window.CSSOptimizer = {
    
    /**
     * 로드된 CSS 파일들 분석
     */
    analyzeCSSFiles() {
        console.group('🎨 CSS Files Analysis');
        
        const stylesheets = Array.from(document.styleSheets);
        const cssData = {
            files: [],
            totalRules: 0,
            duplicateSelectors: [],
            unusedSelectors: [],
            largeFiles: []
        };
        
        stylesheets.forEach((sheet, index) => {
            try {
                const href = sheet.href || 'inline';
                const rules = Array.from(sheet.cssRules || sheet.rules || []);
                
                const fileInfo = {
                    index,
                    href: href.split('/').pop() || 'inline',
                    fullHref: href,
                    rulesCount: rules.length,
                    selectors: rules.map(rule => rule.selectorText).filter(Boolean)
                };
                
                cssData.files.push(fileInfo);
                cssData.totalRules += rules.length;
                
                console.log(`${index + 1}. ${fileInfo.href}`);
                console.log(`   Rules: ${fileInfo.rulesCount}`);
                console.log(`   Source: ${href === 'inline' ? 'Inline styles' : href}`);
                
            } catch (e) {
                console.warn(`CSS 파일 ${index + 1} 분석 실패 (CORS 제한일 수 있음):`, e.message);
            }
        });
        
        console.log(`\n📊 총 ${cssData.files.length}개 파일, ${cssData.totalRules}개 규칙`);
        console.groupEnd();
        
        return cssData;
    },
    
    /**
     * 중복 선택자 찾기
     */
    findDuplicateSelectors() {
        console.group('🔍 Duplicate Selectors Check');
        
        const selectorCounts = {};
        const duplicates = [];
        
        try {
            Array.from(document.styleSheets).forEach(sheet => {
                try {
                    Array.from(sheet.cssRules || []).forEach(rule => {
                        if (rule.selectorText) {
                            const selector = rule.selectorText.trim();
                            selectorCounts[selector] = (selectorCounts[selector] || 0) + 1;
                        }
                    });
                } catch (e) {
                    // CORS 제한으로 접근 불가능한 외부 CSS
                }
            });
            
            Object.entries(selectorCounts).forEach(([selector, count]) => {
                if (count > 1) {
                    duplicates.push({ selector, count });
                }
            });
            
            duplicates.sort((a, b) => b.count - a.count);
            
            if (duplicates.length > 0) {
                console.log('🚨 중복된 선택자들:');
                duplicates.slice(0, 10).forEach(({ selector, count }) => {
                    console.log(`  ${count}번: ${selector}`);
                });
                
                if (duplicates.length > 10) {
                    console.log(`  ... 및 ${duplicates.length - 10}개 더`);
                }
            } else {
                console.log('✅ 중복된 선택자 없음');
            }
            
        } catch (e) {
            console.warn('중복 선택자 검사 실패:', e.message);
        }
        
        console.groupEnd();
        return duplicates;
    },
    
    /**
     * 사용되지 않는 CSS 규칙 찾기
     */
    findUnusedSelectors() {
        console.group('🗑️ Unused Selectors Check');
        
        const unusedSelectors = [];
        let checkedCount = 0;
        let foundCount = 0;
        
        try {
            Array.from(document.styleSheets).forEach(sheet => {
                try {
                    Array.from(sheet.cssRules || []).forEach(rule => {
                        if (rule.selectorText) {
                            checkedCount++;
                            const selector = rule.selectorText.trim();
                            
                            // 의사 클래스 제거 (정확한 매칭을 위해)
                            const cleanSelector = selector
                                .replace(/:hover|:focus|:active|:visited|:before|:after/g, '')
                                .trim();
                            
                            try {
                                const elements = document.querySelectorAll(cleanSelector);
                                if (elements.length === 0) {
                                    unusedSelectors.push(selector);
                                } else {
                                    foundCount++;
                                }
                            } catch (e) {
                                // 복잡한 선택자나 잘못된 선택자는 무시
                            }
                        }
                    });
                } catch (e) {
                    // CORS 제한
                }
            });
            
            console.log(`📊 검사한 선택자: ${checkedCount}개`);
            console.log(`✅ 사용 중인 선택자: ${foundCount}개`);
            console.log(`❌ 사용되지 않는 선택자: ${unusedSelectors.length}개`);
            
            if (unusedSelectors.length > 0) {
                console.log('\n🗑️ 사용되지 않는 선택자들 (일부):');
                unusedSelectors.slice(0, 10).forEach(selector => {
                    console.log(`  ${selector}`);
                });
                
                if (unusedSelectors.length > 10) {
                    console.log(`  ... 및 ${unusedSelectors.length - 10}개 더`);
                }
            }
            
        } catch (e) {
            console.warn('미사용 선택자 검사 실패:', e.message);
        }
        
        console.groupEnd();
        return unusedSelectors;
    },
    
    /**
     * CSS 성능 메트릭 분석
     */
    analyzePerformanceMetrics() {
        console.group('⚡ CSS Performance Metrics');
        
        const metrics = {
            totalStylesheets: document.styleSheets.length,
            inlineStyles: document.querySelectorAll('[style]').length,
            cssVariables: 0,
            complexSelectors: 0,
            importRules: 0
        };
        
        // CSS 변수 개수 확인
        try {
            const rootStyles = getComputedStyle(document.documentElement);
            const rootStyleText = rootStyles.cssText || '';
            metrics.cssVariables = (rootStyleText.match(/--[\w-]+/g) || []).length;
        } catch (e) {
            console.warn('CSS 변수 카운트 실패:', e.message);
        }
        
        // 복잡한 선택자 및 @import 규칙 확인
        try {
            Array.from(document.styleSheets).forEach(sheet => {
                try {
                    Array.from(sheet.cssRules || []).forEach(rule => {
                        if (rule.type === CSSRule.IMPORT_RULE) {
                            metrics.importRules++;
                        }
                        
                        if (rule.selectorText) {
                            // 복잡한 선택자 감지 (3개 이상의 조합자 또는 긴 선택자)
                            const complexity = (rule.selectorText.match(/[>+~\s]/g) || []).length;
                            if (complexity >= 3 || rule.selectorText.length > 50) {
                                metrics.complexSelectors++;
                            }
                        }
                    });
                } catch (e) {
                    // CORS 제한
                }
            });
        } catch (e) {
            console.warn('성능 메트릭 분석 실패:', e.message);
        }
        
        console.log('📊 성능 메트릭:');
        console.log(`  스타일시트 파일: ${metrics.totalStylesheets}개`);
        console.log(`  인라인 스타일: ${metrics.inlineStyles}개`);
        console.log(`  CSS 변수: ${metrics.cssVariables}개`);
        console.log(`  복잡한 선택자: ${metrics.complexSelectors}개`);
        console.log(`  @import 규칙: ${metrics.importRules}개`);
        
        // 성능 권장사항
        console.log('\n💡 성능 권장사항:');
        
        if (metrics.importRules > 0) {
            console.log(`  ⚠️ @import 규칙 ${metrics.importRules}개 발견 - 링크 태그로 변경 권장`);
        }
        
        if (metrics.inlineStyles > 10) {
            console.log(`  ⚠️ 인라인 스타일 ${metrics.inlineStyles}개 - CSS 파일로 이동 권장`);
        }
        
        if (metrics.complexSelectors > 20) {
            console.log(`  ⚠️ 복잡한 선택자 ${metrics.complexSelectors}개 - 단순화 권장`);
        }
        
        if (metrics.totalStylesheets > 5) {
            console.log(`  ⚠️ 스타일시트 ${metrics.totalStylesheets}개 - 번들링 고려`);
        }
        
        console.groupEnd();
        return metrics;
    },
    
    /**
     * CSS 최적화 제안사항 생성
     */
    generateOptimizationSuggestions() {
        console.group('🚀 CSS Optimization Suggestions');
        
        const duplicates = this.findDuplicateSelectors();
        const unused = this.findUnusedSelectors();
        const metrics = this.analyzePerformanceMetrics();
        
        const suggestions = [];
        
        if (duplicates.length > 0) {
            suggestions.push({
                priority: 'high',
                category: 'duplicates',
                message: `${duplicates.length}개의 중복 선택자 제거`,
                details: duplicates.slice(0, 5).map(d => d.selector)
            });
        }
        
        if (unused.length > 10) {
            suggestions.push({
                priority: 'medium',
                category: 'unused',
                message: `${unused.length}개의 미사용 선택자 제거`,
                details: unused.slice(0, 5)
            });
        }
        
        if (metrics.complexSelectors > 20) {
            suggestions.push({
                priority: 'medium',
                category: 'complexity',
                message: '복잡한 선택자 단순화',
                details: ['긴 선택자 체인을 클래스로 대체', 'BEM 방법론 적용 고려']
            });
        }
        
        if (metrics.totalStylesheets > 5) {
            suggestions.push({
                priority: 'low',
                category: 'bundling',
                message: 'CSS 파일 번들링',
                details: ['여러 CSS 파일을 하나로 결합', '크리티컬 CSS 인라인화']
            });
        }
        
        console.log('💡 최적화 제안사항:');
        suggestions.forEach((suggestion, index) => {
            const priorityIcon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }[suggestion.priority];
            
            console.log(`\n${index + 1}. ${priorityIcon} ${suggestion.message}`);
            if (suggestion.details.length > 0) {
                suggestion.details.forEach(detail => {
                    console.log(`   - ${detail}`);
                });
            }
        });
        
        console.groupEnd();
        return suggestions;
    },
    
    /**
     * 전체 CSS 분석 실행
     */
    runFullAnalysis() {
        console.clear();
        console.log('🎨 CSS 최적화 분석 시작');
        console.log('=' .repeat(50));
        
        const cssData = this.analyzeCSSFiles();
        const suggestions = this.generateOptimizationSuggestions();
        
        console.log('=' .repeat(50));
        console.log('✅ CSS 분석 완료!');
        
        return {
            cssData,
            suggestions,
            timestamp: new Date().toISOString()
        };
    }
};

// 전역 함수로 등록
window.optimizeCSS = () => CSSOptimizer.runFullAnalysis();

// 페이지 로드 시 알림
window.addEventListener('load', function() {
    console.log('🎨 CSS 최적화 도구가 로드되었습니다.');
    console.log('💡 사용법: optimizeCSS() - 전체 CSS 분석 실행');
});
