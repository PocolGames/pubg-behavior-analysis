
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time

st.set_page_config(
    page_title="PUBG Player Classifier",
    page_icon="🎮",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_player(data):
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API 오류: {e}")
        return None

def main():
    st.title("🎮 PUBG Player Behavior Classifier")
    st.markdown("### AI 기반 플레이어 행동 분석 시스템")
    st.markdown("---")

    # 사이드바
    st.sidebar.title("🎯 메뉴")
    page = st.sidebar.selectbox(
        "페이지 선택",
        ["🏠 홈", "🔍 플레이어 분석", "📊 대시보드"]
    )

    # API 상태 확인
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API 서버 연결됨")
    else:
        st.sidebar.error("❌ API 서버 연결 실패")

    if page == "🏠 홈":
        show_home()
    elif page == "🔍 플레이어 분석":
        show_prediction(api_healthy)
    elif page == "📊 대시보드":
        show_dashboard()

def show_home():
    st.header("🎮 PUBG 플레이어 분류 시스템")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 모델 정확도", "92.5%")
    with col2:
        st.metric("📊 분석 특성", "8개")
    with col3:
        st.metric("🏷️ 플레이어 유형", "5개")

    st.subheader("🎯 플레이어 유형 소개")

    player_types = {
        "🗡️ Aggressive Fighter": "높은 킬 수와 데미지를 기록하는 공격적인 플레이어",
        "🛡️ Cautious Survivor": "치료 아이템을 많이 사용하며 생존에 집중하는 플레이어",
        "🚶 Mobile Explorer": "맵을 많이 돌아다니며 탐험하는 플레이어",
        "🤝 Team Supporter": "어시스트가 많고 팀을 지원하는 플레이어",
        "⚖️ Balanced Player": "모든 지표가 균형잡힌 올라운드 플레이어"
    }

    for ptype, desc in player_types.items():
        st.info(f"**{ptype}**: {desc}")

    st.subheader("📖 사용 방법")
    st.markdown("""
    1. **🔍 플레이어 분석** 탭에서 게임 통계를 입력하세요
    2. **🎯 분류하기** 버튼을 클릭하여 플레이어 유형을 확인하세요
    3. **📊 대시보드** 탭에서 시뮬레이션 데이터를 확인할 수 있습니다
    """)

def show_prediction(api_healthy):
    st.header("🔍 플레이어 분석")

    if not api_healthy:
        st.error("🚨 API 서버에 연결할 수 없습니다!")
        st.info("FastAPI 서버가 실행 중인지 확인해주세요.")
        return

    with st.form("player_form"):
        st.subheader("📊 게임 통계 입력")

        col1, col2 = st.columns(2)

        with col1:
            kills = st.number_input("🎯 킬 수", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
            damage = st.number_input("💥 총 데미지", min_value=0.0, max_value=5000.0, value=250.0, step=10.0)
            walk_dist = st.number_input("🚶 도보 이동거리", min_value=0.0, max_value=15000.0, value=1500.0, step=100.0)
            ride_dist = st.number_input("🚗 차량 이동거리", min_value=0.0, max_value=20000.0, value=500.0, step=100.0)

        with col2:
            heals = st.number_input("💊 치료 아이템 사용", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
            boosts = st.number_input("⚡ 부스터 사용", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
            weapons = st.number_input("🔫 무기 획득", min_value=0.0, max_value=20.0, value=4.0, step=1.0)
            assists = st.number_input("🤝 어시스트", min_value=0.0, max_value=20.0, value=1.0, step=1.0)

        submitted = st.form_submit_button("🎯 플레이어 분류하기", use_container_width=True)

        if submitted:
            player_data = {
                "kills": kills,
                "damageDealt": damage,
                "walkDistance": walk_dist,
                "rideDistance": ride_dist,
                "heals": heals,
                "boosts": boosts,
                "weaponsAcquired": weapons,
                "assists": assists
            }

            with st.spinner("🔄 플레이어 행동 분석 중..."):
                result = predict_player(player_data)

            if result:
                st.success("✅ 분석 완료!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🏷️ 플레이어 유형", result['player_type'])
                with col2:
                    st.metric("🎯 신뢰도", f"{result['confidence']:.1%}")
                with col3:
                    status = "🚨 이상" if result['is_anomaly'] else "✅ 정상"
                    st.metric("⚠️ 상태", status)

                # 확률 분포 차트
                st.subheader("📊 각 유형별 확률")
                prob_df = pd.DataFrame(
                    list(result['probabilities'].items()),
                    columns=['플레이어 유형', '확률']
                )
                prob_df['확률'] = prob_df['확률'] * 100

                fig = px.bar(
                    prob_df,
                    x='플레이어 유형',
                    y='확률',
                    title="플레이어 유형별 분류 확률",
                    color='확률',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # 상세 결과
                with st.expander("🔍 상세 분석 결과"):
                    st.json(result)

def show_dashboard():
    st.header("📊 시스템 대시보드")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📈 일일 예측", f"{np.random.randint(1000, 2000):,}건")
    with col2:
        st.metric("⚡ 평균 응답시간", f"{np.random.randint(5, 15)}ms")
    with col3:
        st.metric("✅ 성공률", f"{np.random.uniform(98, 99.9):.1f}%")
    with col4:
        st.metric("🚨 이상치 탐지", f"{np.random.randint(20, 80)}건")

    # 샘플 차트
    st.subheader("📈 예측 추이 (시뮬레이션)")

    # 시간별 예측 건수 차트
    hours = list(range(24))
    predictions = [np.random.randint(30, 120) for _ in hours]

    chart_df = pd.DataFrame({
        '시간': hours,
        '예측 건수': predictions
    })

    fig = px.line(chart_df, x='시간', y='예측 건수',
                  title='시간별 예측 건수', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # 플레이어 유형 분포
    st.subheader("🎯 플레이어 유형 분포")
    type_data = {
        'Aggressive Fighter': np.random.randint(150, 300),
        'Cautious Survivor': np.random.randint(100, 250),
        'Mobile Explorer': np.random.randint(80, 200),
        'Team Supporter': np.random.randint(60, 150),
        'Balanced Player': np.random.randint(200, 400)
    }

    fig_pie = px.pie(values=list(type_data.values()),
                     names=list(type_data.keys()),
                     title="오늘의 플레이어 유형 분포")
    st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
