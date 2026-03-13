"""
Streamlit 기반 뉴스 팩트체크 에이전트 UI.

실행 방법:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import main
from main import create_initial_state


# ---------------------------------------------------------------------------
# 파이프라인 초기화 (앱 기동 시 1회)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="파이프라인 초기화 중...")
def load_pipeline():
    """파이프라인을 초기화하고 컴파일된 그래프를 반환합니다.

    Returns:
        컴파일된 LangGraph CompiledGraph 인스턴스.
    """
    main.init_pipeline()
    return main.graph


# ---------------------------------------------------------------------------
# UI 레이아웃
# ---------------------------------------------------------------------------

def render_claim_results(result: dict) -> None:
    """주장 추출 및 검색 결과를 expander로 렌더링합니다.

    Args:
        result: 파이프라인 실행 결과 딕셔너리.
    """
    with st.expander("🔍 추출된 주장 목록", expanded=False):
        claims = result.get("claims", [])
        if claims:
            for idx, claim in enumerate(claims, start=1):
                st.write(f"**{idx}.** {claim}")
        else:
            st.write("_추출된 주장이 없습니다._")

    with st.expander("📡 검색 결과 (근거)", expanded=False):
        search_results = result.get("search_results", [])
        if search_results:
            for item in search_results:
                st.markdown(f"**주장:** {item['claim']}")
                st.caption(item.get("evidence", "(근거 없음)"))
                st.divider()
        else:
            st.write("_검색 결과가 없습니다._")


def render_judgment_results(result: dict) -> None:
    """판단 결과와 LLM Judge 평가를 expander로 렌더링합니다.

    Args:
        result: 파이프라인 실행 결과 딕셔너리.
    """
    verdict_icon = {"TRUE": "✅", "FALSE": "❌", "UNVERIFIABLE": "❓"}

    with st.expander("⚖️ 판단 결과 (Fact Judge)", expanded=False):
        judgment_results = result.get("judgment_results", [])
        if judgment_results:
            for item in judgment_results:
                verdict = item.get("verdict", "UNVERIFIABLE")
                icon = verdict_icon.get(verdict, "❓")
                st.markdown(f"**{icon} {item['claim']}**")
                st.write(f"판정: `{verdict}`")
                st.caption(item.get("reasoning", "-"))
                st.divider()
        else:
            st.write("_판단 결과가 없습니다._")

    with st.expander("🧑‍⚖️ LLM Judge 평가", expanded=False):
        score = result.get("judge_score", 0.0)
        feedback = result.get("judge_feedback", "")
        retries = result.get("correction_retries", 0)

        col1, col2 = st.columns(2)
        col1.metric("신뢰도 점수", f"{score:.2f}")
        col2.metric("Self-Correction 횟수", retries)

        if feedback:
            st.info(f"**피드백:** {feedback}")


def render_error_banner(result: dict) -> None:
    """에러 상태가 있으면 경고 배너를 표시합니다.

    Args:
        result: 파이프라인 실행 결과 딕셔너리.
    """
    error = result.get("error")
    if error:
        st.warning(f"⚠️ 처리 중 오류가 발생했습니다: {error}")


# ---------------------------------------------------------------------------
# 메인 앱
# ---------------------------------------------------------------------------

def main_app() -> None:
    """Streamlit 앱 메인 렌더링 함수."""
    st.set_page_config(
        page_title="뉴스 팩트체크 에이전트",
        page_icon="📰",
        layout="wide",
    )

    st.title("📰 뉴스 팩트체크 에이전트")
    st.caption("LangGraph + DSPy 기반 자동 팩트체크 파이프라인")

    pipeline = load_pipeline()

    # 입력 영역
    st.subheader("팩트체크할 텍스트를 입력하세요")
    input_text = st.text_area(
        label="뉴스 또는 텍스트",
        height=180,
        placeholder="예: BTS는 2020년 한국 가수 최초로 빌보드 핫100 1위를 달성했다...",
        label_visibility="collapsed",
    )

    run_button = st.button("🚀 팩트체크 실행", type="primary", disabled=not input_text.strip())

    if run_button and input_text.strip():
        initial_state = create_initial_state(input_text)

        with st.spinner("파이프라인 실행 중..."):
            result = pipeline.invoke(initial_state)

        st.success("분석 완료!")
        render_error_banner(result)

        # 노드별 결과 expander
        render_claim_results(result)
        render_judgment_results(result)

        # 최종 리포트
        st.subheader("📋 최종 팩트체크 리포트")
        final_report = result.get("final_report", "리포트를 생성하지 못했습니다.")
        st.markdown(final_report)


if __name__ == "__main__":
    main_app()
else:
    # streamlit run app.py 로 실행될 때
    main_app()
