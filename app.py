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

    run_button = st.button("🚀 팩트체크 실행", type="primary")

    if run_button:
        if not input_text.strip():
            st.warning("텍스트를 입력해주세요.")
            st.stop()

        initial_state = create_initial_state(input_text)

        # st.status(): 진행 상태를 실시간으로 표시하는 Streamlit 컨테이너.
        # expanded=True → 실행 중에는 펼쳐진 상태로 노드 진행 상황을 보여줌.
        status = st.status("파이프라인 실행 중...", expanded=True)

        # 최종 결과를 누적할 딕셔너리.
        # pipeline.stream()은 노드별 출력을 조각(chunk)으로 yield하기 때문에
        # 모든 노드가 끝난 후 전체 결과를 쓰려면 여기에 계속 합쳐야 함.
        result: dict = {}

        with status:
            # pipeline.stream(): invoke()와 달리 노드 하나가 완료될 때마다 즉시 yield.
            # chunk 구조: { "노드이름": { 상태필드: 값, ... } }
            # 예: { "claim_extractor": { "claims": ["주장1", "주장2"] } }
            for chunk in pipeline.stream(initial_state):

                # chunk는 항상 키가 하나 — 방금 완료된 노드 이름
                node_name = list(chunk.keys())[0]

                # 해당 노드의 출력(부분 상태)을 전체 결과에 누적.
                # update()를 쓰는 이유: 노드마다 서로 다른 필드를 리턴하므로
                # 덮어쓰지 않고 계속 합쳐야 최종 상태가 완성됨.
                result.update(chunk[node_name])

                # 노드 완료 시 status 라벨을 실시간으로 갱신 → UI에 즉시 반영
                status.update(label=f"✅ {node_name} 완료")

        # 모든 노드 완료 후 status를 "complete" 상태로 전환 (초록색 체크 아이콘)
        status.update(label="분석 완료!", state="complete", expanded=False)

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
