"""
FactCheckAgent 서브그래프 정의.

사실 판단 + LLM Judge + Self-Correction 루프를 하나의 에이전트로 묶습니다.
내부에 조건부 루프가 있어 서브그래프로 분리합니다.
나머지 노드(주장 추출, 근거 검색, 리포트 생성)는 main.py에서 직접 등록합니다.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from config import settings
import nodes
from state import FactCheckState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FactCheckAgent 내부 라우팅
# ---------------------------------------------------------------------------

def _route_in_fact_check_agent(state: FactCheckState) -> str:
    """FactCheckAgent 서브그래프 내부의 라우팅 함수.

    LLM Judge 평가 후 Self-Correction 루프를 계속할지, 에이전트를 종료할지 결정합니다.

    판단 순서:
    1. 에러 상태면 즉시 종료.
    2. 최대 재시도 횟수에 도달하면 종료.
    3. 점수가 기준 이상이면 종료.
    4. 그 외 Self-Correction으로 이동.

    Args:
        state: 현재 파이프라인 상태.

    Returns:
        "self_correction" 또는 END.
    """
    if state.get("error"):
        logger.warning("[FactCheckAgent][router] 에러 상태 감지 → 에이전트 종료.")
        return END

    retries = state.get("correction_retries", 0)
    if retries >= settings.max_correction_retries:
        logger.warning(
            "[FactCheckAgent][router] 최대 재시도 도달(%d/%d) → 에이전트 종료.",
            retries,
            settings.max_correction_retries,
        )
        return END

    score = state.get("judge_score", 0.0)
    if score >= settings.correction_threshold:
        logger.info(
            "[FactCheckAgent][router] 점수 기준 충족(%.2f ≥ %.2f) → 에이전트 종료.",
            score,
            settings.correction_threshold,
        )
        return END

    logger.info(
        "[FactCheckAgent][router] 점수 미달(%.2f < %.2f) → Self-Correction 발동.",
        score,
        settings.correction_threshold,
    )
    return "self_correction"


# ---------------------------------------------------------------------------
# FactCheckAgent 서브그래프
# ---------------------------------------------------------------------------

def build_fact_check_agent() -> Any:
    """사실 검증 에이전트 서브그래프를 빌드하고 컴파일합니다.

    담당 노드:
        fact_judge     : 검색된 근거를 바탕으로 주장의 사실 여부 판단.
        llm_judge      : 판단 결과의 품질을 0.0~1.0 점수로 평가.
        self_correction: 점수 미달 시 재시도 카운터 증가 후 fact_judge로 복귀.

    내부 흐름:
        START → fact_judge → llm_judge → (조건부) self_correction → fact_judge (루프)
                                       → (조건부) END

    Returns:
        컴파일된 서브그래프 인스턴스.
    """
    graph = StateGraph(FactCheckState)

    # 주장과 근거를 바탕으로 사실 판단.
    #graph.add_node("fact_judge", nodes.fact_judge_node)

    # 양측 키보드 배틀 준비.
    graph.add_node( "debate", nodes.debate_node )

    # 키보드 배틀 결과.
    graph.add_node( "debate_judge", nodes.debate_judge_node)

    # LLM as a Judge
    # 사실과 얼마나 부합하는지(0~1), 그리고 개선을 위한 피드백.
    graph.add_node("llm_judge", nodes.llm_judge_node)

    # 자가 수정 준비.    
    graph.add_node("self_correction", nodes.self_correction_node)

    #graph.add_edge(START, "fact_judge")
    #graph.add_edge("fact_judge", "llm_judge")
    graph.add_edge(START, "debate")
    graph.add_edge("debate", "debate_judge")
    graph.add_edge("debate_judge", "llm_judge")

    # LLM Judge 후 점수를 판단해 사실과 부합하다 판단 되면 끝내거나 다시 자가 수정으로 이동.
    graph.add_conditional_edges(
        "llm_judge",
        _route_in_fact_check_agent,
        {
            "self_correction": "self_correction",
            END: END,
        },
    )

    # 자가 수정이 필요한 경우 준비(재시도 수 증가 등...)후 사실 판단 재시작.
    graph.add_edge("self_correction", "debate")

    return graph.compile()
