"""
LangGraph 파이프라인 오케스트레이터 정의.

주장 추출, 근거 검색, 리포트 생성은 노드로 직접 등록하고,
내부 루프(Self-Correction)가 있는 사실 검증만 FactCheckAgent 서브그래프로 분리합니다.
파이프라인 사용 전 반드시 init_pipeline()을 호출해야 합니다.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, START, StateGraph

from agents import build_fact_check_agent
from config import settings
from dspy_modules import setup_dspy
from nodes import (
    claim_extractor_node,
    evidence_searcher_node,
    report_generator_node,
)
from state import FactCheckState

logger = logging.getLogger(__name__)
graph: Optional[Any] = None


def setup_logging() -> None:
    """로깅 포맷 및 레벨을 초기화합니다.

    타임스탬프 + 레벨 + 모듈명 + 메시지 형식으로 DEBUG 이상 로그를 출력합니다.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_graph() -> StateGraph:
    """팩트체크 파이프라인 그래프를 생성합니다.

    주장 추출, 근거 검색, 리포트 생성은 단일 노드로 등록합니다.
    내부에 Self-Correction 루프가 있는 사실 검증만 FactCheckAgent 서브그래프로 분리합니다.

    노드 구성:
        claim_extractor    : 주장 추출 노드
        evidence_searcher  : 근거 검색 노드
        fact_check_agent   : 사실 판단 + LLM Judge + Self-Correction (서브그래프)
        report_generator   : 리포트 생성 노드

    Returns:
        컴파일 전 StateGraph 인스턴스.
    """
    workflow = StateGraph(FactCheckState)

    # 사용자의 입력에 대한 주장을 가져옴
    workflow.add_node("claim_extractor", claim_extractor_node)

    # 주장에 대한 근거를 설정.
    workflow.add_node("evidence_searcher", evidence_searcher_node)

    # 사실 확인 에이전트.
    workflow.add_node("fact_check_agent", build_fact_check_agent())

    # 리포트 생성.
    workflow.add_node("report_generator", report_generator_node)

    workflow.add_edge(START, "claim_extractor")
    workflow.add_edge("claim_extractor", "evidence_searcher")
    workflow.add_edge("evidence_searcher", "fact_check_agent")
    workflow.add_edge("fact_check_agent", "report_generator")
    workflow.add_edge("report_generator", END)

    return workflow


def create_initial_state(input_text: str) -> FactCheckState:
    """파이프라인 실행을 위한 초기 상태를 생성합니다.

    Args:
        input_text: 팩트체크할 뉴스 또는 텍스트.

    Returns:
        모든 필드가 기본값으로 초기화된 FactCheckState.
    """
    return FactCheckState(
        input_text=input_text,
        claims=[],
        search_results=[],
        judgment_results=[],
        debate_pro="",
        debate_con="",
        judge_score=0.0,
        judge_feedback="",
        correction_retries=0,
        final_report="",
        error=None,
    )


def init_pipeline() -> None:
    """파이프라인 전체를 초기화합니다.

    로깅 설정 → DSPy 모듈 준비 → 그래프 컴파일 순서로 실행됩니다.
    graph 변수에 컴파일된 LangGraph 인스턴스가 할당됩니다.
    """
    global graph
    setup_logging()
    setup_dspy(
        openai_api_key=settings.openai_api_key,
        model_name=settings.model_name,
    )
    graph = build_graph().compile()
    logger.info("파이프라인 초기화 완료.")
