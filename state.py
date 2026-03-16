"""
LangGraph 파이프라인 공유 상태 정의.

모든 노드는 이 TypedDict를 읽고 부분 업데이트를 반환합니다.
"""
from __future__ import annotations

from typing import Optional, TypedDict, Annotated


class FactCheckState(TypedDict):
    """팩트체크 파이프라인 전체에서 공유되는 상태 구조."""

    input_text: Annotated[str, "사용자가 입력값"]
    claims: Annotated[list[str], "입력 텍스트에서 추출된 주장 목록"]

    # 주장별 검색 결과 리스트. 각 항목: {"claim": ..., "evidence": ...}
    search_results: Annotated[list[dict[str, str]], "주장별 검색 결과 리스트"]

    # 주장별 판단 결과 리스트. 각 항목: {"claim": ..., "verdict": ..., "reasoning": ...}
    judgment_results: Annotated[list[dict[str, str]], "주장별 판단 결과 리스트"]

    debate_pro: Annotated[str, "토론 주제에 대한 찬성 에이전트의 주장. 주장들을 하나의 문자열로 연결." ]
    debate_con: Annotated[str, "토론 주제에 대한 반대 에이전트의 주장. 주장들을 하나의 문자열로 연결." ]
    
    judge_score: Annotated[float, "LLM Judge 품질 평가 점수 (0.0 ~ 1.0)"]
    judge_feedback: Annotated[str, "LLM Judge 피드백 텍스트. Self-Correction 루프에서 재판단 힌트로 사용."]
    correction_retries: Annotated[int, "Self-Correction 재시도 횟수"]
    final_report: Annotated[str, "최종 팩트체크 마크다운 리포트"]
    error:  Annotated[Optional[str], "파이프라인 실행 중 발생한 에러 메시지. 정상 시 None."]