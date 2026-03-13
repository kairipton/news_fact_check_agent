"""
LangGraph 파이프라인 공유 상태 정의.

모든 노드는 이 TypedDict를 읽고 부분 업데이트를 반환합니다.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class FactCheckState(TypedDict):
    """팩트체크 파이프라인 전체에서 공유되는 상태 구조."""

    # 원본 입력 텍스트
    input_text: str

    # 추출된 팩트체크 가능한 주장 목록.
    claims: list[str]

    # 주장별 검색 결과 리스트. 각 항목: {"claim": ..., "evidence": ...}
    search_results: list[dict[str, str]]

    # 주장별 판단 결과 리스트. 각 항목: {"claim": ..., "verdict": ..., "reasoning": ...}
    judgment_results: list[dict[str, str]]
    
    # LLM Judge가 판단 품질을 평가한 점수 (0.0 ~ 1.0)
    judge_score: float

    # LLM Judge의 피드백 텍스트. Self-Correction 루프에서 재판단 힌트로 사용.   
    judge_feedback: str
    
    # Self-Correction 재시도 횟수
    correction_retries: int

    # 최종 팩트체크 마크다운 리포트 
    final_report: str

    # 파이프라인 실행 중 발생한 에러 메시지. 정상 시 None.
    error: Optional[str]
