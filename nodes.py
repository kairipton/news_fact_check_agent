"""
LangGraph 파이프라인 노드 정의.

팩트체크 파이프라인의 모든 노드 함수를 이 파일에서 관리합니다.
새 노드를 추가할 때는 함수를 작성한 후 main.py의 build_graph()에 등록하면 됩니다.
"""
from __future__ import annotations

import logging
from typing import Any

from openai import OpenAIError
from tavily import TavilyClient

import dspy_modules
from config import settings
from state import FactCheckState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 노드 1: 주장 추출 (Claim Extractor)
# ---------------------------------------------------------------------------

def claim_extractor_node(state: FactCheckState) -> dict[str, Any]:
    """
    주장 추출 노드: 입력 텍스트에서 팩트체크 가능한 주장 목록을 추출합니다.

    Args:
        state: 현재 파이프라인 상태. input_text 필드를 사용합니다.

    Returns:
        claims 필드를 포함한 부분 상태 업데이트 딕셔너리.
        입력값에 대한 주장을 리턴하게 됨.
    """
    logger.info("[claim_extractor] 노드 진입. 입력 길이: %d자", len(state["input_text"]))

    try:
        module = dspy_modules.claim_extractor  # DSPy ChainOfThought 모듈

        logger.debug("[claim_extractor] LLM 호출 시작.")

        # 모듈.forward
        result = module(text=state["input_text"])  # LLM이 주장을 \n 구분 문자열로 반환
        raw_claims = result.claims.strip()

        # "주장1\n주장2" → ["주장1", "주장2"] 형태로 변환
        claims = [line.strip() for line in raw_claims.split("\n") if line.strip()]
        if not claims:
            # LLM이 아무것도 추출 못한 경우 원문 앞부분을 주장으로 대체
            logger.warning("[claim_extractor] 추출된 주장이 없어 원문 첫 200자를 사용합니다.")
            claims = [state["input_text"][:200]]

        logger.info("[claim_extractor] 노드 종료. 추출된 주장 수: %d", len(claims))
        return {"claims": claims }

    except OpenAIError as exc:
        logger.error("[claim_extractor] OpenAI API 호출 실패: %s", exc, exc_info=True)
        return {"error": f"주장 추출 중 OpenAI 오류: {exc}"}
    except Exception as exc:
        logger.error("[claim_extractor] 예기치 않은 오류: %s", exc, exc_info=True)
        return {"error": f"주장 추출 실패: {exc}"}



def evidence_searcher_node(state: FactCheckState) -> dict[str, Any]:
    """
    각 주장에 대해 Tavily API로 근거를 검색.
    Tavaily API 비용이 사용됨.
    주장 여러개를 주장1개, 근거1개의 쌍으로 dict로 만들고 list로 만들어 반환.
    주장: claim
    근거: evidence

    Args:
        state: 현재 파이프라인 상태. claims 필드를 사용합니다.

    Returns:
        주장별 검색 결과(search_results)를 dict로 리턴.
    """
    claims = state.get("claims", [])
    # claims 예시: ["BTS가 빌보드 1위를 달성했다", "삼성 본사는 미국에 있다"]
    logger.info("[evidence_searcher] 노드 진입. 검색할 주장 수: %d", len(claims))

    if state.get("error"):
        logger.warning("[evidence_searcher] 에러 상태 감지. 검색 생략.")
        return {}

    client = TavilyClient(api_key=settings.tavily_api_key)

    # 최종적으로 반환할 결과 리스트.
    # 주장이 여러 개이므로 list[dict] 구조를 사용.
    # 예: [{"claim": "주장1", "evidence": "근거1"}, {"claim": "주장2", "evidence": "근거2"}]
    search_results: list[dict[str, str]] = []

    for claim in claims:  # 주장 하나씩 순차적으로 검색
        logger.debug("[evidence_searcher] 검색 실행: %s", claim)
        try:
            # claim 문자열을 검색 쿼리로 사용. 상위 3개 결과만 가져옴.
            response = client.search(query=claim, max_results=3)

            # response["results"]는 검색 결과 list.
            # 각 결과에서 "content" 필드(본문 텍스트)만 추출.
            evidence_texts = [
                r.get("content", "") for r in response.get("results", [])
            ]

            # 3개 결과를 공백으로 이어붙여 하나의 근거 문자열로 만듦.
            # filter(None, ...)으로 빈 문자열 제거.
            evidence = " ".join(filter(None, evidence_texts))

            # 주장과 근거를 쌍으로 묶어 저장.
            # fact_judge_node에서 이 구조를 그대로 꺼내 씀.
            search_results.append({"claim": claim, "evidence": evidence})
            logger.debug("[evidence_searcher] 검색 완료. 근거 길이: %d자", len(evidence))

        except Exception as exc:
            # 검색 실패 시 파이프라인을 중단하지 않고 빈 근거로 계속 진행
            logger.error(
                "[evidence_searcher] Tavily 검색 실패 (주장: '%s'): %s",
                claim,
                exc,
                exc_info=True,
            )
            # 근거가 없어도 주장은 남겨야 fact_judge가 UNVERIFIABLE로 처리할 수 있음
            search_results.append({"claim": claim, "evidence": ""})

    logger.info("[evidence_searcher] 노드 종료. 결과 수: %d", len(search_results))
    return {"search_results": search_results}




def fact_judge_node(state: FactCheckState) -> dict[str, Any]:
    """
    검색된 근거를 바탕으로 각 주장의 사실 여부를 판단.
    Self-Correction 루프에서 재진입 시 judge_feedback을 근거에 추가해 개선된 판단을 유도.
    주장(claim), 판단(verdict), 근거(reason)를 리스트로 만들어 judgement_results로 리턴.

    Args:
        state: 현재 파이프라인 상태. search_results, judge_feedback 필드를 사용합니다.

    Returns:
        judgment_results 필드를 포함한 부분 상태 업데이트 딕셔너리.
    """
    search_results = state.get("search_results", [])
    logger.info("[fact_judge] 노드 진입. 판단할 항목 수: %d", len(search_results))

    if state.get("error"):
        logger.warning("[fact_judge] 에러 상태 감지. 판단 생략.")
        return {}

    # DSPy ChainOfThought 모듈
    module = dspy_modules.fact_judge  

    # 이전 Self-Correction 루프의 피드백. 초기 실행할떈 없음
    prior_feedback = state.get("judge_feedback", "")  
    judgment_results: list[dict[str, str]] = []

    # 검색 결과들 하나씩 DSPy 모듈 실행.
    for item in search_results:

        # 주장과 근거
        claim = item["claim"]
        evidence = item["evidence"]

        # Self-Correction 재진입 시 이전 피드백을 근거에 붙여 LLM이 개선된 판단을 내리도록 유도
        if prior_feedback:
            evidence = f"{evidence}\n\n[이전 검토 피드백]: {prior_feedback}"

        logger.debug("[fact_judge] LLM 호출: 주장='%s'", claim)
        try:
            # 모듈.forward
            result = module(claim=claim, evidence=evidence)

            # 사실 판단(verdict)과 그 근거(reasoning).
            verdict = result.verdict.strip().upper()
            reasoning = result.reasoning.strip()

            # 주장, 판단, 근거를 append
            judgment_results.append(
                {"claim": claim, "verdict": verdict, "reasoning": reasoning}
            )
            logger.debug("[fact_judge] 판단 완료: '%s' → %s", claim, verdict)

        except OpenAIError as exc:
            logger.error(
                "[fact_judge] OpenAI API 호출 실패 (주장: '%s'): %s",
                claim,
                exc,
                exc_info=True,
            )
            judgment_results.append(
                {
                    "claim": claim,
                    "verdict": "UNVERIFIABLE",
                    "reasoning": f"API 오류로 판단 불가: {exc}",
                }
            )
        except Exception as exc:
            logger.error(
                "[fact_judge] 예기치 않은 오류 (주장: '%s'): %s",
                claim,
                exc,
                exc_info=True,
            )
            judgment_results.append(
                {
                    "claim": claim,
                    "verdict": "UNVERIFIABLE",
                    "reasoning": f"오류 발생: {exc}",
                }
            )

    logger.info("[fact_judge] 노드 종료. 판단 결과 수: %d", len(judgment_results))
    return {"judgment_results": judgment_results}


def debate_node(state: FactCheckState) -> dict[str, Any]:
    """
    Agent Debate 노드: 단순 판단 대신 찬성/반대 에이전트가 토론을 벌여 최종 판정을 내리는 모듈입니다.
    Debate Promopting 기반이며, 멀티라운드가 아니라 단판 승부.

    Args:
        state: 현재 파이프라인 상태.

    Returns:
        State 업데이트.
        - debate_pro(str): 찬성 에이전트의 주장.
        - debate_con(str): 반대 에이전트의 주장.
    """
    logger.info("[debate_node] 노드 진입. 토론 모듈 실행.")

    if state.get("error"):
        logger.warning("[debate_node] 에러 상태 감지. 토론 평가 생략. 점수 0.0 반환.")
        return {}
    
    module = dspy_modules.agent_debate  # DSPy ChainOfThought 모듈

    search_results = state.get("search_results", [])

    # 여러 주장/근거를 " | "로 합쳐 한 번에 모듈 호출
    combined_claim = " | ".join(item["claim"] for item in search_results)
    combined_evidence = " | ".join(item["evidence"] for item in search_results)

    try:
        result = module(claim=combined_claim, evidence=combined_evidence)
        debate_pro = result.debate_pro
        debate_con = result.debate_con
    except OpenAIError as e:
        logger.error("[debate_node] OpenAI API 호출 실패: %s", e, exc_info=True)
        return {
            "error": f"토론 모듈 실행 중 OpenAI 오류: {e}",
            "debate_pro": "",
            "debate_con": "",
        }
    except Exception as e:
        logger.error("[debate_node] 예기치 않은 오류: %s", e, exc_info=True)
        return {
            "error": f"토론 모듈 실행 중 오류: {e}",
            "debate_pro": "",
            "debate_con": "",
        }

    logger.info("[debate_node] 노드 종료. 찬반 논거 생성 완료.")
    return {
        "debate_pro": debate_pro,
        "debate_con": debate_con,
    }


def debate_judge_node(state: FactCheckState) -> dict[str, Any]:
    """
    Debate Judge 노드: 찬성/반대 논거를 검토하여 최종 판정을 내립니다.
    debate_node 이후에 실행되며, fact_judge_node를 대체합니다.

    Args:
        state: 현재 파이프라인 상태. search_results, debate_pro, debate_con 필드를 사용합니다.

    Returns:
        judgment_results 필드를 포함한 부분 상태 업데이트 딕셔너리.
    """
    logger.info("[debate_judge] 노드 진입.")

    if state.get("error"):
        logger.warning("[debate_judge] 에러 상태 감지. 판정 생략.")
        return {}

    module = dspy_modules.debate_judge

    search_results = state.get("search_results", [])
    debate_pro = state.get("debate_pro", "")
    debate_con = state.get("debate_con", "")

    combined_claim = " | ".join(item["claim"] for item in search_results)
    combined_evidence = " | ".join(item["evidence"] for item in search_results)

    try:
        result = module(
            claim=combined_claim,
            evidence=combined_evidence,
            debate_pro=debate_pro,
            debate_con=debate_con,
        )
        verdict = result.verdict.strip().upper()
        reasoning = result.reason.strip()

        # llm_judge_node가 judgment_results 리스트 구조를 소비하므로 동일한 포맷으로 반환
        judgment_results = [{"claim": combined_claim, "verdict": verdict, "reasoning": reasoning}]

    except OpenAIError as exc:
        logger.error("[debate_judge] OpenAI API 호출 실패: %s", exc, exc_info=True)
        judgment_results = [{"claim": combined_claim, "verdict": "UNVERIFIABLE", "reasoning": f"API 오류: {exc}"}]
    except Exception as exc:
        logger.error("[debate_judge] 예기치 않은 오류: %s", exc, exc_info=True)
        judgment_results = [{"claim": combined_claim, "verdict": "UNVERIFIABLE", "reasoning": f"오류 발생: {exc}"}]

    logger.info("[debate_judge] 노드 종료. 판정: %s", judgment_results[0]["verdict"])
    return {"judgment_results": judgment_results}


# ---------------------------------------------------------------------------
# 노드 4: LLM as a Judge
# ---------------------------------------------------------------------------

def llm_judge_node(state: FactCheckState) -> dict[str, Any]:
    """
    LLM Judge 노드: 판단 결과 전체의 품질과 신뢰도를 0.0~1.0 점수로 평가합니다.

    Args:
        state: 현재 파이프라인 상태. judgment_results, search_results 필드를 사용합니다.

    Returns:
        judge_score, judge_feedback 필드를 포함한 부분 상태 업데이트 딕셔너리.
    """
    logger.info("[llm_judge] 노드 진입.")

    if state.get("error"):
        logger.warning("[llm_judge] 에러 상태 감지. 평가 생략. 점수 0.0 반환.")
        return {"judge_score": 0.0, "judge_feedback": ""}

    # 검색 결과와 판단 결과.
    # 검색 결과가 리스트로 들어 있음 -> { "claim": ..., "evidence": ...}
    # 판단 결과가 리스트로 들어 있음 ->  { "claim": ..., "verdict": ..., "reasoning": ...}
    search_results = state.get("search_results", [])
    judgment_results = state.get("judgment_results", [])

    # 여러 주장의 결과를 " | "로 연결해 LLM에게 한 번에 전달
    claims_list = []
    evidence_list = []
    verdict_list = []
    reasoning_list = []

    for r in judgment_results:
        claims_list.append(r["claim"])
        verdict_list.append(r["verdict"])
        reasoning_list.append(r["reasoning"])

    for item in search_results:
        evidence_list.append(item.get("evidence", ""))

    combined_claim = " | ".join(claims_list)
    combined_evidence = " | ".join(evidence_list)
    combined_verdict = " | ".join(verdict_list)
    combined_reasoning = " | ".join(reasoning_list)

    logger.debug("[llm_judge] LLM 호출 시작.")

    # DSPy 모듈. Predict임.
    module = dspy_modules.llm_judge

    try:
        # 모듈.forward
        result = module(
            claim=combined_claim,
            evidence=combined_evidence,
            verdict=combined_verdict,
            reasoning=combined_reasoning,
        )

        score = max(0.0, min(1.0, float(result.quality_score.strip())))  # 0.0~1.0 범위로 클램프
        feedback = result.feedback.strip()

        # 점수와 피드백 리턴.
        logger.info("[llm_judge] 노드 종료. 평가 점수: %.2f", score)
        logger.debug("[llm_judge] 피드백: %s", feedback)
        return {"judge_score": score, "judge_feedback": feedback}

    except ValueError as exc:
        logger.error("[llm_judge] 점수 파싱 실패: %s", exc, exc_info=True)
        return {"judge_score": 0.5, "judge_feedback": "점수 파싱 실패"}
    except OpenAIError as exc:
        logger.error("[llm_judge] OpenAI API 호출 실패: %s", exc, exc_info=True)
        return {"judge_score": 0.5, "judge_feedback": f"API 오류: {exc}"}
    except Exception as exc:
        logger.error("[llm_judge] 예기치 않은 오류: %s", exc, exc_info=True)
        return {"judge_score": 0.5, "judge_feedback": f"평가 중 오류: {exc}"}


# ---------------------------------------------------------------------------
# 노드 5: Self-Correction
# ---------------------------------------------------------------------------

def self_correction_node(state: FactCheckState) -> dict[str, Any]:
    """
    Self-Correction 노드: 재시도 횟수를 증가시키고 재검증을 준비합니다.
    LLM Judge 점수가 기준 미만일때 진입함/

    Args:
        state: 현재 파이프라인 상태. correction_retries, judge_feedback 필드를 사용합니다.

    Returns:
        correction_retries 필드를 포함한 부분 상태 업데이트 딕셔너리.
    """
    retries = state.get("correction_retries", 0) + 1  # 재시도 횟수 증가
    logger.info(
        "[self_correction] Self-Correction 발동. 재시도 %d / %d회.",
        retries,
        settings.max_correction_retries,
    )
    logger.debug("[self_correction] 이전 Judge 피드백: %s", state.get("judge_feedback", ""))
    # 상태 업데이트 후 FactCheckAgent 내부 라우터가 fact_judge로 복귀시킴
    return {"correction_retries": retries}


# ---------------------------------------------------------------------------
# 노드 6: 리포트 생성 (Report Generator)
# ---------------------------------------------------------------------------

def report_generator_node(state: FactCheckState) -> dict[str, Any]:
    """리포트 생성 노드: 팩트체크 전체 결과를 마크다운 리포트로 출력합니다.

    에러 상태인 경우에도 부분 결과로 리포트를 생성합니다.

    Args:
        state: 파이프라인 전체 상태.

    Returns:
        final_report 필드를 포함한 부분 상태 업데이트 딕셔너리.
    """
    logger.info("[report_generator] 노드 진입.")

    lines: list[str] = ["# 📰 뉴스 팩트체크 리포트\n"]

    if state.get("error"):
        # 에러가 있어도 파이프라인을 죽이지 않고 오류 메시지를 리포트에 포함
        lines.append(f"> ⚠️ **처리 중 오류 발생**: {state['error']}\n")

    judgment_results = state.get("judgment_results", [])

    # 찬반 토론 섹션
    debate_pro = state.get("debate_pro", "")
    debate_con = state.get("debate_con", "")
    if debate_pro or debate_con:
        lines.append("## 🗣️ 찬반 토론\n")
        if debate_pro:
            lines.append(f"**👍 찬성 측 논거**\n{debate_pro}\n")
        if debate_con:
            lines.append(f"**👎 반대 측 논거**\n{debate_con}\n")

    if not judgment_results:
        lines.append("_분석된 주장이 없습니다._\n")
    else:
        verdict_icon = {"TRUE": "✅", "FALSE": "❌", "UNVERIFIABLE": "❓"}
        lines.append(f"## 최종 판정\n")

        for idx, item in enumerate(judgment_results, start=1):
            verdict = item.get("verdict", "UNVERIFIABLE")
            icon = verdict_icon.get(verdict, "❓")  # 판정값에 따라 이모지 선택
            lines.append(f"### {idx}. {item['claim']}")
            lines.append(f"**판정**: {icon} `{verdict}`")
            lines.append(f"**근거**: {item.get('reasoning', '-')}\n")

    score = state.get("judge_score", 0.0)
    retries = state.get("correction_retries", 0)
    lines.append("---")
    lines.append(f"**신뢰도 점수**: `{score:.2f}` | **검증 횟수**: `{retries + 1}회`")  # 검증 횟수는 재시도+1 (최초 1회 포함)

    report = "\n".join(lines)
    logger.info("[report_generator] 노드 종료. 리포트 길이: %d자", len(report))
    return {"final_report": report}
