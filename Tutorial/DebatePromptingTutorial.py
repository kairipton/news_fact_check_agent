"""
Debate Prompting 튜토리얼.

찬성 에이전트(Pro)와 반대 에이전트(Con)가 하나의 주장에 대해 서로 반론을 펼치고,
판사 에이전트(Judge)가 양측 주장을 종합해 최종 판단을 내리는 구조.

실행 방법:
    python Tutorial/DebatePromptingTutorial.py

의존성:
    pip install dspy langgraph openai python-dotenv
"""
from __future__ import annotations

import os
from typing import Any, Optional
from typing_extensions import TypedDict

import dspy
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# DSPy LM 초기화
# ---------------------------------------------------------------------------

dspy.configure(
    lm=dspy.LM(
        "openai/gpt-4.1-nano",
        api_key=os.environ["OPENAI_API_KEY"],
    )
)


# ---------------------------------------------------------------------------
# 공유 상태 정의
# ---------------------------------------------------------------------------

class DebateState(TypedDict):
    claim: str               # 토론할 주장
    evidence: str            # 참고할 근거 텍스트
    pro_argument: str        # 찬성 에이전트의 주장
    con_argument: str        # 반대 에이전트의 주장
    final_verdict: str       # 판사의 최종 판정 (TRUE / FALSE / UNVERIFIABLE)
    final_reasoning: str     # 판사의 판정 근거


# ---------------------------------------------------------------------------
# DSPy Signature 정의
# ---------------------------------------------------------------------------

class ProAgentSignature(dspy.Signature):
    """주어진 근거를 바탕으로 해당 주장이 사실임을 지지하는 논거를 제시합니다."""

    claim: str = dspy.InputField(desc="검증할 주장")
    evidence: str = dspy.InputField(desc="참고할 근거 텍스트")
    argument: str = dspy.OutputField(desc="주장이 사실임을 지지하는 논거. 2~3 문장.")


class ConAgentSignature(dspy.Signature):
    """주어진 근거를 바탕으로 해당 주장이 거짓이거나 불확실함을 지적하는 반론을 제시합니다."""

    claim: str = dspy.InputField(desc="검증할 주장")
    evidence: str = dspy.InputField(desc="참고할 근거 텍스트")
    argument: str = dspy.OutputField(desc="주장에 대한 반론. 2~3 문장.")


class JudgeAgentSignature(dspy.Signature):
    """찬성 측과 반대 측 주장을 모두 검토한 후 최종 판정을 내립니다."""

    claim: str = dspy.InputField(desc="검증할 주장")
    pro_argument: str = dspy.InputField(desc="찬성 측 주장")
    con_argument: str = dspy.InputField(desc="반대 측 주장")
    verdict: str = dspy.OutputField(
        desc="최종 판정. 반드시 TRUE, FALSE, UNVERIFIABLE 중 하나."
    )
    reasoning: str = dspy.OutputField(desc="양측 주장을 종합한 최종 판정 이유. 2~3 문장.")


# ---------------------------------------------------------------------------
# DSPy 모듈 인스턴스
# (튜토리얼이므로 컴파일 없이 ChainOfThought 직접 사용)
# ---------------------------------------------------------------------------

pro_agent_module = dspy.ChainOfThought(ProAgentSignature)
con_agent_module = dspy.ChainOfThought(ConAgentSignature)
judge_module = dspy.ChainOfThought(JudgeAgentSignature)


# ---------------------------------------------------------------------------
# 노드 정의
# ---------------------------------------------------------------------------

def pro_agent_node(state: DebateState) -> dict[str, Any]:
    """찬성 에이전트: 주장이 사실임을 지지하는 논거를 생성합니다."""
    print("\n[Pro Agent] 찬성 논거 생성 중...")

    result = pro_agent_module(claim=state["claim"], evidence=state["evidence"])
    argument = result.argument.strip()

    print(f"[Pro Agent] {argument}")
    return {"pro_argument": argument}


def con_agent_node(state: DebateState) -> dict[str, Any]:
    """반대 에이전트: 주장에 대한 반론을 생성합니다."""
    print("\n[Con Agent] 반대 논거 생성 중...")

    result = con_agent_module(claim=state["claim"], evidence=state["evidence"])
    argument = result.argument.strip()

    print(f"[Con Agent] {argument}")
    return {"con_argument": argument}


def judge_node(state: DebateState) -> dict[str, Any]:
    """판사 에이전트: 양측 주장을 종합해 최종 판정을 내립니다."""
    print("\n[Judge] 최종 판정 중...")

    result = judge_module(
        claim=state["claim"],
        pro_argument=state["pro_argument"],
        con_argument=state["con_argument"],
    )
    verdict = result.verdict.strip().upper()
    reasoning = result.reasoning.strip()

    print(f"[Judge] 판정: {verdict}")
    print(f"[Judge] 이유: {reasoning}")
    return {"final_verdict": verdict, "final_reasoning": reasoning}


# ---------------------------------------------------------------------------
# 그래프 구성
# ---------------------------------------------------------------------------

def build_debate_graph() -> Any:
    """
    Debate Prompting 그래프를 생성합니다.

    흐름:
        START → pro_agent → con_agent → judge → END

    찬성과 반대 에이전트는 서로의 주장을 모르는 상태에서 독립적으로 논거를 생성합니다.
    (순차 실행이지만 입력이 공유되지 않아 독립적)
    판사는 양측 주장을 모두 받아 최종 판정을 내립니다.
    """
    workflow = StateGraph(DebateState)

    workflow.add_node("pro_agent", pro_agent_node)
    workflow.add_node("con_agent", con_agent_node)
    workflow.add_node("judge", judge_node)

    workflow.add_edge(START, "pro_agent")
    workflow.add_edge("pro_agent", "con_agent")
    workflow.add_edge("con_agent", "judge")
    workflow.add_edge("judge", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_debate_graph()

    # 테스트할 주장과 근거
    initial_state: DebateState = {
        "claim": "아이폰 12는 세계 최초의 5G 스마트폰이다",
        "evidence": (
            "삼성전자 갤럭시 S10 5G가 2019년 4월 세계 최초로 출시된 5G 스마트폰이다. "
            "아이폰 12는 2020년 출시된 애플 최초의 5G 모델이다."
        ),
        "pro_argument": "",
        "con_argument": "",
        "final_verdict": "",
        "final_reasoning": "",
    }

    print("=" * 60)
    print(f"검증 주장: {initial_state['claim']}")
    print("=" * 60)

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"판정    : {result['final_verdict']}")
    print(f"판정이유: {result['final_reasoning']}")
