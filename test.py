"""
팩트체크 파이프라인 독립 실행 테스트.

Streamlit 없이 단독으로 실행하여 파이프라인 전체 동작을 검증합니다.

사용법:
    python test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import main
from main import create_initial_state



SAMPLE_TEXT_1 = """
BTS는 2020년 빌보드 핫100 차트에서 1위를 기록하며 한국 가수 최초의 성과를 이뤘다.
삼성전자는 미국 캘리포니아에 본사를 둔 글로벌 테크 기업이며,
2023년 반도체 메모리 시장 점유율 1위를 유지했다.
"""

SAMPLE_TEXT_2 = """
아이폰 12는 2020년 출시된 세계 최초의 5G 스마트폰이다.
서울은 세계에서 인구밀도가 가장 높은 도시로 알려져 있다.
"""


def run_single_test(label: str, input_text: str) -> None:
    """단일 테스트 케이스를 실행하고 결과를 출력합니다.

    Args:
        label: 테스트 식별 레이블.
        input_text: 팩트체크할 텍스트.
    """
    print(f"\n{'=' * 60}")
    print(f"테스트: {label}")
    print(f"{'=' * 60}")
    print(f"[입력]\n{input_text.strip()}\n")

    # state 초기화
    initial_state = create_initial_state(input_text)

    # 그래프 invoke
    result = main.graph.invoke(initial_state)

    print("[최종 리포트]")
    print(result["final_report"])
    print()
    print("[상태 요약]")
    print(f"  추출된 주장 수  : {len(result.get('claims', []))}")
    print(f"  신뢰도 점수    : {result.get('judge_score', 0.0):.2f}")
    print(f"  Self-Correction: {result.get('correction_retries', 0)}회")
    print(f"  에러           : {result.get('error')}")

    path_dir = Path( f"test_results" )
    path_dir.mkdir(exist_ok=True)

    debug_path = f"./{str(path_dir)}/.result_{label.replace(' ', '_')}.json"
    with open(debug_path, "w", encoding="utf-8") as fp:
        serializable = {
            k: v for k, v in result.items() if isinstance(v, (str, int, float, list, dict, type(None)))
        }
        json.dump(serializable, fp, ensure_ascii=False, indent=2)
    print(f"  상세 결과 저장 : {debug_path}")


# ----------- 엔트리 포인트 ------------
def run_all_tests() -> None:
    """모든 테스트 케이스를 순차 실행합니다."""

    # 각종 초기화: 그래프 빌드, logging, dspy
    main.init_pipeline()

    run_single_test("케이스1_BTS_삼성", SAMPLE_TEXT_1)
    run_single_test("케이스2_아이폰_서울", SAMPLE_TEXT_2)

    print(f"\n{'=' * 60}")
    print("전체 테스트 완료.")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_tests()
