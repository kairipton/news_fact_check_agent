"""
판정 정확도 측정 스크립트.

정답이 명확한 단문 테스트 케이스 21개를 실행하여
파이프라인의 TRUE / FALSE / UNVERIFIABLE 판정 정확도를 측정합니다.

사용법:
    python eval.py

결과는 터미널과 eval_results/result_YYYYMMDD_HHMMSS.json 에 저장됩니다.

[채점 기준]
- TRUE / FALSE 케이스: 기대값과 정확히 일치해야 정답
- UNVERIFIABLE 케이스: TRUE도 FALSE도 아닌 결과(= UNVERIFIABLE)면 정답
  (LLM 특성상 UNVERIFIABLE 판정 기준이 불안정하므로 관대하게 채점)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import main
from main import create_initial_state


# ---------------------------------------------------------------------------
# 테스트 데이터셋
# 단문으로만 구성 — 복합문은 찬반 주장이 섞여 기대 verdict가 모호해짐
# ---------------------------------------------------------------------------

EVAL_DATASET: list[dict[str, str]] = [
    # ── TRUE (사실로 검증 가능한 단문) ──────────────────────────────────────
    {
        "id": "T01",
        "text": "아이폰은 애플이 만든다.",
        "expected": "TRUE",
    },
    {
        "id": "T02",
        "text": "파이썬은 프로그래밍 언어다.",
        "expected": "TRUE",
    },
    {
        "id": "T03",
        "text": "대한민국의 수도는 서울이다.",
        "expected": "TRUE",
    },
    {
        "id": "T04",
        "text": "BTS의 Dynamite는 2020년 빌보드 핫100 1위를 기록했다.",
        "expected": "TRUE",
    },
    {
        "id": "T05",
        "text": "넷플릭스는 온라인 동영상 스트리밍 서비스다.",
        "expected": "TRUE",
    },
    {
        "id": "T06",
        "text": "테슬라는 전기차를 만드는 회사다.",
        "expected": "TRUE",
    },
    {
        "id": "T07",
        "text": "삼성전자의 본사는 대한민국 수원시에 있다.",
        "expected": "TRUE",
    },

    # ── FALSE (사실이 아닌 것으로 검증 가능한 단문) ──────────────────────────
    {
        "id": "F01",
        "text": "지구는 평평하다.",
        "expected": "FALSE",
    },
    {
        "id": "F02",
        "text": "아이폰 12는 세계 최초의 5G 스마트폰이다.",
        "expected": "FALSE",
    },
    {
        "id": "F03",
        "text": "삼성전자 본사는 미국 실리콘밸리에 있다.",
        "expected": "FALSE",
    },
    {
        "id": "F04",
        "text": "서울은 세계에서 인구밀도가 가장 높은 도시다.",
        "expected": "FALSE",
    },
    {
        "id": "F05",
        "text": "mRNA 백신 기술은 2020년에 처음 개발됐다.",
        "expected": "FALSE",
    },
    {
        "id": "F06",
        "text": "유튜브는 마이크로소프트가 운영하는 동영상 플랫폼이다.",
        "expected": "FALSE",
    },
    {
        "id": "F07",
        "text": "파이썬 언어는 1990년대 마크 저커버그가 만들었다.",
        "expected": "FALSE",
    },

    # ── UNVERIFIABLE (객관적 사실로 검증 불가능한 단문) ──────────────────────
    {
        "id": "U01",
        "text": "이 영화는 역대 가장 훌륭한 작품이다.",
        "expected": "UNVERIFIABLE",
    },
    {
        "id": "U02",
        "text": "이 제품이 시중에서 가장 좋은 품질이다.",
        "expected": "UNVERIFIABLE",
    },
    {
        "id": "U03",
        "text": "한국 음식이 세계에서 제일 맛있다.",
        "expected": "UNVERIFIABLE",
    },
    {
        "id": "U04",
        "text": "A 기업이 업계에서 가장 혁신적인 회사다.",
        "expected": "UNVERIFIABLE",
    },
    {
        "id": "U05",
        "text": "이 책이 올해 최고의 베스트셀러다.",
        "expected": "UNVERIFIABLE",
    },
    {
        "id": "U06",
        "text": "2050년에는 AI가 인간의 모든 직업을 대체할 것이다.",
        "expected": "UNVERIFIABLE",
    },
    {
        "id": "U07",
        "text": "현재 가장 인기 있는 프로그래밍 언어는 파이썬이다.",
        "expected": "UNVERIFIABLE",
    },
]


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------

def is_correct(expected: str, actual: str) -> bool:
    """
    UNVERIFIABLE 케이스는 관대 채점:
    실제 결과가 TRUE도 FALSE도 아니면 정답으로 처리.
    TRUE / FALSE 케이스는 정확 일치.
    """
    actual = actual.strip().upper()
    if expected == "UNVERIFIABLE":
        return actual not in ("TRUE", "FALSE")
    return actual == expected


# ---------------------------------------------------------------------------
# 단일 케이스 실행
# ---------------------------------------------------------------------------

def run_case(case: dict[str, str]) -> dict:
    """케이스 1개를 파이프라인에 실행하고 결과를 반환합니다."""
    initial_state = create_initial_state(case["text"])
    result = main.graph.invoke(initial_state)

    judgment_results = result.get("judgment_results", [])
    actual_verdict = judgment_results[0]["verdict"] if judgment_results else "ERROR"

    correct = is_correct(case["expected"], actual_verdict)

    return {
        "id": case["id"],
        "text": case["text"],
        "expected": case["expected"],
        "actual": actual_verdict,
        "correct": correct,
        "judge_score": result.get("judge_score", 0.0),
        "correction_retries": result.get("correction_retries", 0),
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------

def print_results(results: list[dict]) -> None:
    """결과를 카테고리별로 출력합니다."""
    categories = ["TRUE", "FALSE", "UNVERIFIABLE"]

    print(f"\n{'=' * 70}")
    print("  판정 정확도 측정 결과")
    print(f"{'=' * 70}")

    # 케이스별 상세
    for cat in categories:
        rows = [r for r in results if r["expected"] == cat]
        correct_count = sum(1 for r in rows if r["correct"])
        print(f"\n[{cat}]  {correct_count}/{len(rows)}")
        print(f"  {'ID':<5} {'정답':<15} {'실제':<15} {'결과'}")
        print(f"  {'-'*50}")
        for r in rows:
            mark = "✅" if r["correct"] else "❌"
            print(f"  {r['id']:<5} {r['expected']:<15} {r['actual']:<15} {mark}  {r['text'][:30]}")

    # 전체 집계
    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    true_rows  = [r for r in results if r["expected"] == "TRUE"]
    false_rows = [r for r in results if r["expected"] == "FALSE"]
    unver_rows = [r for r in results if r["expected"] == "UNVERIFIABLE"]

    print(f"\n{'─' * 70}")
    print(f"  카테고리별 정확도")
    print(f"    TRUE          : {sum(r['correct'] for r in true_rows)}/{len(true_rows)}"
          f"  ({sum(r['correct'] for r in true_rows)/len(true_rows)*100:.1f}%)")
    print(f"    FALSE         : {sum(r['correct'] for r in false_rows)}/{len(false_rows)}"
          f"  ({sum(r['correct'] for r in false_rows)/len(false_rows)*100:.1f}%)")
    print(f"    UNVERIFIABLE  : {sum(r['correct'] for r in unver_rows)}/{len(unver_rows)}"
          f"  ({sum(r['correct'] for r in unver_rows)/len(unver_rows)*100:.1f}%)")
    print(f"{'─' * 70}")
    print(f"  전체 정확도     : {total_correct}/{total}  ({total_correct/total*100:.1f}%)")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run_eval() -> None:
    main.init_pipeline()

    print(f"\n총 {len(EVAL_DATASET)}개 케이스 실행 시작...\n")

    results = []
    for i, case in enumerate(EVAL_DATASET, start=1):
        print(f"  [{i:02d}/{len(EVAL_DATASET)}] {case['id']} — {case['text'][:40]}", end=" ", flush=True)
        row = run_case(case)
        mark = "✅" if row["correct"] else "❌"
        print(f"→ {row['actual']} {mark}")
        results.append(row)

    print_results(results)

    # JSON 저장
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"result_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "timestamp": timestamp,
                "total": len(results),
                "correct": sum(r["correct"] for r in results),
                "accuracy": sum(r["correct"] for r in results) / len(results),
                "cases": results,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  결과 저장: {out_path}\n")


if __name__ == "__main__":
    run_eval()
