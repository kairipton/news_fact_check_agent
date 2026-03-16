"""
rapidfuzz 튜토리얼.

문자열 편집 거리 기반 유사도 비교 라이브러리.
의미 이해는 없지만 표기 변형, 띄어쓰기 차이, 오탈자를 잡아줌.
임베딩/API 없이 로컬에서 동작, 패키지 크기 ~1MB.

실행 방법:
    python Tutorial/RapidfuzzTutorial.py
"""
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# 1. fuzz.ratio — 전체 문자열 대 전체 문자열 비교
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. fuzz.ratio (전체 vs 전체)")
print("=" * 60)

# 완전 일치
score = fuzz.ratio("BTS 빌보드", "BTS 빌보드")
print(f"완전 일치       : {score}")  # 100

# 오탈자
score = fuzz.ratio("BTS 빌보드", "BTS 빌보오드")
print(f"오탈자          : {score}")  # 88~90

# 완전히 다른 텍스트
score = fuzz.ratio("BTS 빌보드", "방탄소년단 차트")
print(f"완전 다름       : {score}")  # 낮음


# ---------------------------------------------------------------------------
# 2. fuzz.partial_ratio — 짧은 쪽이 긴 쪽 어딘가에 포함되는지 비교
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("2. fuzz.partial_ratio (부분 포함 비교)")
print("=" * 60)

# 키워드가 긴 문장 안에 포함된 경우
score = fuzz.partial_ratio("BTS 빌보드", "BTS가 빌보드 핫100 1위를 달성했다")
print(f"키워드 포함     : {score}")  # 90 이상

# 띄어쓰기 차이
score = fuzz.partial_ratio("한국가수최초", "한국 가수 최초의 쾌거")
print(f"띄어쓰기 차이   : {score}")  # 어느 정도 높음

# 의미 동치 (표기가 완전히 다름) — rapidfuzz의 한계
score = fuzz.partial_ratio("방탄소년단", "BTS가 빌보드 1위")
print(f"의미 동치 (한계): {score}")  # 낮음 — 의미는 같지만 표기가 달라 못 잡음


# ---------------------------------------------------------------------------
# 3. fuzz.token_sort_ratio — 단어 순서가 달라도 비교
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("3. fuzz.token_sort_ratio (단어 순서 무관 비교)")
print("=" * 60)

# 단어 순서가 다른 경우
score = fuzz.token_sort_ratio("빌보드 BTS 1위", "BTS 빌보드 1위")
print(f"순서 다름       : {score}")  # 100 (단어를 정렬 후 비교)

score = fuzz.token_sort_ratio("삼성 반도체 1위", "반도체 세계 1위 삼성")
print(f"순서 다름2      : {score}")  # 높음


# ---------------------------------------------------------------------------
# 4. 실제 사용 예시 — _claim_extractor_metric 개선 패턴
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("4. 실전 적용 예시 — 키워드 기반 + rapidfuzz 조합")
print("=" * 60)

THRESHOLD = 80  # 유사도 기준점

def check_keyword_match(expected_claims: str, predicted: str) -> bool:
    """
    expected_claims의 키워드가 predicted 안에 있는지 확인.
    1) 완전 포함 (기존 키워드 방식)
    2) partial_ratio 유사도 80 이상 (rapidfuzz 추가)
    둘 중 하나라도 해당하면 True.
    """
    predicted = predicted.lower()

    keywords = []
    for kw in expected_claims.split(","):
        if len(kw.strip()) > 2:
            keywords.append(kw.strip())

    for kw in keywords:
        kw_lower = kw.lower()

        # 완전 포함
        if kw_lower in predicted:
            print(f"  [{kw}] 완전 포함 → True")
            return True

        # 유사도 80 이상
        score = fuzz.partial_ratio(kw_lower, predicted)
        if score >= THRESHOLD:
            print(f"  [{kw}] 유사도 {score} → True")
            return True

        print(f"  [{kw}] 유사도 {score} → 불일치")

    return False


# 케이스 1: 완전 포함
print("\n케이스1 - 완전 포함:")
result = check_keyword_match(
    expected_claims="BTS 빌보드, 한국 가수 최초",
    predicted="BTS가 빌보드 핫100 1위를 기록. 한국 가수 최초의 쾌거."
)
print(f"결과: {result}")

# 케이스 2: 띄어쓰기 차이 (기존 방식은 놓침)
print("\n케이스2 - 띄어쓰기 차이:")
result = check_keyword_match(
    expected_claims="한국가수최초",
    predicted="한국 가수 최초의 쾌거로 알려져 있다"
)
print(f"결과: {result}")

# 케이스 3: 의미 동치 — 여전히 못 잡음
print("\n케이스3 - 의미 동치 (한계):")
result = check_keyword_match(
    expected_claims="방탄소년단 빌보드",
    predicted="BTS가 빌보드 1위를 달성했다"
)
print(f"결과: {result}")
