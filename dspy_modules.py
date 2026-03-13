"""DSPy 시그니처 정의, trainset 구성, 컴파일/로드 관리."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import dspy
from dspy.teleprompt import BootstrapFewShot

logger = logging.getLogger(__name__)

_COMPILED_DIR = ".compiled"

# 컴파일된 모듈 인스턴스 (setup_dspy() 호출 전까지 None)
claim_extractor: Optional[Any] = None
fact_judge: Optional[Any] = None
llm_judge: Optional[Any] = None


# ---------------------------------------------------------------------------
# DSPy 시그니처 정의
# ---------------------------------------------------------------------------

class ClaimExtractorSignature(dspy.Signature):
    """입력 뉴스 텍스트(사용자의 입력)에서 팩트체크 가능한 주장을 한 줄에 하나씩 추출합니다."""

    text: str = dspy.InputField(desc="팩트체크 대상 뉴스 또는 텍스트")
    claims: str = dspy.OutputField(
        desc="추출된 팩트체크 가능한 주장 목록. 각 주장을 개행(\\n)으로 구분하여 출력."
    )


class FactJudgeSignature(dspy.Signature):
    """제시된 근거를 바탕으로 주장의 사실 여부를 판단합니다."""

    claim: str = dspy.InputField(desc="판단할 주장")
    evidence: str = dspy.InputField(desc="판단 근거가 되는 검색 결과 텍스트")
    verdict: str = dspy.OutputField(
        desc="판정 결과. 반드시 TRUE, FALSE, UNVERIFIABLE 중 하나만 출력."
    )
    reasoning: str = dspy.OutputField(desc="판정 이유를 2~3 문장으로 설명.")


class LLMJudgeSignature(dspy.Signature):
    """팩트체크 판단 결과의 품질과 신뢰도를 평가합니다."""

    claim: str = dspy.InputField(desc="팩트체크한 주장")
    evidence: str = dspy.InputField(desc="사용된 근거 텍스트")
    verdict: str = dspy.InputField(desc="판정 결과 (TRUE/FALSE/UNVERIFIABLE)")
    reasoning: str = dspy.InputField(desc="판정 이유")
    quality_score: str = dspy.OutputField(
        desc="판단 품질 점수. 0.0에서 1.0 사이의 소수점 숫자만 출력. 예: 0.85"
    )
    feedback: str = dspy.OutputField(
        desc="판단 품질 개선을 위한 구체적인 피드백. 2~3 문장."
    )


# ---------------------------------------------------------------------------
# 학습 데이터 (Trainset)
# ---------------------------------------------------------------------------

_TRAINING_DATA: list[dict] = [
    {
        # ClaimExtractor 입력: 사용자가 입력한 뉴스/텍스트 원문
        "text": "BTS는 2020년 빌보드 핫100 차트에서 1위를 기록했다. 이는 한국 가수 최초의 쾌거로 알려져 있다.",  

        # ClaimExtractor 정답: 추출되어야 할 기대 주장의 핵심 키워드 (쉼표 구분)
        "expected_claims": "BTS 빌보드 핫100 1위, 한국 가수 최초",  

        # FactJudge/LLMJudge 입력: text에서 추출된 주장 하나
        "claim": "BTS는 한국 가수 최초로 빌보드 핫100 1위를 달성했다",  

        # FactJudge/LLMJudge 입력: Tavily 검색으로 얻은 근거 텍스트
        "evidence": "BTS의 'Dynamite'가 2020년 9월 빌보드 핫100 1위를 기록. 한국 가수 최초의 성과로 공식 확인됨.",  

        # FactJudge 정답: 사실 판정 (TRUE / FALSE / UNVERIFIABLE)
        "verdict": "TRUE",  

        # FactJudge 정답: 판정 이유
        "reasoning": "BTS가 2020년 빌보드 핫100 1위를 달성한 것과 한국 가수 최초라는 점이 검색 결과로 확인됩니다.",  

        # LLMJudge 정답: 판단 품질 점수 (0.0~1.0)
        "quality_score": "0.95",  

        # LLMJudge 정답: 판단 품질 개선 피드백
        "feedback": "검색 근거가 명확하고 판정이 정확합니다. 근거 출처를 추가하면 더 좋습니다.",  
    },
    {
        "text": "삼성전자는 2023년 반도체 메모리 분야에서 세계 시장 점유율 1위를 유지했으며, 본사는 미국 실리콘밸리에 위치한다.",
        "expected_claims": "삼성전자 메모리 세계 1위, 본사 미국",
        "claim": "삼성전자 본사는 미국 실리콘밸리에 위치한다",
        "evidence": "삼성전자 본사는 대한민국 경기도 수원시 영통구에 위치. 미국 법인은 별도 운영됨.",
        "verdict": "FALSE",
        "reasoning": "삼성전자의 본사는 대한민국 수원에 있으며, 미국 실리콘밸리에 위치한다는 주장은 사실이 아닙니다.",
        "quality_score": "0.92",
        "feedback": "판정이 정확하고 근거가 명확합니다. 본사 주소를 구체적으로 명시하면 신뢰도가 높아집니다.",
    },
    {
        "text": "한국은 2002년 FIFA 월드컵 준결승에서 독일에 패해 4강에서 탈락했으며, 3·4위전에서 터키를 꺾고 4위를 기록했다.",
        "expected_claims": "한국 2002 월드컵 4위, 독일에 준결승 패배",
        "claim": "한국은 2002 FIFA 월드컵 준결승에서 독일에 패배했다",
        "evidence": "2002 FIFA 한일 월드컵 준결승: 독일 1-0 한국. 한국은 이후 3·4위전에서 터키를 2-3으로 패해 4위 기록.",
        "verdict": "TRUE",
        "reasoning": "검색 결과로 2002 FIFA 월드컵 준결승에서 한국이 독일에 0-1로 패배한 것이 확인됩니다.",
        "quality_score": "0.90",
        "feedback": "판단이 정확합니다. 경기 스코어가 포함되어 근거의 신뢰도가 높습니다.",
    },
    {
        "text": "애플의 첫 5G 스마트폰인 아이폰 12가 2020년 출시되었으며, 이는 세계 최초의 5G 스마트폰이다.",
        "expected_claims": "아이폰 12 최초 5G 스마트폰",
        "claim": "아이폰 12는 세계 최초의 5G 스마트폰이다",
        "evidence": "삼성전자 갤럭시 S10 5G가 2019년 4월 세계 최초로 출시된 5G 스마트폰. 아이폰 12는 애플 최초 5G 모델(2020).",
        "verdict": "FALSE",
        "reasoning": "세계 최초의 5G 스마트폰은 2019년 삼성전자가 출시한 갤럭시 S10 5G입니다. 아이폰 12는 애플 최초 5G 모델이지만 세계 최초가 아닙니다.",
        "quality_score": "0.93",
        "feedback": "근거가 명확하고 판정이 정확합니다. 출시 날짜와 모델명이 정확히 명시되어 있습니다.",
    },
    {
        "text": "코로나19 mRNA 백신은 2020년에 처음 개발된 기술을 기반으로 하며, 기존에는 사용된 적 없는 완전히 새로운 방식이다.",
        "expected_claims": "mRNA 백신 2020년 최초 개발",
        "claim": "mRNA 백신 기술은 2020년에 처음 개발됐다",
        "evidence": "mRNA 기술 연구는 1990년대부터 진행됨. 코로나19 이전 암 치료 분야에서도 mRNA 기반 연구 진행 중이었음.",
        "verdict": "FALSE",
        "reasoning": "mRNA 기술은 1990년대부터 연구되어 온 기술입니다. 코로나19 백신에 처음 대규모로 적용된 것이지 기술 자체가 2020년에 개발된 것이 아닙니다.",
        "quality_score": "0.88",
        "feedback": "판정이 올바릅니다. 기술 역사에 대한 참고 문헌을 추가하면 더욱 설득력이 높아집니다.",
    },
    {
        "text": "독도는 대한민국 동해에 위치한 섬으로 경상북도 울릉군에 속하며, 현재 대한민국이 실효 지배하고 있다.",
        "expected_claims": "독도 경상북도 울릉군 소속, 대한민국 실효 지배",
        "claim": "독도는 행정구역상 경상북도 울릉군에 속한다",
        "evidence": "독도는 행정구역상 경상북도 울릉군 울릉읍 독도리에 속함. 대한민국 경찰 및 주민 거주 확인됨.",
        "verdict": "TRUE",
        "reasoning": "독도가 경상북도 울릉군 소속임은 대한민국 행정구역 기준으로 공식 확인된 사실입니다.",
        "quality_score": "0.96",
        "feedback": "판정이 명확하고 근거가 충분합니다. 법적 근거를 추가하면 완벽합니다.",
    },
    {
        "text": "서울은 세계에서 인구밀도가 가장 높은 도시로, 1㎢당 인구가 뉴욕보다 10배 이상 높다.",
        "expected_claims": "서울 세계 최고 인구밀도",
        "claim": "서울은 세계에서 인구밀도가 가장 높은 도시다",
        "evidence": "세계 인구밀도 1위는 방글라데시 다카로 알려짐. 서울의 인구밀도는 약 16,000명/㎢으로 높지만 세계 1위는 아님.",
        "verdict": "FALSE",
        "reasoning": "서울의 인구밀도는 세계적으로 높은 편이나, 세계 1위는 다카(방글라데시) 등 다른 도시입니다.",
        "quality_score": "0.87",
        "feedback": "판정이 올바릅니다. 비교 대상 도시의 수치를 함께 제시하면 신뢰도가 향상됩니다.",
    },
    {
        "text": "카카오톡은 대한민국에서 가장 많이 사용되는 메신저 앱으로, 2023년 기준 월간 활성 사용자가 5천만 명을 초과한다.",
        "expected_claims": "카카오톡 국내 1위 메신저, 월간 활성 사용자 5천만 초과",
        "claim": "카카오톡 월간 활성 사용자가 2023년 기준 5천만 명을 초과한다",
        "evidence": "카카오 공식 발표에 따르면 2023년 카카오톡 국내 월간 활성 사용자(MAU)는 약 4,800만 명 수준으로 보고됨.",
        "verdict": "UNVERIFIABLE",
        "reasoning": "카카오톡이 국내 최다 사용 메신저인 것은 맞으나, '5천만 명 초과'라는 수치는 공식 자료로 정확히 확인하기 어렵습니다.",
        "quality_score": "0.78",
        "feedback": "UNVERIFIABLE 판정은 적절하나, 수치의 출처를 더 구체적으로 조사하면 TRUE/FALSE 판정이 가능할 수 있습니다.",
    },
]


# ---------------------------------------------------------------------------
# 평가 지표 (Metric) 함수
# ---------------------------------------------------------------------------

def _claim_extractor_metric(example: dspy.Example, prediction: dspy.Prediction, trace: object = None) -> bool:
    """
    compile 시 ClaimExtractor의 예측 결과를 평가하는 함수.

    - example   : trainset의 한 row. example.expected_claims에 정답 키워드가 들어있음.
    - prediction: LLM이 실제로 예측한 결과. prediction.claims에 추출된 주장들이 들어있음.
    - 반환값    : 정답 키워드 중 하나 이상이 예측 결과에 포함되면 True → 이 예시를 demo로 선별.

    [정답 조건]
    expected_claims을 쉼표로 분리한 키워드 중 하나라도 prediction.claims 텍스트 안에 포함되면 합격.
    예) expected_claims = "삼성, 반도체, 매출" 일 때,
        prediction.claims = "삼성전자가 반도체 부문 매출을 발표했다" → "삼성", "반도체", "매출" 모두 포함 → True
        prediction.claims = "날씨가 맑다" → 키워드 없음 → False
    단, 2글자 이하 키워드("AI", "IT" 등)는 오탐 방지를 위해 비교에서 제외.
    """
    # LLM이 예측한 주장 텍스트 (없으면 빈 문자열)
    # ClaimExtractorSignature의 output이 claims.
    predicted = getattr(prediction, "claims", "").lower()
    if not predicted:
        return False

    # 정답 키워드 목록 (쉼표 구분, 2글자 이하 단어는 제외)
    keywords = [kw.strip() for kw in example.expected_claims.split(",") if len(kw.strip()) > 2]

    # 키워드 중 하나라도 예측 결과에 포함되면 합격
    return any(kw.lower() in predicted for kw in keywords)


def _fact_judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace: object = None) -> bool:
    """
    compile 시 FactJudge의 예측 결과를 평가하는 함수.

    - example   : trainset의 한 row. example.verdict에 정답 판정(TRUE/FALSE/UNVERIFIABLE)이 들어있음.
    - prediction: LLM이 실제로 예측한 결과. prediction.verdict에 예측 판정이 들어있음.
    - 반환값    : 예측 판정이 정답과 정확히 일치하면 True → 이 예시를 demo로 선별.

    [정답 조건]
    prediction.verdict가 example.verdict와 대소문자 무관하게 정확히 일치해야 합격.
    허용 값: TRUE, FALSE, UNVERIFIABLE 세 가지만.
    예) example.verdict = "TRUE",  prediction.verdict = "true"  → True  (대소문자 무시)
        example.verdict = "FALSE", prediction.verdict = "FALSE" → True
        example.verdict = "TRUE",  prediction.verdict = "FALSE" → False (판정 불일치)
        example.verdict = "TRUE",  prediction.verdict = "아마 맞는 것 같습니다" → False (형식 오류)
    """
    return example.verdict.strip().upper() == getattr(prediction, "verdict", "").strip().upper()


def _llm_judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace: object = None) -> bool:
    """
    compile 시 LLMJudge의 예측 결과를 평가하는 함수.

    - example   : trainset의 한 row (이 함수에서는 사용하지 않음).
    - prediction: LLM이 실제로 예측한 결과. prediction.quality_score에 품질 점수가 들어있음.
    - 반환값    : quality_score가 0.0~1.0 범위의 유효한 숫자면 True → 이 예시를 demo로 선별.

    [정답 조건]
    prediction.quality_score가 float으로 변환 가능하고, 0.0 이상 1.0 이하이면 합격.
    판정 내용의 옳고 그름은 따지지 않음. 형식(숫자 범위)만 확인.
    예) prediction.quality_score = "0.85" → True
        prediction.quality_score = "1.0"  → True
        prediction.quality_score = "1.5"  → False (범위 초과)
        prediction.quality_score = "높음" → False (숫자 아님, float() 변환 실패)
    """
    try:
        score = float(getattr(prediction, "quality_score", "").strip())
        return 0.0 <= score <= 1.0
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# 컴파일/로드 헬퍼
# ---------------------------------------------------------------------------

def _compile_or_load(module: Any, save_path: str, trainset: list, metric) -> Any:
    """
    캐시 파일이 있으면 로드, 없으면 BootstrapFewShot 컴파일 후 저장.

    컴파일 비용이 크기 떄문에 캐시를 사용하도록 함.
    compile은 내부적으로 훈련 데이터를 LLM에 실제로 돌려서 좋은 few shot 예시를 선별하며
    이 과정에서 API 호출이 발생해서 비용과 시간이 걸림.
    ※ 여기서 컴파일은 trainset에서 LLM이 처리한 예시들을 골라 few shot 프롬프트에 넣을 데모 리스트를 만드는 과정이다.

    또한, ChainfThought과 Predict 모듈을 컴파일 하지 않고 Save할수 있으나,
    Signature 만 저장되고 few shot 정보가 저장 되는건 아니므로 유의미 하진 않다.

    Args:
        module: 컴파일할 DSPy 모듈 인스턴스 (ChainOfThought 또는 Predict).
        save_path: 컴파일된 모듈을 저장할 파일 경로.
        trainset: 컴파일에 사용할 학습 데이터 리스트.
        metric: 컴파일 최적화에 사용할 평가 지표 함수.
    """

    # dspy.ChainOfThought과 dspy.Predict은 로컬에 저장하고 불러올 수 있다.
    if os.path.exists(save_path):
        logger.info("컴파일된 모듈 로드: %s", save_path)
        module.load(save_path)
        return module

    # optimizer 생성.
    logger.info("컴파일 시작: %s", save_path)
    optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3, max_labeled_demos=4)

    # compiled는 module의 타입 그대로임 (ChainOfThought 또는 Predict)
    compiled = optimizer.compile(module, trainset=trainset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    compiled.save(save_path)
    logger.info("컴파일 완료 및 저장: %s", save_path)
    return compiled


# ---------------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------------

def setup_dspy(openai_api_key: str, model_name: str) -> None:
    """DSPy LM 설정 및 모든 모듈 컴파일/로드. init_pipeline() 에서 호출."""
    global claim_extractor, fact_judge, llm_judge

    dspy.configure(lm=dspy.LM(f"openai/{model_name}", api_key=openai_api_key))
    logger.info("DSPy LM 설정 완료: %s", model_name)


    # region 입력값과 그걸 뒷바팅 하는 정답을 컴파일.
    claim_extractor_trainset = []
    for d in _TRAINING_DATA:
        example = dspy.Example(text=d["text"], expected_claims=d["expected_claims"]).with_inputs("text")
        claim_extractor_trainset.append(example)

    claim_extractor = _compile_or_load(
        module=dspy.ChainOfThought(ClaimExtractorSignature),
        save_path=os.path.join(_COMPILED_DIR, "claim_extractor.json"),
        trainset=claim_extractor_trainset,
        metric=_claim_extractor_metric,
    )
    #endregion


    # region 주장(claim)과 근거(evidence)를 가탕으로 사실인지(verdict) 판단하고, 그 이유(reasoning)를 컴파일.
    # 입력값은 주장과 근거
    fact_judge_trainset = []
    for d in _TRAINING_DATA:
        example = dspy.Example(claim=d["claim"], evidence=d["evidence"], verdict=d["verdict"], reasoning=d["reasoning"]).with_inputs("claim", "evidence")
        fact_judge_trainset.append(example)

    fact_judge = _compile_or_load(
        module=dspy.ChainOfThought(FactJudgeSignature),
        save_path=os.path.join(_COMPILED_DIR, "fact_judge.json"),
        trainset=fact_judge_trainset,
        metric=_fact_judge_metric,
    )
    # endregion

    # endregion 주장(claim)과 근거(evidence)를 가탕으로 사실인지(verdict) 판단한 이유(reasoning)에 대한 점수(quality_score)와 피드백(feedback)을 컴파일.
    # 주장, 근거, 사실, 판단 이유가 입력값.
    llm_judge_trainset = []
    for d in _TRAINING_DATA:
        example = dspy.Example(claim=d["claim"], evidence=d["evidence"], verdict=d["verdict"], reasoning=d["reasoning"], quality_score=d["quality_score"], feedback=d["feedback"]).with_inputs("claim", "evidence", "verdict", "reasoning")
        llm_judge_trainset.append(example)

    # dspy.Predict 사용.
    # 전체적인 추론(주장, 근거 등등)은 이미 다른 dspy 모듈에서 다 했으므로 여기서는 추론(CoT)과정 없이 판단만 한다.
    llm_judge = _compile_or_load(
        module=dspy.Predict(LLMJudgeSignature),
        save_path=os.path.join(_COMPILED_DIR, "llm_judge.json"),
        trainset=llm_judge_trainset,
        metric=_llm_judge_metric,
    )
    # endregion 

    logger.info("DSPy 모듈 초기화 완료.")
