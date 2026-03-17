"""DSPy 시그니처 정의, trainset 구성, 컴파일/로드 관리."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import dspy
from dspy.teleprompt import BootstrapFewShot
import rapidfuzz

logger = logging.getLogger(__name__)

_COMPILED_DIR = ".compiled"

# 컴파일된 모듈 인스턴스 (setup_dspy() 호출 전까지 None)
claim_extractor: Any = None
agent_debate: Any = None # 단순 사실 확인 판단을 넘어 토론을 하여 판단을 결정하는 모듈 (Debate Prompting)
debate_judge: Any = None
llm_judge: Any = None


# ---------------------------------------------------------------------------
# DSPy 시그니처 정의
# ---------------------------------------------------------------------------

class ClaimExtractorSignature(dspy.Signature):
    """
    입력 뉴스 텍스트(사용자의 입력)에서 팩트체크 가능한 찬성/반대 주장을 한 줄에 하나씩 추출합니다.
    """

    text: str = dspy.InputField(desc="팩트체크 대상 뉴스 또는 텍스트")
    claims_pro: str = dspy.OutputField(
        desc="사용자 입력에 대한 찬성 에이전트가 주장에 대해 사실임을 지지하는 주장 목록. 각 주장을 개행(\\n)으로 구분하여 출력."
    )
    claims_con: str = dspy.OutputField(
        desc="사용자 입력에 대한 반대 에이전트가 주장에 대해 사실이 아님을 지적하는 주장 목록. 각 주장을 개행(\\n)으로 구분하여 출력."
    )

class AgentDebateSignature(dspy.Signature):
    """
    주어진 찬성/반대 주장을 바탕으로 찬성 에이전트와 반대 에이전트가 각각 논거를 제시.
    """

    claim_pro: str = dspy.InputField(desc="사용자 입력에 대한 찬성 에이전트가 주장에 대해 사실임을 지지하는 주장")
    claim_con: str = dspy.InputField(desc="사용자 입력에 대한 반대 에이전트가 주장에 대해 사실이 아님을 지적하는 주장")

    evidence_pro: str = dspy.InputField(desc="찬성측 논의에 참고할 근거 텍스트")
    evidence_con: str = dspy.InputField(desc="반대측 논의에 참고할 근거 텍스트")

    debate_pro: str = dspy.OutputField( desc="주장이 사실임을 지지하는 찬성 에이전트의 논거." )
    debate_con: str = dspy.OutputField( desc="주장이 거짓이거나 불확실함을 지적하는 반대 에이전트의 논거." )

class DebateJudgeSignature(dspy.Signature):
    """
    찬성 측과 반대 측 주장을 모두 검토한 후 최종 판정을 내림.
    """

    claim_pro: str = dspy.InputField(desc="사용자 입력에 대한 찬성측에서 논의할 주장")
    claim_con: str = dspy.InputField(desc="사용자 입력에 대한 반대측에서 논의할 주장")
    evidence_pro: str = dspy.InputField(desc="찬성측 논의에 참고할 근거 텍스트")
    evidence_con: str = dspy.InputField(desc="반대측 논의에 참고할 근거 텍스트")
    debate_pro: str = dspy.InputField(desc="찬성 측 주장")
    debate_con: str = dspy.InputField(desc="반대 측 주장")

    verdict: str = dspy.OutputField( desc="최종 판정. 반드시 TRUE, FALSE, UNVERIFIABLE 중 하나만 출력." )
    reason: str = dspy.OutputField( desc="양측 주장을 종합한 최종 판정 이유." )


class LLMJudgeSignature(dspy.Signature):
    """찬반 토론 기반 팩트체크 판단 결과의 품질과 신뢰도를 평가합니다."""

    claim_pro: str = dspy.InputField(desc="찬성 측 주장")
    claim_con: str = dspy.InputField(desc="반대 측 주장")
    evidence_pro: str = dspy.InputField(desc="찬성 측 근거 텍스트")
    evidence_con: str = dspy.InputField(desc="반대 측 근거 텍스트")
    debate_pro: str = dspy.InputField(desc="찬성 에이전트의 논거")
    debate_con: str = dspy.InputField(desc="반대 에이전트의 논거")
    verdict: str = dspy.InputField(desc="최종 판정 결과 (TRUE/FALSE/UNVERIFIABLE)")
    reason: str = dspy.InputField(desc="최종 판정 이유")
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

        # ClaimExtractor 정답: 추출되어야 할 기대 주장의 핵심 키워드 (여러개 포함될 수 있으며, 쉼표로 구분)
        "expected_claims": "BTS 빌보드 핫100 1위, 한국 가수 최초",

        # FactJudge/LLMJudge/DebateAgent 입력: text에서 추출된 찬성 주장
        "claim": "BTS는 한국 가수 최초로 빌보드 핫100 1위를 달성했다",

        # FactJudge/LLMJudge/DebateAgent 입력: text에서 추출된 반대 주장
        "claim_con": "BTS의 'Dynamite'는 영어 곡이므로 순수한 한국 가수 최초 빌보드 핫100 1위 성과로 단정하기 어렵다는 시각이 있다",

        # FactJudge/LLMJudge/DebateAgent 입력: Tavily 검색으로 얻은 근거 텍스트
        "evidence": "BTS의 'Dynamite'가 2020년 9월 빌보드 핫100 1위를 기록. 한국 가수 최초의 성과로 공식 확인됨.",

        # ProAgent 정답: 주장이 사실임을 지지하는 논거
        "debate_pro": "BTS의 'Dynamite'가 2020년 9월 빌보드 핫100 1위를 기록한 것은 공식 차트로 확인된 사실입니다. 이전까지 한국 가수가 빌보드 핫100 1위를 차지한 사례가 없었으므로 한국 가수 최초라는 주장은 타당합니다.",

        # ConAgent 정답: 주장에 대한 반론
        "debate_con": "빌보드 핫100은 주로 영어권 시장을 대상으로 하며, BTS의 'Dynamite'는 영어 곡입니다. 따라서 한국 가수의 성과라기보다 영어 팝 음악으로서의 성과로 보는 시각도 존재합니다. 또한 '최초'라는 표현에는 솔로/그룹 구분 등 기준이 모호한 부분이 있습니다.",

        # FactJudge 정답: 사실 판정 (TRUE / FALSE / UNVERIFIABLE)
        "verdict": "TRUE",

        # FactJudge 정답: 판정 이유
        "reason": "BTS가 2020년 빌보드 핫100 1위를 달성한 것과 한국 가수 최초라는 점이 검색 결과로 확인됩니다.",

        # LLMJudge 정답: 판단 품질 점수 (0.0~1.0)
        "quality_score": "0.95",

        # LLMJudge 정답: 판단 품질 개선 피드백
        "feedback": "검색 근거가 명확하고 판정이 정확합니다. 근거 출처를 추가하면 더 좋습니다.",
    },
    {
        "text": "삼성전자는 2023년 반도체 메모리 분야에서 세계 시장 점유율 1위를 유지했으며, 본사는 미국 실리콘밸리에 위치한다.",
        "expected_claims": "삼성전자 메모리 세계 1위, 본사 미국",
        "claim": "삼성전자 본사는 미국 실리콘밸리에 위치한다",
        "claim_con": "삼성전자의 공식 등기 본사는 대한민국 수원시에 있으며, 실리콘밸리에 있는 것은 별도 법인인 북미 법인이다",
        "evidence": "삼성전자 본사는 대한민국 경기도 수원시 영통구에 위치. 미국 법인은 별도 운영됨.",
        "debate_pro": "삼성전자는 미국 실리콘밸리에 북미 법인을 두고 있으며, 글로벌 R&D 거점으로 운영 중입니다. 실질적인 의사결정과 사업 활동이 실리콘밸리에서도 이루어진다는 점에서 본사 기능의 일부가 그곳에 있다고 볼 수 있습니다.",
        "debate_con": "삼성전자의 공식 등기 본사는 대한민국 경기도 수원시 영통구 삼성로 129입니다. 미국 실리콘밸리에 있는 것은 북미 법인으로, 본사와는 법적으로 별개의 법인입니다. 공식 문서 어디에도 본사가 미국에 있다는 기재는 없습니다.",
        "verdict": "FALSE",
        "reason": "삼성전자의 본사는 대한민국 수원에 있으며, 미국 실리콘밸리에 위치한다는 주장은 사실이 아닙니다.",
        "quality_score": "0.92",
        "feedback": "판정이 정확하고 근거가 명확합니다. 본사 주소를 구체적으로 명시하면 신뢰도가 높아집니다.",
    },
    {
        "text": "한국은 2002년 FIFA 월드컵 준결승에서 독일에 패해 4강에서 탈락했으며, 3·4위전에서 터키를 꺾고 4위를 기록했다.",
        "expected_claims": "한국 2002 월드컵 4위, 독일에 준결승 패배",
        "claim": "한국은 2002 FIFA 월드컵 준결승에서 독일에 패배했다",
        "claim_con": "2002 FIFA 월드컵 준결승에서 한국의 독일전 패배는 당시 심판 판정 논란으로 인해 경기의 공정성에 의문이 제기되었다",
        "evidence": "2002 FIFA 한일 월드컵 준결승: 독일 1-0 한국. 한국은 이후 3·4위전에서 터키를 2-3으로 패해 4위 기록.",
        "debate_pro": "2002 FIFA 한일 월드컵 준결승 경기 공식 기록에 따르면 독일이 한국을 1-0으로 이겼습니다. 이는 FIFA 공식 통계에도 명시되어 있으며, 한국이 패배한 것은 부정할 수 없는 사실입니다.",
        "debate_con": "경기 결과 자체는 사실이지만, 당시 심판 판정 논란으로 인해 경기의 공정성에 의문이 제기된 바 있습니다. 따라서 단순히 '패배했다'는 표현만으로 해당 경기를 완전히 설명하기에는 맥락이 부족합니다.",
        "verdict": "TRUE",
        "reason": "검색 결과로 2002 FIFA 월드컵 준결승에서 한국이 독일에 0-1로 패배한 것이 확인됩니다.",
        "quality_score": "0.90",
        "feedback": "판단이 정확합니다. 경기 스코어가 포함되어 근거의 신뢰도가 높습니다.",
    },
    {
        "text": "애플의 첫 5G 스마트폰인 아이폰 12가 2020년 출시되었으며, 이는 세계 최초의 5G 스마트폰이다.",
        "expected_claims": "아이폰 12 최초 5G 스마트폰",
        "claim": "아이폰 12는 세계 최초의 5G 스마트폰이다",
        "claim_con": "세계 최초의 상용 5G 스마트폰은 2019년 출시된 삼성 갤럭시 S10 5G이며, 아이폰 12는 애플 최초 5G 모델이지 세계 최초가 아니다",
        "evidence": "삼성전자 갤럭시 S10 5G가 2019년 4월 세계 최초로 출시된 5G 스마트폰. 아이폰 12는 애플 최초 5G 모델(2020).",
        "debate_pro": "아이폰 12는 애플 역사상 처음으로 5G를 지원한 스마트폰이며, 전 세계적으로 가장 많이 판매된 5G 스마트폰 중 하나입니다. '세계 최초'라는 표현이 상업적 영향력 기준이라면 아이폰 12가 5G 대중화를 이끌었다고 볼 수 있습니다.",
        "debate_con": "삼성전자 갤럭시 S10 5G가 2019년 4월에 출시되어 세계 최초의 상용 5G 스마트폰으로 공식 기록되어 있습니다. 아이폰 12는 2020년 출시로, 시간적으로 1년 이상 늦습니다. '최초'의 기준이 출시일이라면 아이폰 12는 세계 최초가 아닙니다.",
        "verdict": "FALSE",
        "reason": "세계 최초의 5G 스마트폰은 2019년 삼성전자가 출시한 갤럭시 S10 5G입니다. 아이폰 12는 애플 최초 5G 모델이지만 세계 최초가 아닙니다.",
        "quality_score": "0.93",
        "feedback": "근거가 명확하고 판정이 정확합니다. 출시 날짜와 모델명이 정확히 명시되어 있습니다.",
    },
    {
        "text": "코로나19 mRNA 백신은 2020년에 처음 개발된 기술을 기반으로 하며, 기존에는 사용된 적 없는 완전히 새로운 방식이다.",
        "expected_claims": "mRNA 백신 2020년 최초 개발",
        "claim": "mRNA 백신 기술은 2020년에 처음 개발됐다",
        "claim_con": "mRNA 기술 자체는 1990년대부터 연구되어 온 기술로, 2020년은 최초 개발이 아닌 최초 대규모 상용화 시점이다",
        "evidence": "mRNA 기술 연구는 1990년대부터 진행됨. 코로나19 이전 암 치료 분야에서도 mRNA 기반 연구 진행 중이었음.",
        "debate_pro": "코로나19 mRNA 백신은 기존에 대규모 임상 승인을 받은 적 없는 새로운 형태의 백신입니다. 2020년 이전까지 mRNA 기술이 실제 승인된 백신으로 출시된 사례가 없으므로, 실용화 관점에서 2020년이 최초라고 볼 수 있습니다.",
        "debate_con": "mRNA 기술 자체는 1990년대 카탈린 커리코 박사 등의 연구로 시작되었으며, 코로나19 이전에도 암 면역치료 연구에 활발히 사용되고 있었습니다. 2020년은 최초 개발이 아니라 최초 대규모 상용화 시점입니다.",
        "verdict": "FALSE",
        "reason": "mRNA 기술은 1990년대부터 연구되어 온 기술입니다. 코로나19 백신에 처음 대규모로 적용된 것이지 기술 자체가 2020년에 개발된 것이 아닙니다.",
        "quality_score": "0.88",
        "feedback": "판정이 올바릅니다. 기술 역사에 대한 참고 문헌을 추가하면 더욱 설득력이 높아집니다.",
    },
    {
        "text": "독도는 대한민국 동해에 위치한 섬으로 경상북도 울릉군에 속하며, 현재 대한민국이 실효 지배하고 있다.",
        "expected_claims": "독도 경상북도 울릉군 소속, 대한민국 실효 지배",
        "claim": "독도는 행정구역상 경상북도 울릉군에 속한다",
        "claim_con": "독도 행정구역 귀속은 대한민국 기준이며, 일본은 독도를 시마네현 소속이라 주장하는 영유권 분쟁이 진행 중이다",
        "evidence": "독도는 행정구역상 경상북도 울릉군 울릉읍 독도리에 속함. 대한민국 경찰 및 주민 거주 확인됨.",
        "debate_pro": "대한민국 행정구역 상 독도는 경상북도 울릉군 울릉읍 독도리로 명확히 등재되어 있습니다. 대한민국 경찰이 상주하고 주민등록이 가능하며, 실효 지배 중임이 공식적으로 확인됩니다.",
        "debate_con": "일본은 독도를 시마네현 오키노시마초 소속이라고 주장하며 영유권 분쟁이 진행 중입니다. 국제법적으로 영유권이 완전히 확정된 상태가 아니므로 '경상북도 울릉군 소속'이라는 표현이 국제적으로 보편적으로 인정된다고 단정하기 어렵습니다.",
        "verdict": "TRUE",
        "reason": "독도가 경상북도 울릉군 소속임은 대한민국 행정구역 기준으로 공식 확인된 사실입니다.",
        "quality_score": "0.96",
        "feedback": "판정이 명확하고 근거가 충분합니다. 법적 근거를 추가하면 완벽합니다.",
    },
    {
        "text": "서울은 세계에서 인구밀도가 가장 높은 도시로, 1㎢당 인구가 뉴욕보다 10배 이상 높다.",
        "expected_claims": "서울 세계 최고 인구밀도",
        "claim": "서울은 세계에서 인구밀도가 가장 높은 도시다",
        "claim_con": "세계 인구밀도 1위는 방글라데시 다카 또는 인도 뭄바이 등이며, 서울이 세계에서 가장 높다는 주장은 공식 통계에 부합하지 않는다",
        "evidence": "세계 인구밀도 1위는 방글라데시 다카로 알려짐. 서울의 인구밀도는 약 16,000명/㎢으로 높지만 세계 1위는 아님.",
        "debate_pro": "서울은 약 16,000명/㎢의 인구밀도로 세계 주요 대도시 중 최상위권에 속합니다. 단순 행정구역이 아닌 실제 생활권 기준으로 보면 서울의 인구 집중도는 세계적으로 손꼽히는 수준입니다.",
        "debate_con": "방글라데시 다카, 인도 뭄바이 등은 서울보다 높은 인구밀도를 기록하고 있습니다. 공식 통계에 따르면 서울이 세계 1위라는 근거는 없으며, 이 주장은 사실과 다릅니다.",
        "verdict": "FALSE",
        "reason": "서울의 인구밀도는 세계적으로 높은 편이나, 세계 1위는 다카(방글라데시) 등 다른 도시입니다.",
        "quality_score": "0.87",
        "feedback": "판정이 올바릅니다. 비교 대상 도시의 수치를 함께 제시하면 신뢰도가 향상됩니다.",
    },
    {
        "text": "카카오톡은 대한민국에서 가장 많이 사용되는 메신저 앱으로, 2023년 기준 월간 활성 사용자가 5천만 명을 초과한다.",
        "expected_claims": "카카오톡 국내 1위 메신저, 월간 활성 사용자 5천만 초과",
        "claim": "카카오톡 월간 활성 사용자가 2023년 기준 5천만 명을 초과한다",
        "claim_con": "카카오 공식 발표 기준 2023년 국내 MAU는 약 4,800만 명으로, 5천만 명 초과는 공식 자료로 확인되지 않는 수치다",
        "evidence": "카카오 공식 발표에 따르면 2023년 카카오톡 국내 월간 활성 사용자(MAU)는 약 4,800만 명 수준으로 보고됨.",
        "debate_pro": "대한민국 인구가 약 5,100만 명임을 감안할 때, 카카오톡 MAU가 5천만에 근접한다는 것은 사실상 전 국민이 사용한다는 의미입니다. 해외 거주 한국인까지 포함하면 5천만을 초과할 가능성을 배제할 수 없습니다.",
        "debate_con": "카카오 공식 발표 기준 2023년 국내 MAU는 약 4,800만 명으로, 5천만 초과라는 주장과 200만 명의 차이가 있습니다. 공식 자료에 근거한 정확한 수치를 사용해야 하며, 5천만 초과는 확인되지 않은 수치입니다.",
        "verdict": "UNVERIFIABLE",
        "reason": "카카오톡이 국내 최다 사용 메신저인 것은 맞으나, '5천만 명 초과'라는 수치는 공식 자료로 정확히 확인하기 어렵습니다.",
        "quality_score": "0.78",
        "feedback": "UNVERIFIABLE 판정은 적절하나, 수치의 출처를 더 구체적으로 조사하면 TRUE/FALSE 판정이 가능할 수 있습니다.",
    },
]



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
    # ClaimExtractorSignature의 output이 claims_pro / claims_con.
    predicted_pro = getattr(prediction, "claims_pro", "").lower()
    predicted_con = getattr(prediction, "claims_con", "").lower()
    predicted = f"{predicted_pro} {predicted_con}"
    if not predicted.strip():
        return False

    # 정답 키워드 목록 (쉼표 구분, 2글자 이하 단어는 제외)
    keywords = []
    for kw in example.expected_claims.split(","):
        if len(kw.strip()) > 2:
            keywords.append(kw.strip())

    # 단순 키워드 매칭이면 주장 내용에 키워드가 없다는 이유로 누락될 수 있어서
    # rapidfuzz를 이용해 두 문자열이 얼마나 비슷한지 판단.
    for kw in keywords:
        # if kw.lower() in predicted:
        #     return True
        if 80 <= rapidfuzz.fuzz.partial_ratio( kw.lower(), predicted ):
            return True
        

    return False


def _debate_judge_metric(example: dspy.Example, prediction: dspy.Prediction, trace: object = None) -> bool:
    """
    찬상&반대 에이전트의 토론 결과를 판정하는 함수.
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
    logger.info("컴파일/로드 시도: %s", save_path)

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
    global claim_extractor, llm_judge, agent_debate, debate_judge

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


    # 주장(claim)과 근거(evidence)를 가탕으로 찬성/반대 에이전트가 각각 논거를 제시하는 하는 모듈.
    # 정답을 정할 수 없는 창의적(?) Task라 Few-Shot을 쓰기 애매하고, metric 평가도 애매하므로 컴파일은 하지 않음.
    agent_debate = dspy.ChainOfThought(AgentDebateSignature)


    # 주장(claim), 근거(evidence), 찬성 논거(pro_argument), 반대 논거(con_argument)를 가탕으로 최종 판단(verdict)과 그 이유(reason)를 컴파일.
    debate_judge_trainset = []
    for d in _TRAINING_DATA:
        example = dspy.Example(
            claim_pro=d["claim"],
            claim_con=d["claim_con"],
            evidence_pro=d["evidence"],
            evidence_con=d["evidence"],
            debate_pro=d["debate_pro"],
            debate_con=d["debate_con"],
            verdict=d["verdict"],
            reason=d["reason"],
        ).with_inputs("claim_pro", "claim_con", "evidence_pro", "evidence_con", "debate_pro", "debate_con")
        debate_judge_trainset.append(example)

    debate_judge = _compile_or_load(
        module=dspy.ChainOfThought(DebateJudgeSignature),
        save_path=os.path.join(_COMPILED_DIR, "debate_judge.json"),
        trainset = debate_judge_trainset,
        metric = _debate_judge_metric
    )

    # 주장(claim)과 근거(evidence)를 가탕으로 사실인지(verdict) 판단한 이유(reason)에 대한 점수(quality_score)와 피드백(feedback)을 컴파일.
    # 주장, 근거, 사실, 판단 이유가 입력값.
    llm_judge_trainset = []
    for d in _TRAINING_DATA:
        example = dspy.Example(
            claim_pro=d["claim"],
            claim_con=d["claim_con"],
            evidence_pro=d["evidence"],
            evidence_con=d["evidence"],
            debate_pro=d["debate_pro"],
            debate_con=d["debate_con"],
            verdict=d["verdict"],
            reason=d["reason"],
            quality_score=d["quality_score"],
            feedback=d["feedback"],
        ).with_inputs("claim_pro", "claim_con", "evidence_pro", "evidence_con", "debate_pro", "debate_con", "verdict", "reason")
        llm_judge_trainset.append(example)

    # dspy.Predict 사용.
    # 전체적인 추론(주장, 근거 등등)은 이미 다른 dspy 모듈에서 다 했으므로 여기서는 추론(CoT)과정 없이 판단만 한다.
    llm_judge = _compile_or_load(
        module=dspy.Predict(LLMJudgeSignature),
        save_path=os.path.join(_COMPILED_DIR, "llm_judge.json"),
        trainset=llm_judge_trainset,
        metric=_llm_judge_metric,
    )

    logger.info("DSPy 모듈 초기화 완료.")
