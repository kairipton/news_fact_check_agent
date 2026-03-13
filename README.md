# News Fact-Check Agent

LangGraph + DSPy + Streamlit 기반 뉴스 팩트체크 멀티에이전트 파이프라인.

---

## 주요 기술 스택

| 기술 | 역할 |
|---|---|
| **LangGraph** | 파이프라인 흐름 제어 (멀티에이전트, 서브그래프, Self-Correction 루프) |
| **DSPy** | LLM 호출 추상화 (Signature 선언, BootstrapFewShot 컴파일, few-shot 캐시) |
| **OpenAI** | LLM 백엔드 (`gpt-4.1-nano`) |
| **Tavily** | 뉴스 검색 API (근거 수집) |
| **Streamlit** | 웹 UI |
| **pydantic-settings** | `.env` 기반 환경변수 관리 |

---

## 아키텍처

### 전체 파이프라인 흐름

```
사용자 입력 (뉴스 텍스트)
    │
    ▼
claim_extractor        ← DSPy ChainOfThought: 주장 추출
    │
    ▼
evidence_searcher      ← Tavily API: 주장별 근거 검색
    │
    ▼
── FactCheckAgent (서브그래프) ──
        fact_judge       ← DSPy ChainOfThought: 사실 판단
            │
        llm_judge        ← DSPy Predict: 품질 평가 (0~1)
            │
        점수 < 기준 AND 재시도 횟수 미달
            │
        self_correction  ← 재시도 카운터 증가 후 fact_judge 재진입
─────────────────────────────────
    │
    ▼
report_generator       ← 마크다운 리포트 생성
    │
    ▼
최종 리포트 출력
```

### 멀티에이전트 구조

- **오케스트레이터** (`main.py`): 전체 순차 흐름 관리
- **FactCheckAgent** (`agents.py`): 사실 판단 + LLM Judge + Self-Correction 루프를 서브그래프로 분리

`FactCheckAgent`만 서브그래프로 분리한 이유: 내부에 조건부 루프(Self-Correction)가 있는 유일한 에이전트이기 때문.

### Self-Correction 동작

`llm_judge`가 품질 점수(`judge_score`)를 평가한 뒤:

- 점수 ≥ `correction_threshold` (기본 0.7) → 종료
- 점수 < 기준 AND 재시도 < `max_correction_retries` (기본 3) → `self_correction` → `fact_judge` 재진입
- 재진입 시 이전 피드백(`judge_feedback`)을 근거에 추가해 개선된 판단 유도

---

## DSPy 활용 방식

LLM 프롬프트를 코드에 하드코딩하지 않고 **Signature**로 입출력만 선언합니다.

```python
class FactJudgeSignature(dspy.Signature):
    claim: str = dspy.InputField(...)
    evidence: str = dspy.InputField(...)
    verdict: str = dspy.OutputField(...)   # TRUE / FALSE / UNVERIFIABLE
    reasoning: str = dspy.OutputField(...)
```

**BootstrapFewShot**이 trainset을 LLM에 돌려 좋은 few-shot 예시를 자동 선별하고, 결과를 `.compiled/*.json`에 캐시합니다. 이후 실행에서는 컴파일 없이 로드합니다.

| 모듈 | DSPy 타입 | 이유 |
|---|---|---|
| `claim_extractor` | ChainOfThought | 텍스트에서 주장 추론 필요 |
| `fact_judge` | ChainOfThought | 근거 분석 후 판정 추론 필요 |
| `llm_judge` | Predict | 완성된 결과를 평가만 하므로 추론 불필요 |

---

## 프로젝트 구조

```
news_fact_check_agent/
├── state.py          # LangGraph 공유 상태 (FactCheckState TypedDict)
├── config.py         # 환경변수 설정 (pydantic-settings)
├── dspy_modules.py   # DSPy Signature, trainset, 컴파일/로드 관리
├── nodes.py          # LangGraph 노드 함수 6개
├── agents.py         # FactCheckAgent 서브그래프
├── main.py           # 오케스트레이터 그래프 + init_pipeline()
├── app.py            # Streamlit 웹 UI
├── test.py           # CLI 단독 실행 테스트
├── .env              # API 키 (git 제외)
├── .compiled/        # DSPy 컴파일 캐시 (자동 생성)
└── requirements.txt
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. `.env` 파일 생성

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
MODEL_NAME=gpt-4.1-nano
MAX_CORRECTION_RETRIES=3
CORRECTION_THRESHOLD=0.7
```

### 3. 실행

```bash
# CLI 테스트
python test.py

# Streamlit 웹 UI
streamlit run app.py
```

> **최초 실행 시** DSPy가 trainset을 LLM에 돌려 컴파일합니다 (API 비용 발생).  
> 이후에는 `.compiled/` 캐시를 사용하므로 추가 비용 없습니다.

---

## 상태 필드 (FactCheckState)

| 필드 | 타입 | 설명 |
|---|---|---|
| `input_text` | `str` | 사용자 입력 원문 |
| `claims` | `list[str]` | 추출된 주장 목록 |
| `search_results` | `list[dict]` | 주장별 검색 근거 `{"claim", "evidence"}` |
| `judgment_results` | `list[dict]` | 주장별 판단 결과 `{"claim", "verdict", "reasoning"}` |
| `judge_score` | `float` | LLM Judge 품질 점수 (0.0~1.0) |
| `judge_feedback` | `str` | LLM Judge 개선 피드백 |
| `correction_retries` | `int` | Self-Correction 재시도 횟수 |
| `final_report` | `str` | 최종 마크다운 리포트 |
| `error` | `str \| None` | 오류 메시지 |
