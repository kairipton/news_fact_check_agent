# News Fact-Check Agent

LangGraph + DSPy + Multi Agent + Debate Prompting + Self Correction + LLM as a Judge 기반 사실 확인 검사 에이전트.

> 이 포트폴리오는 이전 포트폴리오에서 제대로 적용하지 못했거나, 오버엔지니어링이라 판단되어 적용하지 않았던 기술들을 이용해 별도의 포트폴리오로 만든 결과물입니다.

> 이전 포트폴리오: [쿠팡 개인정보 유출 사고 분석 AI 에이전트](https://github.com/kairipton/coupang-incident-analysis-agent-python)

**[▶ 데모 바로가기](https://factcheck.mbh.watch/)**

---

## 주요 기술 스택

| 기술 | 역할 |
|---|---|
| **LangGraph** | 파이프라인 흐름 제어 (멀티에이전트, 서브그래프, Self-Correction 루프) |
| **DSPy** | LLM 호출 추상화 (Signature 선언, BootstrapFewShot 컴파일, few-shot 캐시) |
| **OpenAI** | LLM 백엔드 (`gpt-4.1-nano`) |
| **Tavily** | 뉴스 검색 API (근거 수집) |
| **Streamlit** | 웹 UI (노드 단위 실시간 스트리밍) |
| **rapidfuzz** | 편집 거리 기반 문자열 유사도 (DSPy metric 개선) |
| **LangSmith** | 파이프라인 실행 트레이싱 및 모니터링 |
| **pydantic-settings** | `.env` 기반 환경변수 관리 |
| **배포** | Docker, Contabo VPS, nginx |

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
── FactCheckAgent (서브그래프) ──────────────────────
    │
    ├─ debate          ← DSPy ChainOfThought: 찬성/반대 논거 생성 (Debate Prompting)
    │
    ├─ debate_judge    ← DSPy ChainOfThought: 찬반 논거 검토 후 최종 판정
    │
    ├─ llm_judge       ← DSPy Predict: 판정 품질 평가 (0~1)
    │
    └─ 점수 < 기준 AND 재시도 횟수 미달
           │
       self_correction  ← 재시도 카운터 증가 후 debate 재진입
─────────────────────────────────────────────────────
    │
    ▼
report_generator       ← 찬반 토론 + 최종 판정 마크다운 리포트 생성
    │
    ▼
최종 리포트 출력
```

### 멀티에이전트 구조

- **오케스트레이터** (`main.py`): 전체 순차 흐름 관리
- **FactCheckAgent** (`agents.py`): Debate Prompting + LLM Judge + Self-Correction 루프를 서브그래프로 분리

---

## DSPy 활용 방식
```python
class AgentDebateSignature(dspy.Signature):
    claim: str = dspy.InputField(desc="논의할 주장")
    evidence: str = dspy.InputField(desc="논의에 참고할 근거 텍스트")
    debate_pro: str = dspy.OutputField(desc="찬성 에이전트의 논거")
    debate_con: str = dspy.OutputField(desc="반대 에이전트의 논거")
```

| 모듈 | DSPy 타입 | 컴파일 | 이유 |
|---|---|---|---|
| `claim_extractor` | ChainOfThought | ✅ | 텍스트에서 주장 추론 필요 |
| `fact_judge` | ChainOfThought | ✅ | 근거 분석 후 판정 추론 필요 |
| `agent_debate` | ChainOfThought | ❌ | 찬반 논거는 정답이 없어 metric 작성 불가 |
| `debate_judge` | ChainOfThought | ✅ | 최종 판정(TRUE/FALSE/UNVERIFIABLE) metric 사용 |
| `llm_judge` | Predict | ✅ | 완성된 결과를 평가만 하므로 추론 불필요 |

---

## 핵심 구현 포인트 및 기술 채택 이유

### 1. LangGraph
- LangChain의 LCEL은 선형 체인 구조라 조건 분기, 루프, 서브그래프 같은 복잡한 흐름을 표현하기 어려움
- LangGraph는 노드/엣지 기반 그래프 구조로 `Self-Correction` 루프, 서브그래프 분리 등을 명시적으로 표현 가능
- 각 노드의 입출력이 `State` 타입에 맞게 보장되어 디버깅과 유지보수가 용이함

### 2. Multi Agent
- 파이프라인을 단일 그래프로 구성하면 조건 루프가 섞여 흐름 파악이 어려워짐
- `Self-Correction` 루프처럼 내부 반복 구조가 있는 부분만 서브그래프로 분리
- 역할 경계가 명확해져 각 에이전트를 독립적으로 교체하거나 확장할 수 있음

### 3. DSPy
- 프롬프트를 문자열로 하드코딩하면 모델 변경 시 전부 재작성해야 하고 품질 측정이 어려움
- DSPy는 `Signature`로 입출력 사양만 선언하면 프롬프트를 자동 생성하고, `BootstrapFewShot`으로 trainset 기반 few-shot 예시를 자동 선별해 응답 품질을 최적화
- 컴파일 결과를 캐시해 이후 실행에서는 추가 API 비용 없이 재사용

### 4. LLM as a Judge
- LLM이 직접 판정 결과의 품질을 0.0~1.0 점수로 평가하고 개선 피드백을 생성
- 사람이 정답을 레이블링하지 않아도 품질 기준을 자동으로 적용할 수 있음
- 점수가 기준 미달일 경우 `Self-Correction` 루프를 트리거하는 게이트 역할을 겸함

### 5. Self Correction
- LLM as a Judge의 점수가 기준 미달(`correction_threshold=0.7`)일 때 자동으로 재시도하는 루프
- 재진입 시 이전 판정에 대한 Judge 피드백을 근거에 추가해 LLM이 개선된 판단을 내리도록 유도
- 무한 루프 방지를 위해 최대 재시도 횟수(`max_correction_retries=3`)를 설정값으로 관리

### 6. Debate Prompting
- 단일 LLM에게 TRUE/FALSE를 묻는 방식은 편향된 판정을 낼 수 있다는 한계가 있음
- 같은 LLM을 찬성/반대 두 역할로 나눠 각각 논거를 생성하게 한 뒤, Judge가 양측을 검토해 판정
- 단순 판단보다 다각도 분석이 가능하고, 리포트에 찬반 논거가 함께 출력되어 사용자에게 판단 근거를 투명하게 제공

### 7. Rapidfuzz
- 사용자 입력에 대한 주장이 올바르게 생성되었는지 확인할 때, 키워드 매칭만으로는 한계가 있다 판단
- 예) 정답 키워드 `"BTS 빌보드"` vs 예측 결과 `"BTS가 빌보드 핫100에서"` → 단순 매칭은 False
- 그렇다고 유사도 검색을 넣기엔 오버엔지니어링
- rapidfuzz의 `partial_ratio`를 사용해 편집 거리 기반 유사도 80점 이상이면 합격으로 처리, 표현 변형에도 robust한 metric 구현