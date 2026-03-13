"""
pydantic_settings 기반 환경 설정 모듈.

.env 파일에서 환경변수를 로드합니다.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수로 관리되는 애플리케이션 설정."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    """OpenAI API 키."""

    tavily_api_key: str
    """Tavily 검색 API 키."""

    model_name: str = "gpt-4.1-nano"
    """사용할 OpenAI 모델명."""

    max_correction_retries: int = 3
    """Self-Correction 최대 재시도 횟수."""

    correction_threshold: float = 0.7
    """Self-Correction을 건너뛸 LLM Judge 최소 기준 점수."""


settings = Settings()
