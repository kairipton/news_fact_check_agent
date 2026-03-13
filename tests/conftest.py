"""
pytest 공통 fixture 정의
- 모든 테스트에서 재사용 가능한 fixture를 여기에 등록합니다.
"""
import pytest


@pytest.fixture
def sample_news_text() -> str:
    return "  정부는   오늘  새로운  경제  정책을  발표했습니다.  "


@pytest.fixture
def sample_url() -> str:
    return "https://www.example.com/news/12345"


@pytest.fixture
def long_text() -> str:
    return "가" * 600
