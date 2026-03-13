"""
app/utils.py 에 대한 단위 테스트
"""
import pytest
from app.utils import clean_news_text, is_valid_url, truncate_text, extract_keywords


class TestCleanNewsText:
    def test_removes_extra_spaces(self, sample_news_text):
        result = clean_news_text(sample_news_text)
        assert result == "정부는 오늘 새로운 경제 정책을 발표했습니다."

    def test_already_clean_text(self):
        text = "깨끗한 텍스트입니다."
        assert clean_news_text(text) == text

    def test_empty_string(self):
        assert clean_news_text("") == ""

    def test_raises_on_non_string(self):
        with pytest.raises(TypeError):
            clean_news_text(12345)


class TestIsValidUrl:
    def test_https_url(self, sample_url):
        assert is_valid_url(sample_url) is True

    def test_http_url(self):
        assert is_valid_url("http://example.com") is True

    def test_invalid_url(self):
        assert is_valid_url("ftp://example.com") is False

    def test_plain_text_not_url(self):
        assert is_valid_url("not-a-url") is False


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "짧은 텍스트"
        assert truncate_text(text, max_length=100) == text

    def test_long_text_truncated(self, long_text):
        result = truncate_text(long_text, max_length=500)
        assert len(result) == 503  # 500 + len("...")
        assert result.endswith("...")

    def test_exact_length_not_truncated(self):
        text = "a" * 500
        assert truncate_text(text, max_length=500) == text

    def test_raises_on_zero_max_length(self):
        with pytest.raises(ValueError):
            truncate_text("텍스트", max_length=0)


class TestExtractKeywords:
    def test_extracts_long_words(self):
        text = "AI 기반 뉴스 팩트체크 시스템"
        keywords = extract_keywords(text)
        assert "팩트체크" in keywords
        assert "시스템" in keywords

    def test_filters_short_words(self):
        text = "AI 기반 뉴스"
        keywords = extract_keywords(text)
        assert "AI" not in keywords  # 2글자 제외
        assert "기반" not in keywords  # 2글자 제외
        assert "뉴스" not in keywords  # 2글자 제외

    def test_empty_text_returns_empty_list(self):
        assert extract_keywords("") == []


@pytest.mark.parametrize("url,expected", [
    ("https://naver.com", True),
    ("http://daum.net", True),
    ("www.google.com", False),
    ("", False),
])
def test_is_valid_url_parametrize(url, expected):
    assert is_valid_url(url) is expected
