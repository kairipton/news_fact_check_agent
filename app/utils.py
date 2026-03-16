"""
뉴스 팩트체크 에이전트 - 유틸리티 함수 모음
"""

def clean_news_text(text: str) -> str:
    """뉴스 본문에서 불필요한 공백과 특수문자를 정리합니다."""
    if not isinstance(text, str):
        raise TypeError(f"text는 문자열이어야 합니다. 전달된 타입: {type(text)}")
    return " ".join(text.split())


def is_valid_url(url: str) -> bool:
    """URL 형식이 유효한지 간단히 검사합니다."""
    return url.startswith("http://") or url.startswith("https://")


def truncate_text(text: str, max_length: int = 500) -> str:
    """텍스트를 지정된 길이로 자릅니다."""
    if max_length <= 0:
        raise ValueError("max_length는 1 이상이어야 합니다.")
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_keywords(text: str) -> list[str]:
    """텍스트에서 의미 있는 단어(3글자 이상)를 추출합니다."""
    words = text.split()
    return [w for w in words if len(w) >= 3]
