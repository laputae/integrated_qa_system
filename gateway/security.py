import re
from html import escape
from typing import Optional, Tuple

SQL_INJECTION_PATTERNS = [
    r"(?i)UNION\s+SELECT",
    r"(?i)DROP\s+TABLE",
    r"(?i)DROP\s+DATABASE",
    r"(?i)ALTER\s+TABLE",
    r"(?i)TRUNCATE\s+TABLE",
    r"(?i)DELETE\s+FROM",
    r"(?i)INSERT\s+INTO",
    r"(?i)UPDATE\s+\w+\s+SET",
    r"(?i)EXEC\s*\(.*\)",
    r"(?i)EXECUTE\s+IMMEDIATE",
    r"(?i)--\s*$",
    r"(?i)'\s+OR\s+'1'\s*=\s*'1",
    r"(?i)'\s+OR\s+\d+\s*=\s*\d+",
    r"(?i);\s*DROP",
    r"(?i);\s*DELETE",
    r"(?i)SLEEP\s*\(",
    r"(?i)BENCHMARK\s*\(",
    r"(?i)INFORMATION_SCHEMA",
]

XSS_PATTERNS = [
    r"(?i)<script[^>]*>",
    r"(?i)</script>",
    r"(?i)javascript\s*:",
    r"(?i)onerror\s*=",
    r"(?i)onload\s*=",
    r"(?i)onclick\s*=",
    r"(?i)<iframe",
    r"(?i)<embed",
    r"(?i)<object",
    r"(?i)eval\s*\(.*\)",
    r"(?i)document\.cookie",
    r"(?i)alert\s*\(.*\)",
    r"(?i)<img[^>]+onerror",
]

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")


class SecurityFilter:
    @staticmethod
    def detect_sql_injection(value: str) -> Optional[str]:
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, value):
                return f"SQL injection pattern detected: {pattern}"
        return None

    @staticmethod
    def detect_xss(value: str) -> Optional[str]:
        for pattern in XSS_PATTERNS:
            if re.search(pattern, value):
                return f"XSS pattern detected: {pattern}"
        return None

    @staticmethod
    def validate_username(username: str) -> Tuple[bool, Optional[str]]:
        if not USERNAME_PATTERN.match(username):
            return False, "用户名只允许字母、数字、下划线和连字符，长度3-50"
        return True, None

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, Optional[str]]:
        if not (6 <= len(password) <= 128):
            return False, "密码长度必须在6-128个字符之间"
        return True, None

    @staticmethod
    def validate_uuid(value: str) -> Tuple[bool, Optional[str]]:
        uuid_pattern = re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        )
        if not uuid_pattern.match(value):
            return False, "Invalid UUID format"
        return True, None

    @staticmethod
    def sanitize_html(value: str) -> str:
        return escape(value)

    @staticmethod
    def scan(value: str) -> Tuple[bool, Optional[str]]:
        sql_result = SecurityFilter.detect_sql_injection(value)
        if sql_result:
            return False, sql_result
        xss_result = SecurityFilter.detect_xss(value)
        if xss_result:
            return False, xss_result
        return True, None
