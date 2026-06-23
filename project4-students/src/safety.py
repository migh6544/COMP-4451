from __future__ import annotations

import re

BLOCKED_PATTERNS = [
    r"\b(make|build|manufacture)\b.*\b(bomb|explosive|weapon)\b",
    r"\bsteal\b.*\b(api key|password|credential|token)\b",
    r"\bmalware|ransomware|keylogger|phishing\b",
]


def validate_query(query: str) -> tuple[bool, str]:
    cleaned = query.strip()
    if not cleaned:
        return False, "Please enter a question."
    if len(cleaned) > 2000:
        return False, "Please shorten the question."
    lowered = cleaned.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return False, "I can't help process that request. Please ask a safe question about the indexed documents."
    return True, ""
