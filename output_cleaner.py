import re


FORBIDDEN_SECTION_HEADERS = (
    "evidence", "evidence points", "references",
    "citation", "citations",
    "note",
    "summary",
    "unresolved point",
    "actionable next step",
)

STRIP_PREFIXES = (
    "here's my response",
    "here is my response",
    "here is the response",
    "please note that",
    "(note:",
)

def clean_agent_output(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    lines = cleaned.splitlines()

    kept = []
    for line in lines:
        normalized = line.strip().lower().rstrip(":")
        if normalized in FORBIDDEN_SECTION_HEADERS:
            break
        # 过滤开头的废话行
        if any(normalized.startswith(p) for p in STRIP_PREFIXES):
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()