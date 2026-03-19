import re


FORBIDDEN_SECTION_HEADERS = (
    "evidence",
    "evidence points",
    "references",
    "citation",
    "citations",
)


def clean_agent_output(text: str) -> str:
    """
    Normalize model output for dialogue turns:
    1) remove <think>...</think> traces
    2) drop trailing Evidence/References sections
    3) collapse extra blank lines
    """
    if not text:
        return ""

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    lines = cleaned.splitlines()

    kept = []
    for line in lines:
        normalized = line.strip().lower().rstrip(":")
        if normalized in FORBIDDEN_SECTION_HEADERS:
            break
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
