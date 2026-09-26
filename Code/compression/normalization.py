from __future__ import annotations


def normalize_text(text: str, lowercase: bool = True, strip: bool = True) -> str:
    """Normalize input text to reduce superficial compression noise."""
    value = text
    if lowercase:
        value = value.lower()
    if strip:
        value = value.strip()
    return value
