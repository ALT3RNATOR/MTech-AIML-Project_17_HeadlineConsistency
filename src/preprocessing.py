"""Small preprocessing helpers for the headline/article consistency task."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

STANCE_TO_LABEL = {"agree": 1, "discuss": 1, "disagree": 0, "unrelated": 0}


def map_stance(stance: str) -> int:
    """Map an FNC-1 stance to 1=consistent or 0=inconsistent."""
    key = str(stance).strip().lower()
    if key not in STANCE_TO_LABEL:
        raise ValueError(f"Unknown FNC-1 stance: {stance!r}")
    return STANCE_TO_LABEL[key]


def clean_text(value: object) -> str:
    """Normalize Unicode and whitespace without removing meaningful words."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def map_stances(values: Iterable[str]) -> list[int]:
    """Vector-friendly wrapper for `map_stance`."""
    return [map_stance(value) for value in values]
