"""
Helpers for normalizing POI IDs across data files.
"""
from __future__ import annotations

from typing import Any


def normalize_poi_id(value: Any) -> str:
    """
    Normalize POI ID to a canonical string form.

    Examples:
    - "001001" -> "1001"
    - 1001 -> "1001"
    - "POI_0001" -> "POI_0001"
    """
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text.isdigit():
        return str(int(text))
    return text

