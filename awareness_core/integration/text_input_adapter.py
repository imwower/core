"""Text input adapter: raw text -> normalized features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TextFeatures:
    """High-level features extracted from raw text input."""

    external_text: str
    salience: float
    metadata: Dict[str, str]


class TextInputAdapter:
    """Converts raw text strings into TextFeatures for the text axis."""

    def encode(self, text: Optional[str]) -> Optional[TextFeatures]:
        if text is None:
            return None
        stripped = text.strip()
        if not stripped:
            return None
        salience = min(1.0, len(stripped) / 80.0)
        return TextFeatures(external_text=stripped, salience=salience, metadata={"length": str(len(stripped))})
