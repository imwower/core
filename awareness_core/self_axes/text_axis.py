"""Text axis handling external/internal language representations."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional

from .base_axis import BaseAxis, AxisSummary


class TextAxis(BaseAxis):
    """Tracks external and internal text with a simple deterministic embedding."""

    def __init__(self, name: str = "text", dim: int = 12) -> None:
        super().__init__(name=name, dim=dim)
        self.external_text: Optional[str] = None
        self.internal_text: Optional[str] = None
        self._vector: List[float] = [0.0 for _ in range(dim)]

    def reset(self) -> None:
        self.external_text = None
        self.internal_text = None
        self._vector = [0.0 for _ in range(self.dim)]

    def update_from_input(self, features: Mapping[str, Any]) -> None:
        external = features.get("external_text")
        internal = features.get("internal_text")
        if external is not None:
            self.external_text = str(external)
        if internal is not None:
            self.internal_text = str(internal)
        combined = " ".join(filter(None, [self.external_text, self.internal_text])) or ""
        self._vector = self._encode_text(combined)

    def to_vector(self) -> List[float]:
        return list(self._vector)

    def summary(self) -> AxisSummary:
        return AxisSummary(
            name=self.name,
            dim=self.dim,
            extras={
                "external_text": (self.external_text[:50] + "...") if self.external_text and len(self.external_text) > 53 else self.external_text,
                "internal_text": (self.internal_text[:50] + "...") if self.internal_text and len(self.internal_text) > 53 else self.internal_text,
            },
        )

    def _encode_text(self, text: str) -> List[float]:
        if not text:
            return [0.0 for _ in range(self.dim)]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        ints = list(digest)
        vector: List[float] = []
        for i in range(self.dim):
            value = ints[i % len(ints)]
            vector.append((value / 255.0) * 2 - 1)
        return vector
