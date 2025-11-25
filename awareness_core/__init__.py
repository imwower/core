
"""Awareness Core package.

Provides a minimal, extensible implementation of the awareness-core
architecture described in the repository README.
"""

from .config import CoreConfig
from .core_state import AwarenessFrame, AwarenessState
from .proto_self import ProtoSelf, ProtoState

__all__ = [
    "CoreConfig",
    "AwarenessState",
    "AwarenessFrame",
    "ProtoSelf",
    "ProtoState",
]
