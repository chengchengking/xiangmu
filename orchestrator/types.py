from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .mode import Mode


@dataclass
class TurnContext:
    turn_id: int
    mode: Mode
    topic_id: Optional[str] = None
    reply_to: Optional[str] = None
    ack: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

