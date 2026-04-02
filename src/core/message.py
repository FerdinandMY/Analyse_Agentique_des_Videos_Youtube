from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MessageEnvelope:
    """
    Envelope metadata shared across agent-to-agent communications.
    """

    msg_type: str
    sender: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class AgentMessage:
    """
    Message container: envelope + payload.
    """

    envelope: MessageEnvelope
    payload: Dict[str, Any]