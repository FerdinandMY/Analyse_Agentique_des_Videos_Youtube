from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Base class for all specialized agents in the multi-agent pipeline.

    Agents should keep I/O boundaries clean:
    - accept a payload
    - return a payload
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, payload: Any) -> Any:
        raise NotImplementedError