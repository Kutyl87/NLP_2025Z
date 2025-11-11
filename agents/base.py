from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    name: str
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError