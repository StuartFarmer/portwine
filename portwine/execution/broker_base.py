"""
Abstract interface that any live / paper broker must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .portfolio import Order, Position


class Broker(ABC):
    # ---------- read‑only ----------
    @abstractmethod
    def get_positions(self) -> List[Position]: ...

    @abstractmethod
    def get_cash(self) -> float: ...

    # ---------- actions ------------
    @abstractmethod
    def submit_orders(self, orders: List[Order]) -> None: ...
