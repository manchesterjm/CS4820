"""
Abstract base classes for inference engines.

This module defines interfaces that allow extending inference methods
without modifying existing code.

Based on Russell & Norvig Chapter 7: Logical Agents

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from abc import ABC, abstractmethod
from typing import List, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    """
    Immutable result from inference query.
    """
    entailed: bool
    trace: List[str]
    elapsed_time: float

    def __str__(self) -> str:
        """String representation of result."""
        status = "ENTAILED" if self.entailed else "NOT ENTAILED"
        return f"Result: {status}, Time: {self.elapsed_time:.6f}s"


class InferenceEngine(ABC):
    """
    Abstract base for all inference engines.
    """

    @abstractmethod
    def infer(self, query: str) -> InferenceResult:
        """
        Determine if query is entailed by knowledge base.

        Args:
            query: Query to check for entailment

        Returns:
            InferenceResult with entailment status and trace
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return name of inference algorithm."""
        pass


class KnowledgeBase(ABC):
    """
    Abstract knowledge base interface.
    """

    @abstractmethod
    def tell(self, sentence: Any) -> None:
        """Add knowledge to KB."""
        pass

    @abstractmethod
    def ask(self, query: str) -> bool:
        """Query KB for entailment."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return KB size."""
        pass
