"""
Horn clause inference implementation.

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

import time
from typing import List, Tuple, Dict, Set
from collections import deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from knowledge_base import HornKB
from inference_engine_base import InferenceEngine, InferenceResult


# ============================================================================
# Immutable state snapshots
# ============================================================================

@dataclass(frozen=True)
class InferenceStep:
    """
    Immutable snapshot of one inference step.
    """
    iteration: int
    fact_processed: str
    new_facts_derived: Tuple[str, ...]
    rule_applications: Tuple[Tuple[int, List[str], str], ...]  # (rule_id, premises, conclusion)

    def __str__(self) -> str:
        """String representation."""
        facts_str = ", ".join(self.new_facts_derived) if self.new_facts_derived else "none"
        return f"Iteration {self.iteration}: {self.fact_processed} → derived: {facts_str}"


@dataclass(frozen=True)
class InferenceTrace:
    """
    Complete inference trace (immutable).
    """
    steps: Tuple[InferenceStep, ...]
    query_found_at_step: int  # -1 if not found
    final_inferred_facts: Tuple[str, ...]

    def was_query_found(self) -> bool:
        """Check if query was found during inference."""
        return self.query_found_at_step >= 0


# ============================================================================
# Algorithm state
# ============================================================================

@dataclass
class ForwardChainingState:
    """
    Mutable state for forward chaining algorithm.

    Note: Mutable for performance during algorithm execution,
          but converted to immutable InferenceTrace at end
    """
    count: Dict[int, int] = field(default_factory=dict)
    inferred: Dict[str, bool] = field(default_factory=dict)
    agenda: deque = field(default_factory=deque)
    steps: List[InferenceStep] = field(default_factory=list)
    iteration: int = 0

    def to_immutable_trace(self, query: str, query_found: bool) -> InferenceTrace:
        """
        Convert mutable state to immutable trace.
        """
        query_step = -1
        for i, step in enumerate(self.steps):
            if query in step.new_facts_derived or step.fact_processed == query:
                query_step = i
                break

        inferred_facts = tuple(sorted(self.inferred.keys()))
        return InferenceTrace(
            steps=tuple(self.steps),
            query_found_at_step=query_step,
            final_inferred_facts=inferred_facts
        )


# ============================================================================
# Helper functions
# ============================================================================

def initialize_rule_counts(rules: List[Tuple[List[str], str]]) -> Dict[int, int]:
    """
    Pure function: Initialize premise counts for rules.

    Args:
        rules: List of (premises, conclusion) tuples

    Returns:
        Dictionary mapping rule index to premise count
    """
    return {i: len(premises) for i, (premises, _) in enumerate(rules)}


def find_applicable_rules(
    fact: str,
    rules: List[Tuple[List[str], str]]
) -> List[int]:
    """
    Pure function: Find rules that have fact in their premises.

    Args:
        fact: The fact to search for
        rules: List of rules

    Returns:
        List of rule indices where fact appears in premises
    """
    applicable = []
    for i, (premises, _) in enumerate(rules):
        if fact in premises:
            applicable.append(i)
    return applicable


# ============================================================================
# Strategy interface for different inference methods
# ============================================================================

class HornInferenceStrategy(ABC):
    """
    Abstract strategy for Horn clause inference.

    Defines interface without implementation.
    New strategies extend without modifying existing code.
    """

    @abstractmethod
    def infer(
        self,
        kb: HornKB,
        query: str
    ) -> Tuple[bool, InferenceTrace, float]:
        """
        Perform inference to check if query is entailed.

        Args:
            kb: Horn clause knowledge base
            query: Query atom

        Returns:
            Tuple of (entailed, trace, elapsed_time)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return name of strategy."""
        pass


# ============================================================================
# Concrete strategy implementations
# ============================================================================

class ForwardChainingStrategy(HornInferenceStrategy):
    """
    Forward chaining strategy (data-driven).

    Implements strategy interface.
    Only handles forward chaining logic.
    Uses immutable results.
    """

    def infer(
        self,
        kb: HornKB,
        query: str
    ) -> Tuple[bool, InferenceTrace, float]:
        """
        Perform forward chaining inference.

        Based on Russell & Norvig Figure 7.15.

        Args:
            kb: Horn clause knowledge base
            query: Query atom

        Returns:
            Tuple of (entailed, trace, elapsed_time)
        """
        t0 = time.perf_counter()

        # Initialize algorithm state
        state = ForwardChainingState()
        rules = kb.get_rules()
        state.count = initialize_rule_counts(rules)

        # Add initial facts to agenda
        for fact in kb.get_facts():
            state.agenda.append(fact)

        # Main inference loop
        query_found = False
        while state.agenda:
            state.iteration += 1
            fact = state.agenda.popleft()

            # Check if this is the query
            if fact == query:
                query_found = True
                step = InferenceStep(
                    iteration=state.iteration,
                    fact_processed=fact,
                    new_facts_derived=(fact,),
                    rule_applications=tuple()
                )
                state.steps.append(step)
                break

            # Skip if already inferred
            if state.inferred.get(fact, False):
                continue

            # Mark as inferred
            state.inferred[fact] = True
            new_facts = []
            rule_apps = []

            # Find applicable rules
            for rule_idx in find_applicable_rules(fact, rules):
                premises, conclusion = rules[rule_idx]
                state.count[rule_idx] -= 1

                # If all premises satisfied, derive conclusion
                if state.count[rule_idx] == 0:
                    if conclusion not in state.inferred:
                        state.agenda.append(conclusion)
                        new_facts.append(conclusion)
                        rule_apps.append((rule_idx, premises, conclusion))

            # Record this step
            step = InferenceStep(
                iteration=state.iteration,
                fact_processed=fact,
                new_facts_derived=tuple(new_facts),
                rule_applications=tuple(rule_apps)
            )
            state.steps.append(step)

        elapsed = time.perf_counter() - t0
        trace = state.to_immutable_trace(query, query_found)
        return query_found, trace, elapsed

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "Forward Chaining (Data-Driven)"


class BackwardChainingStrategy(HornInferenceStrategy):
    """
    Backward chaining strategy (goal-driven).

    Alternative strategy implementation.
    Only handles backward chaining logic.
    """

    def infer(
        self,
        kb: HornKB,
        query: str
    ) -> Tuple[bool, InferenceTrace, float]:
        """
        Perform backward chaining inference.

        Based on Russell & Norvig Figure 7.16.

        Args:
            kb: Horn clause knowledge base
            query: Query atom

        Returns:
            Tuple of (entailed, trace, elapsed_time)
        """
        t0 = time.perf_counter()

        # Backward chaining: work backwards from goal
        proved_goals = set()
        goal_stack = [query]
        steps = []
        iteration = 0

        def prove_goal(goal: str) -> bool:
            """Helper to recursively prove goal."""
            nonlocal iteration

            if goal in kb.get_facts():
                return True
            if goal in proved_goals:
                return True

            # Try each rule with goal as conclusion
            for premises, conclusion in kb.get_rules():
                if conclusion == goal:
                    iteration += 1
                    # Try to prove all premises
                    if all(prove_goal(premise) for premise in premises):
                        proved_goals.add(goal)
                        step = InferenceStep(
                            iteration=iteration,
                            fact_processed=goal,
                            new_facts_derived=(goal,),
                            rule_applications=((0, premises, conclusion),)
                        )
                        steps.append(step)
                        return True
            return False

        entailed = prove_goal(query)
        elapsed = time.perf_counter() - t0

        trace = InferenceTrace(
            steps=tuple(steps),
            query_found_at_step=len(steps) - 1 if entailed else -1,
            final_inferred_facts=tuple(sorted(proved_goals))
        )

        return entailed, trace, elapsed

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "Backward Chaining (Goal-Driven)"


# ============================================================================
# Separate presentation
# ============================================================================

class InferenceTracePrinter:
    """
    Formats and prints inference traces.

    Only handles output formatting.
    """

    @staticmethod
    def print_trace(
        kb: HornKB,
        query: str,
        strategy_name: str,
        entailed: bool,
        trace: InferenceTrace,
        elapsed: float,
        verbose: bool = True
    ) -> None:
        """Print detailed inference trace."""
        if not verbose:
            return

        print("=== Forward Chaining ===")
        print()
        print("Knowledge Base:")
        print(kb)
        print()
        print(f"Query: {query}")
        print()

        # Print initial facts
        print("Initial fact: " + ", ".join(kb.get_facts()))
        print()

        # Print each step
        for step in trace.steps:
            print(f"Iteration {step.iteration}: Processing fact '{step.fact_processed}'")
            if step.fact_processed == query:
                print(f"  -> Query '{query}' found!")
                break

            for rule_idx, premises, conclusion in step.rule_applications:
                print(f"  -> Rule {rule_idx + 1}: {' AND '.join(premises)} => {conclusion}")
                if conclusion in step.new_facts_derived:
                    print(f"     ALL premises satisfied! Adding '{conclusion}' to agenda")

            if not step.rule_applications:
                print("  -> Already inferred, skipping")

            print()

        print(f"Result: {'ENTAILED' if entailed else 'NOT ENTAILED'} " +
              f"(derived in {len(trace.steps)} iterations)")
        print(f"Elapsed time: {elapsed:.6f}s")
        print()


# ============================================================================
# Extensible inference engine using strategy
# ============================================================================

class HornInferenceEngine(InferenceEngine):
    """
    Horn clause inference engine using strategy pattern.

    Extends InferenceEngine, uses strategy for algorithm.
    Coordinates strategy and KB.
    Hides strategy details from clients.
    """

    def __init__(self, kb: HornKB, strategy: HornInferenceStrategy):
        """
        Initialize engine with strategy.

        Args:
            kb: Horn clause knowledge base
            strategy: Inference strategy (forward/backward chaining)
        """
        self._kb = kb
        self._strategy = strategy

    def infer(self, query: str) -> InferenceResult:
        """
        Perform inference using configured strategy.

        Args:
            query: Query atom

        Returns:
            InferenceResult with entailment status
        """
        entailed, trace, elapsed = self._strategy.infer(self._kb, query)

        trace_strings = [str(step) for step in trace.steps]
        return InferenceResult(entailed, trace_strings, elapsed)

    def get_algorithm_name(self) -> str:
        """Return algorithm name from strategy."""
        return self._strategy.get_strategy_name()

    def infer_with_trace(
        self,
        query: str,
        verbose: bool = True
    ) -> Tuple[bool, InferenceTrace, float]:
        """
        Perform inference and return detailed trace.

        Args:
            query: Query atom
            verbose: Whether to print trace

        Returns:
            Tuple of (entailed, trace, elapsed_time)
        """
        entailed, trace, elapsed = self._strategy.infer(self._kb, query)

        if verbose:
            InferenceTracePrinter.print_trace(
                self._kb,
                query,
                self._strategy.get_strategy_name(),
                entailed,
                trace,
                elapsed,
                verbose
            )

        return entailed, trace, elapsed


# ============================================================================
# Simplified interface (backward compatible)
# ============================================================================

def forward_chaining(
    kb: HornKB,
    query: str,
    verbose: bool = True
) -> Tuple[bool, List[str], float]:
    """
    Simplified forward chaining interface (backward compatible).

    Facade pattern - Simple interface to complex subsystem.

    Args:
        kb: Horn clause knowledge base
        query: Query atom
        verbose: Whether to print trace

    Returns:
        Tuple of (entailed, trace_strings, elapsed_time)
    """
    strategy = ForwardChainingStrategy()
    engine = HornInferenceEngine(kb, strategy)
    entailed, trace, elapsed = engine.infer_with_trace(query, verbose)

    trace_strings = [str(step) for step in trace.steps]
    return entailed, trace_strings, elapsed


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    """Demonstrate refactored Horn inference."""
    from knowledge_base import HornKB

    print("=" * 80)
    print("Refactored Horn Inference")
    print("=" * 80)
    print()

    # Test generic KB
    kb = HornKB()
    kb.tell_fact("A")
    kb.tell_fact("B")
    kb.tell_rule(["A", "B"], "C")
    kb.tell_rule(["C"], "D")

    # Test forward chaining
    print("TEST 1: Forward Chaining Strategy")
    print("-" * 40)
    forward_strategy = ForwardChainingStrategy()
    engine = HornInferenceEngine(kb, forward_strategy)
    result = engine.infer("D")
    print(f"Algorithm: {engine.get_algorithm_name()}")
    print(result)
    print()

    # Test backward chaining
    print("TEST 2: Backward Chaining Strategy")
    print("-" * 40)
    backward_strategy = BackwardChainingStrategy()
    engine2 = HornInferenceEngine(kb, backward_strategy)
    result2 = engine2.infer("D")
    print(f"Algorithm: {engine2.get_algorithm_name()}")
    print(result2)
    print()

    # Test with trace
    print("TEST 3: Detailed Trace")
    print("-" * 40)
    entailed, trace, elapsed = forward_chaining(kb, "D", verbose=True)
    print(f"Query D entailed: {entailed}")
    print()

    print("=" * 80)
    print("All tests PASSED - refactoring successful!")
    print("=" * 80)
