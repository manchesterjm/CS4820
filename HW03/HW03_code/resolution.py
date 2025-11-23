"""
Refactored resolution-based inference.

Design improvements:
- Single Responsibility: Separated parsing, resolution logic, and presentation
- Open/Closed: Extensible CNF converters through strategy pattern
- Functional: Immutable clauses, pure resolution operations
- Abstraction: Clear interfaces for CNF conversion and resolution

Based on Russell & Norvig Section 7.5.2: Resolution Algorithm

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time


# ============================================================================
# Immutable data structures
# ============================================================================

@dataclass(frozen=True)
class Literal:
    """
    Immutable literal representation.

    Frozen dataclass ensures immutability.
    """
    symbol: str
    is_negated: bool

    def __str__(self) -> str:
        """String representation."""
        return f"-{self.symbol}" if self.is_negated else self.symbol

    def negate(self) -> 'Literal':
        """Return negated version of this literal."""
        return Literal(self.symbol, not self.is_negated)

    def __eq__(self, other) -> bool:
        """Equality check."""
        if not isinstance(other, Literal):
            return False
        return self.symbol == other.symbol and self.is_negated == other.is_negated

    def __hash__(self) -> int:
        """Hash for set operations."""
        return hash((self.symbol, self.is_negated))


@dataclass(frozen=True)
class Clause:
    """
    Immutable clause (disjunction of literals).

    Frozen tuple of literals.
    """
    literals: Tuple[Literal, ...]

    def __str__(self) -> str:
        """String representation."""
        if not self.literals:
            return "[]"
        return "[" + " | ".join(str(lit) for lit in self.literals) + "]"

    def is_empty(self) -> bool:
        """Check if this is the empty clause (contradiction)."""
        return len(self.literals) == 0

    def contains_complementary_pair(self) -> bool:
        """Check if clause contains both P and -P (tautology)."""
        lit_set = set(self.literals)
        for lit in self.literals:
            if lit.negate() in lit_set:
                return True
        return False


@dataclass(frozen=True)
class ResolutionStep:
    """
    Immutable record of one resolution step.

    Complete snapshot of resolution operation.
    """
    clause1: Clause
    clause2: Clause
    resolvent: Clause
    resolved_on: Literal

    def __str__(self) -> str:
        """String representation."""
        return (f"Resolve {self.clause1} and {self.clause2} "
                f"on {self.resolved_on} => {self.resolvent}")


# ============================================================================
# Pure helper functions
# ============================================================================

def parse_literal(lit_str: str) -> Literal:
    """
    Pure function: Parse literal string to Literal object.

    No side effects, deterministic.

    Args:
        lit_str: Literal string ("P", "-P", "not P", "!P")

    Returns:
        Immutable Literal object
    """
    lit = lit_str.strip()

    # Handle various negation formats
    if lit.startswith("not "):
        return Literal(lit[4:].strip(), is_negated=True)
    elif lit.startswith("!"):
        return Literal(lit[1:].strip(), is_negated=True)
    elif lit.startswith("-"):
        return Literal(lit[1:].strip(), is_negated=True)
    else:
        return Literal(lit, is_negated=False)


def resolve_clauses(c1: Clause, c2: Clause) -> Optional[Tuple[Clause, Literal]]:
    """
    Pure function: Apply resolution rule to two clauses.

    Pure operation, returns new immutable clause.

    Resolution rule (Russell & Norvig Figure 7.12):
    - If c1 contains literal L and c2 contains -L,
    - Then resolvent = (c1 - {L}) ∪ (c2 - {-L})

    Args:
        c1: First clause
        c2: Second clause

    Returns:
        Tuple of (resolvent clause, literal resolved on) or None if no resolution

    Time Complexity: O(n*m) where n=|c1|, m=|c2|
    """
    # Find complementary pair
    for lit1 in c1.literals:
        negated = lit1.negate()
        if negated in c2.literals:
            # Found complementary pair
            # Build resolvent = (c1 - {lit1}) ∪ (c2 - {negated})
            new_literals = set()

            # Add from c1 except lit1
            for lit in c1.literals:
                if lit != lit1:
                    new_literals.add(lit)

            # Add from c2 except negated
            for lit in c2.literals:
                if lit != negated:
                    new_literals.add(lit)

            resolvent = Clause(tuple(sorted(new_literals, key=str)))

            # Check for tautology (contains P and -P)
            if resolvent.contains_complementary_pair():
                return None  # Tautology, discard

            return (resolvent, lit1)

    return None  # No complementary pair


# ============================================================================
# CNF conversion strategy interface
# ============================================================================

class CNFConverter(ABC):
    """
    Abstract strategy for CNF conversion.

    Defines interface without implementation.
    New converters extend without modifying resolver.
    """

    @abstractmethod
    def convert(self, sentence: str) -> List[Clause]:
        """
        Convert sentence to CNF clauses.

        Args:
            sentence: Propositional sentence

        Returns:
            List of immutable Clause objects
        """
        pass


class SimpleCNFConverter(CNFConverter):
    """
    Simple CNF converter for common patterns.

    Implements CNFConverter interface.

    Handles:
    - Implications: "P => Q" becomes ["-P | Q"]
    - Conjunctions: "(A | B) & (C | D)"
    - Single literals: "P"
    """

    def convert(self, sentence: str) -> List[Clause]:
        """Convert sentence to CNF using simple pattern matching."""
        sentence = sentence.strip()

        # Handle implication: P => Q becomes -P | Q
        if "=>" in sentence:
            parts = sentence.split("=>")
            if len(parts) == 2:
                ant = parse_literal(parts[0].strip())
                cons = parse_literal(parts[1].strip())
                return [Clause((ant.negate(), cons))]

        # Handle conjunction: (A | B) & (C | D)
        if "&" in sentence:
            clause_strs = sentence.split("&")
            clauses = []
            for clause_str in clause_strs:
                clause_str = clause_str.strip().strip("()")
                clauses.append(self._parse_clause(clause_str))
            return clauses

        # Single clause (possibly disjunction)
        return [self._parse_clause(sentence)]

    def _parse_clause(self, clause_str: str) -> Clause:
        """Parse a single clause string to Clause object."""
        clause_str = clause_str.strip().strip("()")

        if "|" in clause_str:
            # Disjunction: A | B | C
            literals = tuple(parse_literal(lit) for lit in clause_str.split("|"))
        else:
            # Single literal
            literals = (parse_literal(clause_str),)

        return Clause(literals)


# ============================================================================
# Resolution engine (no I/O)
# ============================================================================

@dataclass
class ResolutionState:
    """
    Mutable state for resolution algorithm.

    Only tracks algorithm state.
    """
    clauses: List[Clause]
    steps: List[ResolutionStep]
    iteration: int = 0


class ResolutionEngine:
    """
    Resolution-based theorem prover.

    Only resolution logic, no parsing or printing.
    Uses CNFConverter strategy.
    Operates on immutable clauses.

    Algorithm (Proof by Refutation):
    1. Convert KB to CNF
    2. Convert -query to CNF
    3. Apply resolution until empty clause or saturation

    Based on Russell & Norvig Section 7.5.2, Figure 7.12
    """

    def __init__(self, converter: CNFConverter = None, max_iterations: int = 100):
        """
        Initialize resolution engine.

        Args:
            converter: CNF conversion strategy
            max_iterations: Maximum resolution iterations
        """
        self._converter = converter or SimpleCNFConverter()
        self._max_iterations = max_iterations

    def prove_entailment(
        self,
        kb: List[str],
        query: str
    ) -> Tuple[bool, List[ResolutionStep], float]:
        """
        Prove KB |= query using resolution.

        Returns immutable results.

        Args:
            kb: Knowledge base sentences
            query: Query sentence

        Returns:
            Tuple of (entailed, steps, elapsed_time)

        Time Complexity: Exponential worst case O(2^n)
        """
        t0 = time.perf_counter()

        # Initialize state
        state = ResolutionState(clauses=[], steps=[])

        # Step 1: Convert KB to CNF
        for sentence in kb:
            clauses = self._converter.convert(sentence)
            state.clauses.extend(clauses)

        # Step 2: Negate query and convert to CNF
        query_lit = parse_literal(query)
        negated_query = Clause((query_lit.negate(),))
        state.clauses.append(negated_query)

        # Step 3: Apply resolution
        entailed = self._resolution_loop(state)

        elapsed = time.perf_counter() - t0
        return entailed, state.steps, elapsed

    def _resolution_loop(self, state: ResolutionState) -> bool:
        """
        Main resolution loop.

        Returns:
            True if empty clause derived (entailed), False otherwise
        """
        while state.iteration < self._max_iterations:
            state.iteration += 1
            new_clauses = []

            # Try resolving every pair
            for i in range(len(state.clauses)):
                for j in range(i + 1, len(state.clauses)):
                    result = resolve_clauses(state.clauses[i], state.clauses[j])

                    if result:
                        resolvent, resolved_on = result

                        # Check for empty clause (contradiction)
                        if resolvent.is_empty():
                            step = ResolutionStep(
                                clause1=state.clauses[i],
                                clause2=state.clauses[j],
                                resolvent=resolvent,
                                resolved_on=resolved_on
                            )
                            state.steps.append(step)
                            return True  # Entailed!

                        # Check if resolvent is new
                        if resolvent not in state.clauses and resolvent not in new_clauses:
                            new_clauses.append(resolvent)
                            step = ResolutionStep(
                                clause1=state.clauses[i],
                                clause2=state.clauses[j],
                                resolvent=resolvent,
                                resolved_on=resolved_on
                            )
                            state.steps.append(step)

            # If no new clauses, saturation reached
            if not new_clauses:
                return False  # Not entailed

            # Add new clauses
            state.clauses.extend(new_clauses)

        # Timeout
        return False


# ============================================================================
# Separate presentation
# ============================================================================

class ResolutionPrinter:
    """
    Formats and prints resolution traces.

    Only handles output formatting.
    """

    @staticmethod
    def print_trace(
        kb: List[str],
        query: str,
        entailed: bool,
        steps: List[ResolutionStep],
        initial_clauses: List[Clause],
        elapsed: float
    ) -> None:
        """Print complete resolution trace."""
        print("=" * 80)
        print("Resolution-Based Inference")
        print("=" * 80)
        print()
        print("Knowledge Base:")
        for i, sentence in enumerate(kb, 1):
            print(f"  {i}. {sentence}")
        print()
        print(f"Query: {query}")
        print()

        print("Initial CNF clauses:")
        for i, clause in enumerate(initial_clauses, 1):
            print(f"  {i}. {clause}")
        print()

        if steps:
            print("Resolution steps:")
            print()
            for i, step in enumerate(steps, 1):
                print(f"  Step {i}: {step}")

                if step.resolvent.is_empty():
                    print("            => [] (EMPTY CLAUSE - CONTRADICTION)")
                    break
            print()

        result_str = "ENTAILED" if entailed else "NOT ENTAILED"
        print(f"Result: {result_str}")
        print(f"Elapsed time: {elapsed:.6f}s")
        print()


# ============================================================================
# Simplified interface (backward compatible)
# ============================================================================

def resolution_entailment(
    kb: List[str],
    query: str,
    verbose: bool = True
) -> Tuple[bool, List[str], float]:
    """
    Resolution inference with optional output.

    Facade pattern - Simple interface to complex subsystem.

    Args:
        kb: Knowledge base sentences
        query: Query sentence
        verbose: Whether to print trace

    Returns:
        Tuple of (entailed, trace_strings, elapsed_time)
    """
    # Create engine and converter
    converter = SimpleCNFConverter()
    engine = ResolutionEngine(converter=converter)

    # Get initial clauses for printing
    initial_clauses = []
    for sentence in kb:
        initial_clauses.extend(converter.convert(sentence))
    query_lit = parse_literal(query)
    initial_clauses.append(Clause((query_lit.negate(),)))

    # Run resolution
    entailed, steps, elapsed = engine.prove_entailment(kb, query)

    # Print if verbose
    if verbose:
        ResolutionPrinter.print_trace(kb, query, entailed, steps, initial_clauses, elapsed)

    # Convert to string trace for backward compatibility
    trace_strings = [str(step) for step in steps]
    return entailed, trace_strings, elapsed


# ============================================================================
# Test functions (backward compatible)
# ============================================================================

def test_resolution_entailed() -> None:
    """
    Test resolution on entailed query.

    KB: P => Q, Q => R, P
    Query: R
    Expected: True
    """
    print("=" * 80)
    print("Part D: Resolution Test 1 - Entailed Query")
    print("=" * 80)
    print()

    kb = ["P => Q", "Q => R", "P"]
    query = "R"

    entailed, _, _ = resolution_entailment(kb, query, verbose=True)

    assert entailed is True, "Expected R to be entailed"
    print("TEST PASSED: Resolution correctly proved entailment")
    print()


def test_resolution_not_entailed() -> None:
    """
    Test resolution on non-entailed query.

    KB: P => Q, Q
    Query: P
    Expected: False
    """
    print("=" * 80)
    print("Part D: Resolution Test 2 - Non-Entailed Query")
    print("=" * 80)
    print()

    kb = ["P => Q", "Q"]
    query = "P"

    entailed, _, _ = resolution_entailment(kb, query, verbose=True)

    assert entailed is False, "Expected P to NOT be entailed"
    print("TEST PASSED: Resolution correctly determined non-entailment")
    print()


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    """Demonstrate refactored resolution."""
    print("=" * 80)
    print("Refactored Resolution Inference")
    print("=" * 80)
    print()

    # Test 1: Entailed
    test_resolution_entailed()
    print()

    # Test 2: Not entailed
    test_resolution_not_entailed()

    print("=" * 80)
    print("Part D: Resolution-Based Inference - COMPLETE")
    print("=" * 80)
    print()
    print("Design principles applied:")
    print("  - Single Responsibility: Parser, Engine, Printer separated")
    print("  - Open/Closed: Extensible through CNFConverter strategy")
    print("  - Functional: Immutable Literal, Clause, ResolutionStep")
    print("  - Abstraction: Clear interfaces, hidden implementation details")
