"""
Refactored propositional logic using SOFA principles.

SOFA Improvements:
- Single Responsibility: Separated logic evaluation, printing, and orchestration
- Open/Closed: Extensible through composition
- Functional: Pure functions, immutable data structures
- Abstraction: Clear interfaces for truth tables and model checking

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Dict, Tuple, Callable
from itertools import product
from dataclasses import dataclass
from inference_engine_base import InferenceEngine, InferenceResult
import time


# ============================================================================
# SOFA: Functional - Pure data structures (immutable)
# ============================================================================

@dataclass(frozen=True)
class TruthTableRow:
    """Immutable truth table row."""
    variables: Tuple[Tuple[str, bool], ...]  # (name, value) pairs
    left_value: bool
    right_value: bool
    equivalent: bool

    def to_dict(self) -> Dict[str, bool]:
        """Convert to variable assignment dict."""
        return dict(self.variables)


@dataclass(frozen=True)
class TruthTable:
    """Immutable truth table result."""
    formula_left: str
    formula_right: str
    rows: Tuple[TruthTableRow, ...]
    all_equivalent: bool


@dataclass(frozen=True)
class Model:
    """Immutable truth assignment to propositional symbols."""
    assignment: Tuple[Tuple[str, bool], ...]  # Frozen tuple of pairs

    def __getitem__(self, symbol: str) -> bool:
        """Get truth value for symbol."""
        for sym, val in self.assignment:
            if sym == symbol:
                return val
        raise KeyError(f"Symbol {symbol} not in model")

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary for easier access."""
        return dict(self.assignment)

    def satisfies_kb(self, kb: List[str], evaluator: Callable) -> bool:
        """Check if this model satisfies all KB sentences."""
        return all(evaluator(sentence, self.to_dict()) for sentence in kb)


# ============================================================================
# SOFA: Functional - Pure functions (no side effects)
# ============================================================================

def evaluate_propositional_formula(formula: str, model: Dict[str, bool]) -> bool:
    """
    Pure function: Evaluate propositional formula in a model.

    SOFA: Functional - No side effects, deterministic output

    Args:
        formula: Propositional formula string
        model: Truth assignment

    Returns:
        Boolean truth value
    """
    import re

    expr = formula

    # Normalize operators to placeholders
    expr = expr.replace("=>", " __IMPLIES__ ")
    expr = expr.replace("NOT", " __NOT__ ")
    expr = expr.replace("AND", " __AND__ ")
    expr = expr.replace("OR", " __OR__ ")
    expr = expr.replace("!", " __NOT__ ")
    expr = expr.replace("&", " __AND__ ")
    expr = expr.replace("|", " __OR__ ")

    # Replace symbols with values
    for symbol, value in sorted(model.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(symbol) + r'\b'
        expr = re.sub(pattern, str(value), expr)

    # Handle implication (must be done before replacing __IMPLIES__)
    # Pattern matches: (expression) __IMPLIES__ (expression) or word __IMPLIES__ word
    while "__IMPLIES__" in expr:
        # Try to match parenthesized expressions first: (...) __IMPLIES__ (...)
        pattern = r'\(([^()]+)\)\s+__IMPLIES__\s+\(([^()]+)\)'
        match = re.search(pattern, expr)
        if match:
            ant = match.group(1).strip()
            cons = match.group(2).strip()
            replacement = f"(({ant}) or ({cons}))"  # Implication: A => B == -A or B
            # But we need to negate the antecedent
            replacement = f"((not ({ant})) or ({cons}))"
            expr = expr[:match.start()] + replacement + expr[match.end():]
            continue

        # Try simple word __IMPLIES__ word pattern
        pattern = r'(\w+)\s+__IMPLIES__\s+(\w+)'
        match = re.search(pattern, expr)
        if match:
            ant = match.group(1)
            cons = match.group(2)
            replacement = f"((not {ant}) or {cons})"
            expr = expr[:match.start()] + replacement + expr[match.end():]
        else:
            # If no pattern matches but __IMPLIES__ still present, break to avoid infinite loop
            break

    # Replace operators
    expr = expr.replace("__NOT__", "not")
    expr = expr.replace("__AND__", "and")
    expr = expr.replace("__OR__", "or")

    return bool(eval(expr))


def generate_all_models(symbols: List[str]) -> Tuple[Model, ...]:
    """
    Pure function: Generate all possible truth assignments.

    SOFA: Functional - Returns immutable tuple of models

    Args:
        symbols: List of propositional symbols

    Returns:
        Tuple of all possible models
    """
    models = []
    for values in product([True, False], repeat=len(symbols)):
        assignment = tuple(zip(symbols, values))
        models.append(Model(assignment))
    return tuple(models)


def check_equivalence(
    left_formula: str,
    right_formula: str,
    symbols: List[str]
) -> TruthTable:
    """
    Pure function: Check logical equivalence using truth tables.

    SOFA:
    - Functional: Pure function, returns immutable result
    - Single Responsibility: Only computes equivalence

    Args:
        left_formula: Left side of equivalence
        right_formula: Right side of equivalence
        symbols: Propositional symbols

    Returns:
        TruthTable with all rows and equivalence result
    """
    rows = []
    all_equivalent = True

    for model in generate_all_models(symbols):
        model_dict = model.to_dict()
        left_val = evaluate_propositional_formula(left_formula, model_dict)
        right_val = evaluate_propositional_formula(right_formula, model_dict)
        equiv = (left_val == right_val)
        all_equivalent = all_equivalent and equiv

        rows.append(TruthTableRow(
            variables=model.assignment,
            left_value=left_val,
            right_value=right_val,
            equivalent=equiv
        ))

    return TruthTable(
        formula_left=left_formula,
        formula_right=right_formula,
        rows=tuple(rows),
        all_equivalent=all_equivalent
    )


def model_check_pure(
    kb: List[str],
    query: str,
    symbols: List[str]
) -> Tuple[bool, List[Model], List[Model]]:
    """
    Pure function: Model checking without side effects.

    SOFA:
    - Functional: Pure, no printing
    - Single Responsibility: Only checks entailment

    Args:
        kb: Knowledge base sentences
        query: Query sentence
        symbols: Propositional symbols

    Returns:
        Tuple of (entailed, satisfying_models, counterexamples)
    """
    satisfying_models = []
    counterexamples = []

    for model in generate_all_models(symbols):
        if model.satisfies_kb(kb, evaluate_propositional_formula):
            satisfying_models.append(model)
            if not evaluate_propositional_formula(query, model.to_dict()):
                counterexamples.append(model)

    entailed = len(counterexamples) == 0
    return entailed, satisfying_models, counterexamples


# ============================================================================
# SOFA: Single Responsibility - Separate presentation from logic
# ============================================================================

class TruthTablePrinter:
    """
    Handles truth table formatting.

    SOFA: Single Responsibility - Only formats and prints
    """

    @staticmethod
    def print_demorgan_table(table: TruthTable) -> None:
        """Print De Morgan's Law truth table."""
        print("=== Logical Equivalence: De Morgan's Law ===")
        print()
        print("Formula: NOT(P OR Q) == (NOT P) AND (NOT Q)?")
        print()
        print("Truth Table:")
        print("P | Q | P OR Q | NOT(P OR Q) | NOT P | NOT Q | " +
              "(NOT P) AND (NOT Q) | Equivalent?")
        print("-" * 80)

        for row in table.rows:
            model_dict = row.to_dict()
            p_val = model_dict['P']
            q_val = model_dict['Q']
            p_or_q = p_val or q_val

            p_str = "T" if p_val else "F"
            q_str = "T" if q_val else "F"
            or_str = "T" if p_or_q else "F"
            left_str = "T" if row.left_value else "F"
            np_str = "T" if not p_val else "F"
            nq_str = "T" if not q_val else "F"
            right_str = "T" if row.right_value else "F"
            equiv_str = "YES" if row.equivalent else "NO"

            print(f"{p_str} | {q_str} |   {or_str}   |      {left_str}      | " +
                  f"  {np_str}  |   {nq_str}  |         {right_str}        |    {equiv_str}")

        print()
        result_str = "EQUIVALENT" if table.all_equivalent else "NOT EQUIVALENT"
        print(f"Result: {result_str} (De Morgan's Law confirmed)")
        print()

    @staticmethod
    def print_contraposition_table(table: TruthTable) -> None:
        """Print Contraposition truth table."""
        print("=== Logical Equivalence: Contraposition ===")
        print()
        print("Formula: (P => Q) == (NOT Q => NOT P)?")
        print()
        print("Truth Table:")
        print("P | Q | P => Q | NOT Q | NOT P | " +
              "(NOT Q) => (NOT P) | Equivalent?")
        print("-" * 75)

        for row in table.rows:
            model_dict = row.to_dict()
            p_val = model_dict['P']
            q_val = model_dict['Q']

            p_str = "T" if p_val else "F"
            q_str = "T" if q_val else "F"
            left_str = "T" if row.left_value else "F"
            nq_str = "T" if not q_val else "F"
            np_str = "T" if not p_val else "F"
            right_str = "T" if row.right_value else "F"
            equiv_str = "YES" if row.equivalent else "NO"

            print(f"{p_str} | {q_str} |   {left_str}    |   {nq_str}   | " +
                  f"  {np_str}   |         {right_str}          |    {equiv_str}")

        print()
        result_str = "EQUIVALENT" if table.all_equivalent else "NOT EQUIVALENT"
        print(f"Result: {result_str} (Contraposition confirmed)")
        print()


class ModelCheckPrinter:
    """
    Handles model checking output formatting.

    SOFA: Single Responsibility - Only formats and prints
    """

    @staticmethod
    def print_results(
        kb: List[str],
        query: str,
        symbols: List[str],
        entailed: bool,
        satisfying_models: List[Model],
        counterexamples: List[Model]
    ) -> None:
        """Print model checking results."""
        print("=== Model Checking ===")
        print()
        print("Knowledge Base:")
        for i, sentence in enumerate(kb, 1):
            print(f"  {i}. {sentence}")
        print()
        print(f"Query: {query}")
        print(f"Symbols: {{{', '.join(symbols)}}}")
        print()
        print("Enumerating all possible models...")
        print()

        total_models = 2 ** len(symbols)

        for i, model in enumerate(satisfying_models, 1):
            model_str = ", ".join(f"{s}={model[s]}" for s in symbols)
            query_val = evaluate_propositional_formula(query, model.to_dict())
            status = "KB=True, Query=True" if query_val else \
                     "KB=True, Query=FALSE (counterexample!)"
            print(f"  Model {i}: {{{model_str}}}  =>  {status}")

        print()
        print(f"Total models: {total_models}")
        print(f"Models where KB is true: {len(satisfying_models)}")
        print(f"Counterexamples (KB true, query false): {len(counterexamples)}")
        print()

        if entailed:
            print("Result: KB |= query (ENTAILED)")
            print("Reasoning: In all models where KB is true, query is also true")
        else:
            print("Result: KB does NOT entail query (NOT ENTAILED)")
            print("Reasoning: Found at least one model where KB is true " +
                  "but query is false")
        print()


# ============================================================================
# SOFA: Open/Closed - Extensible inference engine
# ============================================================================

class ModelCheckingEngine(InferenceEngine):
    """
    Model checking inference engine.

    SOFA:
    - Open/Closed: Extends InferenceEngine without modifying it
    - Single Responsibility: Only model checking
    - Abstraction: Implements abstract interface
    """

    def __init__(self, kb: List[str], symbols: List[str]):
        """
        Initialize model checking engine.

        Args:
            kb: Knowledge base sentences
            symbols: Propositional symbols
        """
        self._kb = kb
        self._symbols = symbols

    def infer(self, query: str) -> InferenceResult:
        """
        Check entailment using model checking.

        Args:
            query: Query to check

        Returns:
            InferenceResult with entailment status
        """
        t0 = time.perf_counter()
        entailed, satisfying, counterex = model_check_pure(
            self._kb, query, self._symbols
        )
        elapsed = time.perf_counter() - t0

        trace = [
            f"Total models: {2 ** len(self._symbols)}",
            f"KB-satisfying models: {len(satisfying)}",
            f"Counterexamples: {len(counterex)}"
        ]

        return InferenceResult(entailed, trace, elapsed)

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "Model Checking (TT-Entails)"


# ============================================================================
# SOFA: Facade pattern - Simplified interface
# ============================================================================

def check_demorgan_equivalence(show_table: bool = True) -> bool:
    """
    Check De Morgan's Law with optional table display.

    SOFA: Facade - Simplified interface combining multiple operations

    Args:
        show_table: Whether to print truth table

    Returns:
        True if equivalent, False otherwise
    """
    left = "NOT(P OR Q)"
    right = "(NOT P) AND (NOT Q)"
    table = check_equivalence(left, right, ['P', 'Q'])

    if show_table:
        TruthTablePrinter.print_demorgan_table(table)

    return table.all_equivalent


def check_contraposition_equivalence(show_table: bool = True) -> bool:
    """
    Check Contraposition with optional table display.

    SOFA: Facade - Simplified interface

    Args:
        show_table: Whether to print truth table

    Returns:
        True if equivalent, False otherwise
    """
    left = "P => Q"
    right = "(NOT Q) => (NOT P)"
    table = check_equivalence(left, right, ['P', 'Q'])

    if show_table:
        TruthTablePrinter.print_contraposition_table(table)

    return table.all_equivalent


def model_check_with_output(
    kb: List[str],
    query: str,
    symbols: List[str],
    verbose: bool = True
) -> bool:
    """
    Model checking with optional output.

    SOFA: Facade - Combines pure function with presentation

    Args:
        kb: Knowledge base
        query: Query sentence
        symbols: Propositional symbols
        verbose: Whether to print output

    Returns:
        True if entailed, False otherwise
    """
    entailed, satisfying, counterex = model_check_pure(kb, query, symbols)

    if verbose:
        ModelCheckPrinter.print_results(
            kb, query, symbols, entailed, satisfying, counterex
        )

    return entailed


# ============================================================================
# Main demonstration (backward compatible)
# ============================================================================

if __name__ == "__main__":
    """Demonstrate refactored propositional logic."""
    print("=" * 80)
    print("Refactored Propositional Logic (SOFA Principles)")
    print("=" * 80)
    print()

    # Test equivalences
    demorgan_ok = check_demorgan_equivalence(show_table=True)
    assert demorgan_ok, "De Morgan's Law failed"

    contraposition_ok = check_contraposition_equivalence(show_table=True)
    assert contraposition_ok, "Contraposition failed"

    # Test model checking
    kb = ["P => Q", "Q => R", "P"]
    result = model_check_with_output(kb, "R", ["P", "Q", "R"], verbose=True)
    assert result is True, "Model checking failed"

    # Demonstrate inference engine interface
    print("=" * 80)
    print("Using InferenceEngine Interface")
    print("=" * 80)
    print()

    engine = ModelCheckingEngine(kb, ["P", "Q", "R"])
    inference_result = engine.infer("R")
    print(f"Engine: {engine.get_algorithm_name()}")
    print(inference_result)
    print(f"Trace: {inference_result.trace}")

    print()
    print("=" * 80)
    print("All tests PASSED - SOFA refactoring successful!")
    print("=" * 80)
