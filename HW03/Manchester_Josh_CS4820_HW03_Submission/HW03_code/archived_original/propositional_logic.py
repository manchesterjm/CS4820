"""
Propositional logic inference and model checking.

This module implements knowledge-based agents and propositional reasoning
including logical equivalences and model checking entailment.

KNOWLEDGE-BASED AGENT OVERVIEW:
===============================
A knowledge-based (KB) agent maintains an internal representation of the world
and uses logical inference to decide what actions to take (Russell & Norvig
Section 7.1, Lecture 8 Part I Slides 1-15).

Components of a KB agent:
1. Knowledge Base (KB): Collection of sentences representing facts and rules
2. Tell operation: Add new sentences to the KB
3. Ask operation: Query whether a sentence is entailed by the KB
4. Inference engine: Derives new sentences from existing knowledge

Agent loop:
    - Perceive environment
    - Tell KB about percepts
    - Ask KB what action to take
    - Perform action
    - Repeat

Why use KB agents?
- Declarative: Separate knowledge from reasoning algorithm
- Flexible: Add new knowledge without changing inference code
- Explainable: Can trace reasoning steps
- Compositional: Combine multiple sources of knowledge

Based on Russell & Norvig:
- Section 7.1: Knowledge-Based Agents
- Section 7.2: The Wumpus World
- Section 7.4: Propositional Logic
- Section 7.5: Propositional Theorem Proving

Lecture References:
- Lecture 8 Part I: Introduction to Logical Agents
- Lecture 8 Part II: Propositional Logic Syntax and Semantics

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Dict
from itertools import product


def check_equivalence_demorgan(show_table: bool = True) -> bool:
    """
    Check if De Morgan's Law holds: ¬(P ∨ Q) ≡ (¬P) ∧ (¬Q).

    Based on Russell & Norvig Section 7.4.2 (Logical Equivalence).
    De Morgan's Laws are fundamental equivalences in propositional logic:
    - ¬(A ∨ B) ≡ (¬A) ∧ (¬B)
    - ¬(A ∧ B) ≡ (¬A) ∨ (¬B)

    This function verifies the first law by enumerating all possible
    truth assignments to P and Q (truth table method).

    Args:
        show_table: If True, print the truth table

    Returns:
        True if the equivalence holds for all models, False otherwise

    Time Complexity: O(2^n) where n = 2 symbols
    Space Complexity: O(1)
    """
    if show_table:
        print("=== Logical Equivalence: De Morgan's Law ===")
        print()
        print("Formula: NOT(P OR Q) == (NOT P) AND (NOT Q)?")
        print()
        print("Truth Table:")
        print("P | Q | P OR Q | NOT(P OR Q) | NOT P | NOT Q | " +
              "(NOT P) AND (NOT Q) | Equivalent?")
        print("-" * 80)

    # Track if equivalence holds for all rows
    all_equivalent = True

    # Enumerate all truth assignments: (P, Q) in {T, F} x {T, F}
    for p_val, q_val in product([True, False], repeat=2):
        # Evaluate left side: ¬(P ∨ Q)
        p_or_q = p_val or q_val
        left_side = not p_or_q

        # Evaluate right side: (¬P) ∧ (¬Q)
        not_p = not p_val
        not_q = not q_val
        right_side = not_p and not_q

        # Check equivalence for this row
        equivalent = (left_side == right_side)
        all_equivalent = all_equivalent and equivalent

        if show_table:
            # Format booleans as single characters for table
            p_str = "T" if p_val else "F"
            q_str = "T" if q_val else "F"
            or_str = "T" if p_or_q else "F"
            left_str = "T" if left_side else "F"
            np_str = "T" if not_p else "F"
            nq_str = "T" if not_q else "F"
            right_str = "T" if right_side else "F"
            equiv_str = "YES" if equivalent else "NO"

            print(f"{p_str} | {q_str} |   {or_str}   |      " +
                  f"{left_str}      |   {np_str}  |   {nq_str}  |  " +
                  f"       {right_str}        |    {equiv_str}")

    if show_table:
        print()
        if all_equivalent:
            print("Result: EQUIVALENT (De Morgan's Law confirmed)")
        else:
            print("Result: NOT EQUIVALENT (De Morgan's Law violated!)")
        print()

    return all_equivalent


def check_equivalence_contraposition(show_table: bool = True) -> bool:
    """
    Check if Contraposition holds: (P ⇒ Q) ≡ (¬Q ⇒ ¬P).

    Based on Russell & Norvig Section 7.4.2 (Logical Equivalence).
    Contraposition is a key equivalence used in proof by contrapositive:
    - To prove P ⇒ Q, we can instead prove ¬Q ⇒ ¬P

    Implication semantics:
    - P ⇒ Q is false only when P is true and Q is false
    - Equivalently: P ⇒ Q ≡ ¬P ∨ Q

    Args:
        show_table: If True, print the truth table

    Returns:
        True if the equivalence holds for all models, False otherwise

    Time Complexity: O(2^n) where n = 2 symbols
    Space Complexity: O(1)
    """
    if show_table:
        print("=== Logical Equivalence: Contraposition ===")
        print()
        print("Formula: (P => Q) == (NOT Q => NOT P)?")
        print()
        print("Truth Table:")
        print("P | Q | P => Q | NOT Q | NOT P | " +
              "(NOT Q) => (NOT P) | Equivalent?")
        print("-" * 75)

    all_equivalent = True

    # Enumerate all truth assignments
    for p_val, q_val in product([True, False], repeat=2):
        # Evaluate left side: P ⇒ Q
        # Implication is false only when antecedent is true and consequent is false
        left_side = (not p_val) or q_val

        # Evaluate right side: ¬Q ⇒ ¬P
        not_q = not q_val
        not_p = not p_val
        right_side = (not not_q) or not_p  # Same as: Q or (not P)

        # Check equivalence
        equivalent = (left_side == right_side)
        all_equivalent = all_equivalent and equivalent

        if show_table:
            p_str = "T" if p_val else "F"
            q_str = "T" if q_val else "F"
            left_str = "T" if left_side else "F"
            nq_str = "T" if not_q else "F"
            np_str = "T" if not_p else "F"
            right_str = "T" if right_side else "F"
            equiv_str = "YES" if equivalent else "NO"

            print(f"{p_str} | {q_str} |   {left_str}    |   {nq_str}   | " +
                  f"  {np_str}   |         {right_str}          |    {equiv_str}")

    if show_table:
        print()
        if all_equivalent:
            print("Result: EQUIVALENT (Contraposition confirmed)")
        else:
            print("Result: NOT EQUIVALENT (Contraposition violated!)")
        print()

    return all_equivalent


def evaluate_sentence(sentence: str, model: Dict[str, bool]) -> bool:
    """
    Evaluate a propositional sentence given a truth assignment.

    Supports operators:
    - NOT, not, ! for negation
    - AND, and, & for conjunction
    - OR, or, | for disjunction
    - =>, IMPLIES for implication

    Args:
        sentence: Propositional sentence (e.g., "P => Q")
        model: Dictionary mapping symbols to truth values

    Returns:
        Truth value of sentence in the given model

    Example:
        evaluate_sentence("P => Q", {"P": True, "Q": False})  # Returns False
        evaluate_sentence("P & Q", {"P": True, "Q": True})    # Returns True
    """
    import re

    expr = sentence

    # First, normalize operators to temporary placeholders (to avoid symbol conflicts)
    expr = expr.replace("=>", " __IMPLIES__ ")
    expr = expr.replace("NOT", " __NOT__ ")
    expr = expr.replace("AND", " __AND__ ")
    expr = expr.replace("OR", " __OR__ ")
    expr = expr.replace("!", " __NOT__ ")
    expr = expr.replace("&", " __AND__ ")
    expr = expr.replace("|", " __OR__ ")

    # Replace each symbol with its boolean value using word boundaries
    for symbol, value in sorted(model.items(), key=lambda x: len(x[0]), reverse=True):
        # Use regex with word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(symbol) + r'\b'
        expr = re.sub(pattern, str(value), expr)

    # Convert implication: A __IMPLIES__ B becomes ((not A) or B)
    while "__IMPLIES__" in expr:
        pattern = r'(\w+)\s+__IMPLIES__\s+(\w+)'
        match = re.search(pattern, expr)
        if match:
            antecedent = match.group(1)
            consequent = match.group(2)
            replacement = f"((not {antecedent}) or {consequent})"
            expr = expr[:match.start()] + replacement + expr[match.end():]
        else:
            break

    # Replace operators with Python syntax
    expr = expr.replace("__NOT__", "not")
    expr = expr.replace("__AND__", "and")
    expr = expr.replace("__OR__", "or")

    # Evaluate the expression
    try:
        result = eval(expr)
        return bool(result)
    except Exception as e:
        raise ValueError(f"Error evaluating sentence '{sentence}': {e}\n" +
                         f"Transformed expression: {expr}")


def model_check(kb: List[str], query: str, symbols: List[str],
                verbose: bool = True) -> bool:
    """
    Model checking algorithm for propositional entailment.

    Based on Russell & Norvig Figure 7.10, Lecture 8 Part I Slide 45.

    Determines if KB |= query (KB entails query) by enumerating all possible
    truth assignments to symbols and checking if query is true in all models
    where KB is true.

    Algorithm:
        For each possible truth assignment (model):
            Evaluate KB in this model
            If KB is true:
                Check if query is also true
                If query is false, KB does not entail query
        If all KB-satisfying models also satisfy query, then KB |= query

    Args:
        kb: List of propositional sentences (knowledge base)
        query: Query sentence to check for entailment
        symbols: List of propositional symbols appearing in KB and query
        verbose: If True, print reasoning trace

    Returns:
        True if KB entails query, False otherwise

    Time Complexity: O(2^n * m) where n = number of symbols, m = KB size
    Space Complexity: O(n) for recursion/enumeration

    Example:
        kb = ["P => Q", "Q => R", "P"]
        symbols = ["P", "Q", "R"]
        query = "R"
        model_check(kb, query, symbols)  # Returns True
    """
    if verbose:
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

    # Track models where KB is true
    kb_models = []
    # Track models where KB is true but query is false
    counterexamples = []

    # Enumerate all possible truth assignments (2^n models)
    num_models = 2 ** len(symbols)
    for i, truth_values in enumerate(product([True, False], repeat=len(symbols))):
        # Create model: mapping from symbols to truth values
        model = dict(zip(symbols, truth_values))

        # Evaluate KB in this model
        kb_true = all(evaluate_sentence(s, model) for s in kb)

        if kb_true:
            kb_models.append(model.copy())
            # Check if query is true in this model
            query_true = evaluate_sentence(query, model)

            if verbose:
                model_str = ", ".join(f"{s}={model[s]}" for s in symbols)
                status = "KB=True, Query=True" if query_true else \
                         "KB=True, Query=FALSE (counterexample!)"
                print(f"  Model {i+1}: {{{model_str}}}  =>  {status}")

            if not query_true:
                counterexamples.append(model.copy())

    if verbose:
        print()
        print(f"Total models: {num_models}")
        print(f"Models where KB is true: {len(kb_models)}")
        print(f"Counterexamples (KB true, query false): {len(counterexamples)}")
        print()

    # KB entails query if there are no counterexamples
    entailed = len(counterexamples) == 0

    if verbose:
        if entailed:
            print("Result: KB |= query (ENTAILED)")
            print("Reasoning: In all models where KB is true, query is also true")
        else:
            print("Result: KB does NOT entail query (NOT ENTAILED)")
            print("Reasoning: Found at least one model where KB is true " +
                  "but query is false")
        print()

    return entailed


def test_model_checking_example() -> None:
    """
    Test model checking on the example from assignment.

    KB: P => Q, Q => R, P
    Query: R
    Expected: True (entailed)
    """
    print("=" * 80)
    print("Part A: Model Checking Example")
    print("=" * 80)
    print()

    kb = ["P => Q", "Q => R", "P"]
    query = "R"
    symbols = ["P", "Q", "R"]

    result = model_check(kb, query, symbols, verbose=True)

    # Verify result
    assert result is True, "Expected KB to entail R"
    print("TEST PASSED: Model checking correctly determined entailment")
    print()


def test_logical_equivalences() -> None:
    """
    Test logical equivalence checking for De Morgan and Contraposition.
    """
    print("=" * 80)
    print("Part A: Logical Equivalences")
    print("=" * 80)
    print()

    # Test De Morgan's Law
    demorgan_result = check_equivalence_demorgan(show_table=True)
    assert demorgan_result is True, "De Morgan's Law should hold"

    # Test Contraposition
    contraposition_result = check_equivalence_contraposition(show_table=True)
    assert contraposition_result is True, "Contraposition should hold"

    print("=" * 80)
    print("All equivalence tests PASSED")
    print("=" * 80)
    print()


if __name__ == "__main__":
    """
    Run all Part A demonstrations.
    """
    # Part A.1: KB Agent Overview
    print(__doc__)
    print()

    # Part A.2: Logical Equivalences
    test_logical_equivalences()

    # Part A.3: Model Checking
    test_model_checking_example()

    print("=" * 80)
    print("Part A: Propositional Logic - COMPLETE")
    print("=" * 80)
