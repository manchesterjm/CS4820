"""
Horn clause inference using forward chaining.

This module implements forward chaining inference for Horn clauses,
a restricted but efficient form of propositional logic.

HORN CLAUSES:
=============
A Horn clause is a disjunction of literals with at most one positive literal:
    (¬P1 ∨ ¬P2 ∨ ... ∨ ¬Pn ∨ Q)

Equivalently (and more intuitively):
    (P1 ∧ P2 ∧ ... ∧ Pn) ⇒ Q

Where P1, ..., Pn are premises and Q is the conclusion.

Special cases:
- Fact: Horn clause with no premises (just Q)
- Definite clause: Horn clause with exactly one positive literal

Why Horn clauses?
- Efficient inference: O(n) time where n = KB size
- Sound and complete for Horn KB entailment
- Used in logic programming (Prolog), expert systems, databases

Based on Russell & Norvig:
- Section 7.5.3: Forward and Backward Chaining
- Figure 7.15: Forward Chaining Algorithm
- Figure 7.16: Backward Chaining Algorithm

Lecture References:
- Lecture 8 Part III: Inference in First-Order Logic
- Slides 75-90: Horn Clause Inference

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

import time
from typing import List, Tuple, Dict
from collections import deque
from knowledge_base import HornKB


def forward_chaining(kb: HornKB, query: str, verbose: bool = True) -> \
        Tuple[bool, List[str], float]:
    """
    Forward chaining inference for Horn clauses.

    Based on Russell & Norvig Figure 7.15, Lecture 8 Part III Slide 80.

    Forward chaining is data-driven reasoning:
    - Start with known facts
    - Apply rules whose premises are satisfied
    - Derive new facts until query is proven or no more facts can be derived

    Algorithm:
        1. Initialize count[rule] = number of premises for each rule
        2. Initialize inferred = set() (facts derived so far)
        3. Initialize agenda = queue of known facts
        4. While agenda is not empty:
            a. Pop fact p from agenda
            b. If p == query, return True
            c. If p not already inferred:
                - Mark p as inferred
                - For each rule where p is a premise:
                    - Decrement count[rule]
                    - If count[rule] == 0, add conclusion to agenda
        5. Return False (query not entailed)

    Args:
        kb: Horn clause knowledge base
        query: Ground atom to prove (e.g., "D")
        verbose: If True, print detailed inference trace

    Returns:
        Tuple of (entailed, trace, elapsed_time):
        - entailed: True if query is entailed by KB
        - trace: List of inferred facts in derivation order
        - elapsed_time: Time taken in seconds

    Time Complexity: O(n) where n = KB size (linear time!)
    Space Complexity: O(n) for count dictionary and inferred set

    Example:
        KB facts: A, B
        KB rules: A ∧ B ⇒ C, C ⇒ D
        Query: D
        Result: True (derive C from A,B then D from C)
    """
    t0 = time.perf_counter()

    if verbose:
        print("=== Forward Chaining ===")
        print()
        print("Knowledge Base:")
        print(kb)
        print()
        print(f"Query: {query}")
        print()

    # Initialize data structures
    # count[rule_idx] = number of unsatisfied premises for rule
    count: Dict[int, int] = {}
    # inferred[fact] = True if fact has been derived
    inferred: Dict[str, bool] = {}
    # agenda = queue of facts to process
    agenda: deque = deque()
    # trace = list of derived facts (for output)
    trace: List[str] = []

    # Build count dictionary for rules
    rules = kb.get_rules()
    for i, (premises, conclusion) in enumerate(rules):
        count[i] = len(premises)

    # Initialize agenda with known facts
    for fact in kb.get_facts():
        agenda.append(fact)
        if verbose:
            print(f"Initial fact: {fact}")

    if verbose and agenda:
        print()

    iteration = 0
    # Main loop: process facts from agenda
    while agenda:
        iteration += 1
        # Pop a fact from the agenda
        p = agenda.popleft()

        if verbose:
            print(f"Iteration {iteration}: Processing fact '{p}'")

        # Check if this fact is the query
        if p == query:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"  -> Query '{query}' found!")
                print()
                print(f"Result: ENTAILED (derived in {iteration} iterations)")
                print(f"Elapsed time: {elapsed:.6f}s")
                print()
            return True, trace + [p], elapsed

        # Skip if already inferred
        if inferred.get(p, False):
            if verbose:
                print("  -> Already inferred, skipping")
            continue

        # Mark as inferred
        inferred[p] = True
        trace.append(p)

        if verbose:
            print(f"  -> Marked '{p}' as inferred")

        # Check all rules where p appears in premises
        for i, (premises, conclusion) in enumerate(rules):
            if p in premises:
                # Decrement count for this rule
                count[i] -= 1
                if verbose:
                    print(f"  -> Rule {i+1}: {' AND '.join(premises)} => " +
                          f"{conclusion}")
                    print(f"     Count decreased to {count[i]}")

                # If all premises satisfied, add conclusion to agenda
                if count[i] == 0:
                    if conclusion not in inferred:
                        agenda.append(conclusion)
                        if verbose:
                            print(f"     ALL premises satisfied! " +
                                  f"Adding '{conclusion}' to agenda")
                    else:
                        if verbose:
                            print(f"     Conclusion '{conclusion}' " +
                                  "already inferred")

        if verbose:
            print()

    # Query not entailed
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"Result: NOT ENTAILED (exhausted all rules in " +
              f"{iteration} iterations)")
        print(f"Elapsed time: {elapsed:.6f}s")
        print()

    return False, trace, elapsed


def test_generic_kb() -> None:
    """
    Test forward chaining on a generic Horn clause KB.

    KB:
        Facts: A, B
        Rules:
            1. A ∧ B ⇒ C
            2. C ⇒ D
            3. D ∧ E ⇒ F

    Test queries:
        - C: Should be entailed (derived from A, B)
        - D: Should be entailed (derived from C)
        - F: Should NOT be entailed (E not known)
    """
    print("=" * 80)
    print("Part B: Forward Chaining - Generic KB Test")
    print("=" * 80)
    print()

    # Create KB
    kb = HornKB()
    kb.tell_fact("A")
    kb.tell_fact("B")
    kb.tell_rule(["A", "B"], "C")
    kb.tell_rule(["C"], "D")
    kb.tell_rule(["D", "E"], "F")

    # Test 1: Query C (should be entailed)
    print("TEST 1: Query 'C'")
    print("-" * 40)
    entailed, trace, elapsed = forward_chaining(kb, "C", verbose=True)
    assert entailed is True, "Expected C to be entailed"
    print("PASSED: C correctly entailed")
    print()

    # Test 2: Query D (should be entailed)
    print("TEST 2: Query 'D'")
    print("-" * 40)
    entailed, trace, elapsed = forward_chaining(kb, "D", verbose=True)
    assert entailed is True, "Expected D to be entailed"
    print("PASSED: D correctly entailed")
    print()

    # Test 3: Query F (should NOT be entailed)
    print("TEST 3: Query 'F'")
    print("-" * 40)
    entailed, trace, elapsed = forward_chaining(kb, "F", verbose=True)
    assert entailed is False, "Expected F to NOT be entailed"
    print("PASSED: F correctly not entailed (E is unknown)")
    print()


def test_wumpus_fragment() -> None:
    """
    Test forward chaining on Wumpus World fragment.

    Based on Russell & Norvig Section 7.7 and assignment requirements.

    Wumpus World rules:
    - If no breeze at (x,y), then no pits in adjacent cells
    - B(x,y) denotes breeze at (x,y)
    - P(x,y) denotes pit at (x,y)
    - not_B(x,y) denotes no breeze at (x,y)
    - not_P(x,y) denotes no pit at (x,y)

    Scenario (4x4 grid):
        Agent starts at (1,1)
        Percepts:
            - No breeze at (1,1)  ->  not_B_1_1
            - Breeze at (2,1)     ->  B_2_1
            - No breeze at (1,2)  ->  not_B_1_2

        Rules:
            - not_B_1_1 => not_P_2_1 (no breeze at 1,1 means no pit at 2,1)
            - not_B_1_1 => not_P_1_2 (no breeze at 1,1 means no pit at 1,2)
            - not_B_1_2 => not_P_2_2 (no breeze at 1,2 means no pit at 2,2)
            - not_B_1_2 => not_P_1_1 (no breeze at 1,2 means no pit at 1,1)

        Queries:
            - not_P_1_2: Should be entailed (from not_B_1_1)
            - not_P_2_1: Should be entailed (from not_B_1_1)
            - not_P_2_2: Should be entailed (from not_B_1_2)
    """
    print("=" * 80)
    print("Part B: Forward Chaining - Wumpus Fragment Test")
    print("=" * 80)
    print()

    # Create Wumpus KB
    kb = HornKB()

    # Add percepts as facts
    kb.tell_fact("not_B_1_1")  # No breeze at (1,1)
    kb.tell_fact("B_2_1")      # Breeze at (2,1)
    kb.tell_fact("not_B_1_2")  # No breeze at (1,2)

    # Add Wumpus World rules
    # If no breeze at (1,1), then no pits in neighbors: (2,1) and (1,2)
    kb.tell_rule(["not_B_1_1"], "not_P_2_1")
    kb.tell_rule(["not_B_1_1"], "not_P_1_2")

    # If no breeze at (1,2), then no pits in neighbors: (1,1), (2,2), (1,3)
    kb.tell_rule(["not_B_1_2"], "not_P_1_1")
    kb.tell_rule(["not_B_1_2"], "not_P_2_2")
    kb.tell_rule(["not_B_1_2"], "not_P_1_3")

    # If no breeze at (2,1), then no pits in neighbors
    # (not applicable here since we have breeze at 2,1)

    # Test 1: Query not_P_1_2
    print("TEST 1: Query 'not_P_1_2' (no pit at 1,2)")
    print("-" * 40)
    entailed, _, _ = forward_chaining(kb, "not_P_1_2", verbose=True)
    assert entailed is True, "Expected not_P_1_2 to be entailed"
    print("PASSED: not_P_1_2 correctly entailed (safe to move to 1,2)")
    print()

    # Test 2: Query not_P_2_1
    print("TEST 2: Query 'not_P_2_1' (no pit at 2,1)")
    print("-" * 40)
    entailed, _, _ = forward_chaining(kb, "not_P_2_1", verbose=True)
    assert entailed is True, "Expected not_P_2_1 to be entailed"
    print("PASSED: not_P_2_1 correctly entailed (safe to move to 2,1)")
    print()

    # Test 3: Query not_P_2_2
    print("TEST 3: Query 'not_P_2_2' (no pit at 2,2)")
    print("-" * 40)
    entailed, _, _ = forward_chaining(kb, "not_P_2_2", verbose=True)
    assert entailed is True, "Expected not_P_2_2 to be entailed"
    print("PASSED: not_P_2_2 correctly entailed (from not_B_1_2)")
    print()


if __name__ == "__main__":
    """
    Run all Part B demonstrations.
    """
    # Test 1: Generic KB
    test_generic_kb()

    print()
    print()

    # Test 2: Wumpus fragment
    test_wumpus_fragment()

    print("=" * 80)
    print("Part B: Horn Clause Inference - COMPLETE")
    print("=" * 80)
