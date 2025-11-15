"""
Resolution-based inference for propositional logic.

This module implements the resolution algorithm for determining entailment
in propositional logic using Conjunctive Normal Form (CNF).

RESOLUTION OVERVIEW:
===================
Resolution is a complete inference rule for propositional logic:
- If clause C1 contains literal L and clause C2 contains ¬L,
- Then resolve(C1, C2) = (C1 - {L}) ∪ (C2 - {¬L})

Resolution is refutation-complete:
- To prove KB |= α, we prove KB ∧ ¬α is unsatisfiable
- Convert KB ∧ ¬α to CNF
- Apply resolution until either:
    a) Empty clause derived (contradiction) -> KB |= α
    b) No new clauses derived (no contradiction) -> KB does NOT entail α

Why resolution?
- Complete: If KB |= α, resolution will prove it
- Systematic: Can be automated
- Generalizes to first-order logic
- Foundation for logic programming and theorem proving

Based on Russell & Norvig:
- Section 7.5.2: Resolution
- Figure 7.12: Resolution Algorithm
- Figure 7.13: CNF Conversion

Lecture References:
- Lecture 8 Part III: Propositional Theorem Proving
- Slides 95-110: Resolution and CNF Conversion

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Tuple, Optional
import time


def normalize_literal(literal: str) -> str:
    """
    Normalize a literal to canonical form.

    Args:
        literal: Literal string (e.g., "P", "-P", "not P")

    Returns:
        Normalized literal ("-P" or "P")
    """
    lit = literal.strip()
    # Handle various negation formats
    if lit.startswith("not "):
        return "-" + lit[4:].strip()
    elif lit.startswith("!"):
        return "-" + lit[1:].strip()
    elif lit.startswith("-"):
        return lit  # Already normalized
    else:
        return lit  # Positive literal


def negate_literal(literal: str) -> str:
    """
    Negate a literal.

    Args:
        literal: Literal in normalized form ("-P" or "P")

    Returns:
        Negated literal
    """
    if literal.startswith("-"):
        return literal[1:]  # Remove negation
    else:
        return "-" + literal  # Add negation


def parse_simple_cnf(sentence: str) -> List[List[str]]:
    """
    Parse a simple CNF sentence into clauses.

    This is a simplified parser for common patterns like:
    - "P => Q" becomes [["-P", "Q"]]
    - "(A | B) & (C | D)" becomes [["A", "B"], ["C", "D"]]
    - "P" becomes [["P"]]

    Args:
        sentence: Propositional sentence

    Returns:
        List of clauses, where each clause is a list of literals
    """
    sentence = sentence.strip()

    # Handle implication: P => Q becomes -P | Q
    if "=>" in sentence:
        parts = sentence.split("=>")
        if len(parts) == 2:
            antecedent = parts[0].strip()
            consequent = parts[1].strip()
            # P => Q is equivalent to -P | Q
            return [["-" + antecedent, consequent]]

    # Handle conjunction of clauses: (A | B) & (C | D)
    if "&" in sentence:
        # Split by & to get individual clauses
        clause_strs = sentence.split("&")
        clauses = []
        for clause_str in clause_strs:
            clause_str = clause_str.strip()
            # Remove parentheses
            clause_str = clause_str.strip("()")
            # Split by | to get literals
            if "|" in clause_str:
                literals = [normalize_literal(lit) for lit in clause_str.split("|")]
            else:
                literals = [normalize_literal(clause_str)]
            clauses.append(literals)
        return clauses

    # Handle disjunction: A | B | C
    if "|" in sentence:
        literals = [normalize_literal(lit) for lit in sentence.split("|")]
        return [literals]

    # Single literal
    return [[normalize_literal(sentence)]]


def convert_to_cnf(sentence: str) -> List[List[str]]:
    """
    Convert propositional sentence to Conjunctive Normal Form (CNF).

    Based on Russell & Norvig Section 7.5.2, Figure 7.13.

    CNF conversion steps:
        1. Eliminate implications: (A => B) becomes (-A | B)
        2. Move negations inward using De Morgan's laws
        3. Distribute | over &
        4. Flatten to list of clauses

    This is a simplified implementation for common patterns.
    A full implementation would require parsing arbitrary formulas.

    Args:
        sentence: Propositional sentence (e.g., "P => Q", "(A | B) & C")

    Returns:
        List of clauses, where each clause is a list of literals

    Example:
        Input: "P => Q"
        Output: [["-P", "Q"]]
    """
    return parse_simple_cnf(sentence)


def resolve(clause1: List[str], clause2: List[str]) -> Optional[List[str]]:
    """
    Apply resolution rule to two clauses.

    Based on Russell & Norvig Figure 7.12.

    Resolution rule:
    - If clause1 contains literal L and clause2 contains -L (or vice versa),
    - Then resolvent = (clause1 - {L}) union (clause2 - {-L})

    Args:
        clause1: First clause (list of literals)
        clause2: Second clause (list of literals)

    Returns:
        Resolvent clause, or None if no resolution possible

    Example:
        clause1 = ["-P", "Q"]
        clause2 = ["P", "R"]
        resolve(clause1, clause2) = ["Q", "R"]  (resolve on P)
    """
    # Find complementary literals
    for lit1 in clause1:
        negated = negate_literal(lit1)
        if negated in clause2:
            # Found complementary pair: lit1 and negated
            # Resolvent = (clause1 - {lit1}) union (clause2 - {negated})
            resolvent = []
            # Add literals from clause1 except lit1
            for lit in clause1:
                if lit != lit1 and lit not in resolvent:
                    resolvent.append(lit)
            # Add literals from clause2 except negated
            for lit in clause2:
                if lit != negated and lit not in resolvent:
                    resolvent.append(lit)
            return resolvent

    # No complementary literals found
    return None


def resolution_entailment(kb: List[str], query: str,
                         verbose: bool = True) -> Tuple[bool, List[str], float]:
    """
    Resolution-based inference for propositional logic.

    Based on Russell & Norvig Figure 7.12, Lecture 8 Part III Slide 110.

    Algorithm (Proof by Refutation):
        1. Convert KB to CNF
        2. Convert -query to CNF
        3. Add -query clauses to KB
        4. Repeat:
            a. Select two clauses that can be resolved
            b. Add resolvent to KB
            c. If resolvent is empty clause [], return True (contradiction)
            d. If no new clauses can be derived, return False
        5. Return False if no contradiction found

    Args:
        kb: Knowledge base (list of sentences)
        query: Query sentence
        verbose: If True, print resolution trace

    Returns:
        Tuple of (entailed, trace, elapsed_time):
        - entailed: True if KB entails query
        - trace: List of resolution steps
        - elapsed_time: Time taken in seconds

    Time Complexity: Exponential worst case (can generate O(2^n) clauses)
    Space Complexity: O(2^n) for clause storage

    Example:
        KB = ["P => Q", "Q => R", "P"]
        Query = "R"
        Result: True (KB entails R)
    """
    t0 = time.perf_counter()

    if verbose:
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

    # Step 1: Convert KB to CNF
    clauses: List[List[str]] = []
    for sentence in kb:
        cnf_clauses = convert_to_cnf(sentence)
        clauses.extend(cnf_clauses)

    if verbose:
        print("KB in CNF:")
        for i, clause in enumerate(clauses, 1):
            clause_str = " | ".join(clause) if clause else "(empty)"
            print(f"  {i}. [{clause_str}]")
        print()

    # Step 2: Negate query and convert to CNF
    # For simple queries (single literal), negation is straightforward
    negated_query = negate_literal(normalize_literal(query))
    query_clauses = [[negated_query]]
    clauses.extend(query_clauses)

    if verbose:
        print("Negated query in CNF:")
        for clause in query_clauses:
            clause_str = " | ".join(clause)
            print(f"  [{clause_str}]")
        print()
        print("Combined clauses (KB union -query):")
        for i, clause in enumerate(clauses, 1):
            clause_str = " | ".join(clause) if clause else "(empty)"
            print(f"  {i}. [{clause_str}]")
        print()

    # Step 3: Apply resolution repeatedly
    trace = []
    iteration = 0
    max_iterations = 100  # Safety limit to prevent infinite loops

    if verbose:
        print("Resolution steps:")
        print()

    while iteration < max_iterations:
        iteration += 1
        new_clauses = []

        # Try to resolve every pair of clauses
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                resolvent = resolve(clauses[i], clauses[j])

                if resolvent is not None:
                    # Check if resolvent is empty clause (contradiction)
                    if len(resolvent) == 0:
                        elapsed = time.perf_counter() - t0
                        if verbose:
                            c1_str = " | ".join(clauses[i])
                            c2_str = " | ".join(clauses[j])
                            print(f"  Step {iteration}: Resolve " +
                                  f"[{c1_str}] and [{c2_str}]")
                            print("            => [] (EMPTY CLAUSE)")
                            print()
                            print("Result: ENTAILED (contradiction found)")
                            print(f"Elapsed time: {elapsed:.6f}s")
                            print()
                        return True, trace + ["Empty clause derived"], elapsed

                    # Check if resolvent is new
                    if resolvent not in clauses and resolvent not in new_clauses:
                        new_clauses.append(resolvent)
                        if verbose:
                            c1_str = " | ".join(clauses[i])
                            c2_str = " | ".join(clauses[j])
                            res_str = " | ".join(resolvent)
                            print(f"  Step {iteration}: Resolve " +
                                  f"[{c1_str}] and [{c2_str}]")
                            print(f"            => [{res_str}]")
                            trace.append(f"Resolve [{c1_str}] and [{c2_str}] " +
                                       f"=> [{res_str}]")

        # If no new clauses, resolution failed to prove entailment
        if not new_clauses:
            elapsed = time.perf_counter() - t0
            if verbose:
                print()
                print("Result: NOT ENTAILED (no new clauses can be derived)")
                print(f"Elapsed time: {elapsed:.6f}s")
                print()
            return False, trace, elapsed

        # Add new clauses to KB
        clauses.extend(new_clauses)

    # Reached iteration limit
    elapsed = time.perf_counter() - t0
    if verbose:
        print()
        print(f"Result: TIMEOUT (reached {max_iterations} iterations)")
        print(f"Elapsed time: {elapsed:.6f}s")
        print()
    return False, trace, elapsed


def test_resolution_entailed() -> None:
    """
    Test resolution on an entailed query.

    KB: P => Q, Q => R, P
    Query: R
    Expected: True (KB entails R)
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
    Test resolution on a non-entailed query.

    KB: P => Q, Q
    Query: P
    Expected: False (KB does not entail P)
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


if __name__ == "__main__":
    """
    Run all Part D demonstrations.
    """
    # Test 1: Entailed query
    test_resolution_entailed()

    print()

    # Test 2: Non-entailed query
    test_resolution_not_entailed()

    print("=" * 80)
    print("Part D: Resolution-Based Inference - COMPLETE")
    print("=" * 80)
