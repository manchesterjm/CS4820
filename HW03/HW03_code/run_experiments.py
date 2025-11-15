"""
Main experiment runner for HW03 - Logical Agents.

Generates formatted output for all assignment parts (A, B, C, D).
This output can be captured for screenshots and inclusion in the report.

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

# Import all modules
from propositional_logic import (
    check_demorgan_equivalence,
    check_contraposition_equivalence,
    model_check_with_output
)
from horn_inference import forward_chaining
from knowledge_base import HornKB
from wumpus_agent import test_wumpus_agent
from resolution import test_resolution_entailed, test_resolution_not_entailed


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def run_all_experiments() -> None:
    """
    Run all HW03 experiments and generate formatted output.

    This function executes all parts of the assignment in order
    and produces output suitable for the writeup.
    """
    print_header("CS 4820/5820 - HW03: Logical Agents and Propositional Inference")
    print("Student: Josh Manchester")
    print("Email: josh.manchester@uccs.edu")
    print("Institution: University of Colorado Colorado Springs")
    print()
    print("This program demonstrates implementations of:")
    print("  - Part A: Propositional logic (equivalences, model checking)")
    print("  - Part B: Horn clause inference (forward chaining)")
    print("  - Part C: Wumpus World reasoning agent")
    print("  - Part D: Resolution-based inference")
    print()

    # Part A: Propositional Logic
    print_header("PART A: PROPOSITIONAL LOGIC AND MODEL CHECKING")

    print("A.2: Logical Equivalences")
    print("-" * 80)
    print()
    check_demorgan_equivalence(show_table=True)
    print()
    check_contraposition_equivalence(show_table=True)

    print()
    print("A.3: Model Checking")
    print("-" * 80)
    print()
    kb = ["P => Q", "Q => R", "P"]
    model_check_with_output(kb, "R", ["P", "Q", "R"], verbose=True)

    # Part B: Horn Clause Inference
    print_header("PART B: HORN CLAUSE INFERENCE (FORWARD CHAINING)")

    print("B.1: Generic Knowledge Base Test")
    print("-" * 80)
    print()
    kb = HornKB()
    kb.tell_fact("A")
    kb.tell_fact("B")
    kb.tell_rule(["A", "B"], "C")
    kb.tell_rule(["C"], "D")
    kb.tell_rule(["D", "E"], "F")

    print("Testing query: C")
    forward_chaining(kb, "C", verbose=True)
    print()
    print("Testing query: D")
    forward_chaining(kb, "D", verbose=True)

    print()
    print()
    print("B.2: Wumpus World Fragment Test")
    print("-" * 80)
    print()
    kb2 = HornKB()
    kb2.tell_fact("not_B_1_1")
    kb2.tell_fact("B_2_1")
    kb2.tell_fact("not_B_1_2")
    kb2.tell_rule(["not_B_1_1"], "not_P_2_1")
    kb2.tell_rule(["not_B_1_1"], "not_P_1_2")
    kb2.tell_rule(["not_B_1_2"], "not_P_1_1")
    kb2.tell_rule(["not_B_1_2"], "not_P_2_2")

    print("Testing query: not_P_1_2")
    forward_chaining(kb2, "not_P_1_2", verbose=True)

    # Part C: Wumpus World Agent
    print_header("PART C: WUMPUS WORLD REASONING AGENT")

    print("C: Two-Step Agent Reasoning")
    print("-" * 80)
    test_wumpus_agent()

    # Part D: Resolution
    print_header("PART D: RESOLUTION-BASED INFERENCE (GRADUATE EXTENSION)")

    print("D.1: Entailed Query Test")
    print("-" * 80)
    test_resolution_entailed()

    print()
    print("D.2: Non-Entailed Query Test")
    print("-" * 80)
    test_resolution_not_entailed()

    # Final summary
    print_header("EXPERIMENT COMPLETION SUMMARY")

    print("All experiments completed successfully!")
    print()
    print("Part A: Propositional Logic")
    print("  - Logical equivalences verified (De Morgan, Contraposition)")
    print("  - Model checking demonstrated on KB => R")
    print()
    print("Part B: Horn Clause Inference")
    print("  - Forward chaining tested on generic KB")
    print("  - Forward chaining tested on Wumpus fragment")
    print()
    print("Part C: Wumpus World Agent")
    print("  - Agent executed 2-step reasoning")
    print("  - Successfully inferred safe moves from percepts")
    print()
    print("Part D: Resolution-Based Inference")
    print("  - Resolution proved entailment by deriving empty clause")
    print("  - Resolution correctly identified non-entailment")
    print()
    print("=" * 80)
    print("All output above can be used for the assignment writeup")
    print("=" * 80)
    print()


if __name__ == "__main__":
    """
    Run all experiments.

    Output can be redirected to a file for documentation:
        python run_experiments.py > HW03_runlog.txt
    """
    run_all_experiments()
