"""
Comprehensive test suite for HW03 - Logical Agents.

Tests all components:
- Part A: Propositional logic (equivalences, model checking)
- Part B: Horn clause inference (forward chaining)
- Part C: Wumpus World agent
- Part D: Resolution-based inference

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from propositional_logic import (
    check_demorgan_equivalence,
    check_contraposition_equivalence,
    model_check_with_output
)
from horn_inference import forward_chaining
from knowledge_base import HornKB
from wumpus_agent import WumpusAgent, WumpusWorld, test_wumpus_agent
from resolution import resolution_entailment


def test_part_a_equivalences() -> bool:
    """Test Part A: Logical equivalences."""
    print("Testing Part A: Logical Equivalences...")

    try:
        # Test De Morgan's Law
        result = check_demorgan_equivalence(show_table=False)
        assert result is True, "De Morgan's Law failed"

        # Test Contraposition
        result = check_contraposition_equivalence(show_table=False)
        assert result is True, "Contraposition failed"

        print("  PASS: All equivalences correct")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


def test_part_a_model_checking() -> bool:
    """Test Part A: Model checking."""
    print("Testing Part A: Model Checking...")

    try:
        # Test 1: Entailed query
        kb = ["P => Q", "Q => R", "P"]
        result = model_check_with_output(kb, "R", ["P", "Q", "R"], verbose=False)
        assert result is True, "Expected R to be entailed"

        # Test 2: Non-entailed query
        kb = ["P => Q", "Q"]
        result = model_check_with_output(kb, "P", ["P", "Q"], verbose=False)
        assert result is False, "Expected P to NOT be entailed"

        print("  PASS: Model checking works correctly")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


def test_part_b_generic_kb() -> bool:
    """Test Part B: Forward chaining on generic KB."""
    print("Testing Part B: Forward Chaining (Generic KB)...")

    try:
        kb = HornKB()
        kb.tell_fact("A")
        kb.tell_fact("B")
        kb.tell_rule(["A", "B"], "C")
        kb.tell_rule(["C"], "D")
        kb.tell_rule(["D", "E"], "F")

        # Test 1: C should be entailed
        entailed, _, _ = forward_chaining(kb, "C", verbose=False)
        assert entailed is True, "Expected C to be entailed"

        # Test 2: D should be entailed
        entailed, _, _ = forward_chaining(kb, "D", verbose=False)
        assert entailed is True, "Expected D to be entailed"

        # Test 3: F should NOT be entailed
        entailed, _, _ = forward_chaining(kb, "F", verbose=False)
        assert entailed is False, "Expected F to NOT be entailed"

        print("  PASS: Forward chaining works correctly on generic KB")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


def test_part_b_wumpus_kb() -> bool:
    """Test Part B: Forward chaining on Wumpus fragment."""
    print("Testing Part B: Forward Chaining (Wumpus KB)...")

    try:
        kb = HornKB()
        kb.tell_fact("not_B_1_1")
        kb.tell_fact("B_2_1")
        kb.tell_fact("not_B_1_2")

        kb.tell_rule(["not_B_1_1"], "not_P_2_1")
        kb.tell_rule(["not_B_1_1"], "not_P_1_2")
        kb.tell_rule(["not_B_1_2"], "not_P_1_1")
        kb.tell_rule(["not_B_1_2"], "not_P_2_2")

        # Test 1: not_P_1_2 should be entailed
        entailed, _, _ = forward_chaining(kb, "not_P_1_2", verbose=False)
        assert entailed is True, "Expected not_P_1_2 to be entailed"

        # Test 2: not_P_2_1 should be entailed
        entailed, _, _ = forward_chaining(kb, "not_P_2_1", verbose=False)
        assert entailed is True, "Expected not_P_2_1 to be entailed"

        print("  PASS: Forward chaining works correctly on Wumpus KB")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


def test_part_c_wumpus_agent() -> bool:
    """Test Part C: Wumpus World agent."""
    print("Testing Part C: Wumpus World Agent...")

    try:
        # Create world with pit and Wumpus
        world = WumpusWorld(grid_size=4)
        world.add_pit(3, 1)
        world.add_wumpus(1, 3)

        # Create agent
        agent = WumpusAgent(grid_size=4)

        # Agent should start at (1,1)
        assert agent.get_position() == (1, 1), "Agent should start at (1,1)"

        # Run two steps
        steps = agent.run_n_steps(world, n=2)

        # Agent should have made at least one move
        assert len(agent.get_visited()) >= 2, "Agent should have visited at least 2 cells"

        # Agent should have returned steps
        assert len(steps) >= 1, "Agent should have returned at least 1 step"

        print("  PASS: Wumpus agent executes correctly")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


def test_part_d_resolution() -> bool:
    """Test Part D: Resolution-based inference."""
    print("Testing Part D: Resolution-Based Inference...")

    try:
        # Test 1: Entailed query
        kb = ["P => Q", "Q => R", "P"]
        entailed, _, _ = resolution_entailment(kb, "R", verbose=False)
        assert entailed is True, "Expected R to be entailed"

        # Test 2: Non-entailed query
        kb = ["P => Q", "Q"]
        entailed, _, _ = resolution_entailment(kb, "P", verbose=False)
        assert entailed is False, "Expected P to NOT be entailed"

        print("  PASS: Resolution works correctly")
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


def run_all_tests() -> None:
    """Run all tests and report results."""
    print("=" * 80)
    print("HW03 - Comprehensive Test Suite")
    print("=" * 80)
    print()

    tests = [
        ("Part A: Equivalences", test_part_a_equivalences),
        ("Part A: Model Checking", test_part_a_model_checking),
        ("Part B: Generic KB", test_part_b_generic_kb),
        ("Part B: Wumpus KB", test_part_b_wumpus_kb),
        ("Part C: Wumpus Agent", test_part_c_wumpus_agent),
        ("Part D: Resolution", test_part_d_resolution),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print()

    # Print summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()

    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {test_name}")

    print()
    print(f"Total: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"WARNING: {failed} test(s) failed")

    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
