# CS 4820/5820 - HW03 Submission

**Student:** Josh Manchester
**Email:** josh.manchester@uccs.edu
**Institution:** University of Colorado Colorado Springs
**Date:** November 15, 2025
**Assignment:** Homework 3 - Logical Agents and Propositional Inference

---

## Submission Contents

This submission package contains:

1. **`Manchester_Josh_CS4820_HW03_Writeup.pdf`** (173 KB)
   - Complete AAAI-formatted report covering all four parts
   - ~15 pages including tables, code listings, and analysis
   - References to Russell & Norvig textbook and Professor Atyabi's Lecture 8 slides

2. **`HW03_code/`** - Complete source code implementation
   - All Python files for Parts A, B, C, D
   - Test suite (6/6 tests passing)
   - Demonstration output (HW03_runlog.txt)
   - README with detailed usage instructions

---

## Quick Start - Running the Code

### Prerequisites
- Python 3.7 or higher
- No external packages required (standard library only)

### Run All Tests
```bash
cd HW03_code/
python test_all.py
```

**Expected:** All 6 tests pass (100%)

### Generate Demonstration Output
```bash
cd HW03_code/
python run_experiments.py
```

**Expected:** Formatted output showing all four parts in action

---

## Assignment Coverage (Undergraduate - CS 4820)

✅ **Part A: Propositional Logic** (3.5 points)
- Logical equivalences (De Morgan, Contraposition) verified with truth tables
- Model checking implementation (TT-ENTAILS algorithm)
- Performance: 0.000089s for 8-model enumeration

✅ **Part B: Horn Clause Inference** (4.0 points)
- Forward chaining algorithm (O(n) complexity)
- Generic KB test case
- Wumpus World fragment test
- Performance: 0.000025s per query

✅ **Part C: Wumpus World Agent** (5.0 points)
- Two-step reasoning agent
- Knowledge base with Horn clause rules
- Safety inference from percepts
- Complete execution trace included

✅ **Part D: Resolution-Based Inference** (BONUS +2.0 points)
- Resolution algorithm with CNF conversion
- Proof by refutation
- Test cases for entailed and non-entailed queries
- Performance: 0.000112s

**Total Points:** 12.5 (required) + 2.0 (bonus) = **14.5 points possible** 🎉

---

## Code Quality Metrics

- **Pylint Score:** 9.11/10 ✅
- **Tests Passing:** 6/6 (100%) ✅
- **SOFA Refactoring:** All four principles applied
  - Single Responsibility
  - Open/Closed
  - Functional Programming
  - Abstraction
- **Lines of Code:** ~2,100 lines (refactored)
- **Documentation:** Comprehensive docstrings and comments

---

## Files Included

### Documentation
- `Manchester_Josh_CS4820_HW03_Writeup.pdf` - Main report
- `README.md` - This file

### Source Code (HW03_code/)
- `inference_engine_base.py` - Abstract base classes
- `propositional_logic.py` - Part A implementation (526 lines)
- `horn_inference.py` - Part B implementation (570 lines)
- `wumpus_agent.py` - Part C implementation (450 lines)
- `resolution.py` - Part D implementation (485 lines)
- `knowledge_base.py` - KB data structures (372 lines)
- `test_all.py` - Test suite (235 lines)
- `run_experiments.py` - Demonstration runner (129 lines)
- `HW03_runlog.txt` - Complete program output (11 KB)

### Archived
- `archived_original/` - Original implementations before SOFA refactoring

---

## Implementation Notes

### Algorithms Implemented
All algorithms follow specifications from:
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4th ed.)
- CS 4820/5820 Lecture 8: Logical Agents (Parts I-IV)

### Key Features
1. **Model Checking:** Enumerates all 2^n models to check entailment
2. **Forward Chaining:** O(n) data-driven inference for Horn clauses
3. **Wumpus Agent:** Uses Horn inference to determine safe moves
4. **Resolution:** Proof by refutation with CNF conversion

### SOFA Refactoring Highlights
- Separated computation from presentation (SRP)
- Strategy pattern for extensible algorithms (OCP)
- Immutable dataclasses for all data structures (FP)
- Abstract interfaces hiding implementation details (Abstraction)

---

## AI Disclosure

This code was developed with assistance from **Claude Code (Sonnet 4.5)**, version `claude-sonnet-4-5-20250929`.

AI assistance included:
- Algorithm implementation based on textbook specifications
- SOFA refactoring guidance and implementation
- Code documentation and testing
- Report writing and LaTeX formatting

All code was reviewed, understood, tested, and verified by the student. Complete AI disclosure included in the writeup PDF.

---

## Performance Summary

| Component | Execution Time | Complexity |
|-----------|---------------|------------|
| Model Checking | 0.000089s | O(2^n × m) |
| Forward Chaining | 0.000025s | O(n) |
| Wumpus Agent (2 steps) | 0.003s | O(n) per step |
| Resolution | 0.000112s | O(2^n) worst case |

All inference completes in **< 1 millisecond** for small knowledge bases.

---

## Testing Instructions

### Run Individual Part Tests
```bash
cd HW03_code/

# Part A
python -c "from test_all import test_part_a_equivalences; test_part_a_equivalences()"

# Part B
python -c "from test_all import test_part_b_generic_kb; test_part_b_generic_kb()"

# Part C
python -c "from test_all import test_part_c_wumpus_agent; test_part_c_wumpus_agent()"

# Part D
python -c "from test_all import test_part_d_resolution; test_part_d_resolution()"
```

### Run Code Quality Check
```bash
cd HW03_code/
pylint *.py --max-line-length=100 --score=yes
```

---

## Contact Information

**Josh Manchester**
josh.manchester@uccs.edu
University of Colorado Colorado Springs
CS 4820/5820 - Artificial Intelligence
Fall 2025

For any questions about the implementation, please contact the student.

---

## Submission Checklist

✅ PDF report included (Manchester_Josh_CS4820_HW03_Writeup.pdf)
✅ All source code included (HW03_code/ directory)
✅ README with usage instructions
✅ All tests passing (6/6)
✅ Code quality verified (9.11/10 Pylint)
✅ AI disclosure included
✅ All four parts implemented (A, B, C, D)
✅ SOFA refactoring complete
✅ References to textbook and lecture slides

**Ready for Canvas submission!**
