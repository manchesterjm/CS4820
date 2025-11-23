# CS 4820/5820 - HW03 Submission

**Student:** Josh Manchester
**Email:** josh.manchester@uccs.edu
**Institution:** University of Colorado Colorado Springs
**Date:** November 15, 2025
**Assignment:** Homework 3 - Logical Agents and Propositional Inference

---

## Submission Contents

This submission package contains:

1. **`Manchester_Josh_CS4820_HW03_Writeup.pdf`** (148 KB)
   - Complete AAAI-formatted report covering all four parts
   - ~12 pages including tables, code listings, and analysis
   - References to Russell & Norvig textbook and Professor Atyabi's Lecture 8 slides

2. **`assignment_code/`** - Required assignment implementation (Parts A-D)
   - All Python files for Parts A, B, C, D
   - Demonstration output (HW03_runlog.txt)
   - README with detailed usage instructions

3. **`interactive_game/`** - Bonus: Interactive Wumpus World game
   - Visual pygame implementation of Wumpus World
   - Knowledge-based agent with real-time reasoning display
   - README with game instructions

---

## Quick Start - Running the Code

### Prerequisites
- Python 3.7 or higher
- **For assignment code (Parts A-D):** No external packages required (standard library only)
- **For interactive game (bonus):** Install requirements with `pip install -r requirements.txt`

### Run Assignment Code (Required Parts A-D)

```bash
cd assignment_code/

# Run all demonstrations
python run_experiments.py

# Or run individual parts
python propositional_logic.py    # Part A
python horn_inference.py          # Part B
python wumpus_agent.py            # Part C
python resolution.py              # Part D
```

### Run Interactive Game (Bonus)

```bash
# Install requirements (pygame-ce)
pip install -r requirements.txt

# Run the game
cd interactive_game/
python wumpus_game_visual.py
```

**Controls:** SPACE (step), A (auto-play), R (reset), Q (quit)

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

- **Lines of Code:** ~2,100 lines
- **Documentation:** Comprehensive docstrings and comments

---

## Files Included

### Documentation
- `Manchester_Josh_CS4820_HW03_Writeup.pdf` - Main report
- `README.md` - This file
- `requirements.txt` - Python package requirements (pygame-ce for interactive game only)

### Assignment Code (assignment_code/)
- `inference_engine_base.py` - Abstract base classes
- `propositional_logic.py` - Part A implementation (526 lines)
- `horn_inference.py` - Part B implementation (570 lines)
- `wumpus_agent.py` - Part C implementation (450 lines)
- `resolution.py` - Part D implementation (485 lines)
- `knowledge_base.py` - KB data structures (372 lines)
- `run_experiments.py` - Demonstration runner (129 lines)
- `HW03_runlog.txt` - Complete program output (11 KB)
- `README.md` - Detailed instructions

### Interactive Game (interactive_game/)
- `wumpus_game_visual.py` - Pygame interactive game
- `wumpus_agent.py` - Core agent logic
- `knowledge_base.py` - KB data structures
- `inference_engine_base.py` - Base inference engine
- `README.md` - Game instructions and features

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

---

## AI Disclosure

This code was developed with assistance from **Claude Code (Sonnet 4.5)**, version `claude-sonnet-4-5-20250929`.

AI assistance included:
- Algorithm implementation based on textbook specifications
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
✅ All source code included
✅ README with usage instructions
✅ AI disclosure included
✅ All four parts implemented (A, B, C, D)
✅ References to textbook and lecture slides

**Ready for Canvas submission!**
