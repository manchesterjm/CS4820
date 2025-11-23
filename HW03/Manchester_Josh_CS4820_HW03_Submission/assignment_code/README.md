# CS 4820/5820 Homework 3 - Logical Agents and Propositional Inference

**Author:** Josh Manchester
**Institution:** University of Colorado Colorado Springs
**Email:** josh.manchester@uccs.edu
**Course:** CS 4820/5820 - Artificial Intelligence

## AI Disclosure

This code was generated with assistance from **Claude Code (Sonnet 4.5)**, version **claude-sonnet-4-5-20250929**.

The AI assistant helped with:
- Implementation of propositional logic inference (Model Checking)
- Implementation of Horn clause inference (Forward Chaining)
- Implementation of Wumpus World reasoning agent
- Implementation of Resolution-based inference (CNF conversion, resolution rule)
- Code structure, documentation, and comments
- Test suite development

All algorithms were implemented according to specifications from:
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" (Chapters 7)
- CS 4820/5820 Lecture 8: Logical Agents (Parts I-IV)

## Requirements

- Python 3.7 or higher
- No external logic or SAT libraries (all algorithms implemented from scratch)

## File Structure

```
assignment_code/
├── propositional_logic.py       # Part A: KB agents, equivalences, model checking
├── horn_inference.py            # Part B: Forward chaining on Horn clauses
├── wumpus_agent.py              # Part C: Wumpus World reasoning agent (core logic)
├── resolution.py                # Part D: Resolution-based inference
├── knowledge_base.py            # KB data structures and utilities
├── inference_engine_base.py     # Base class for inference engines
├── test_all.py                  # Comprehensive test suite
├── run_experiments.py           # Main experiment runner (generates results)
├── run_all.ps1                  # PowerShell script to run all programs
├── HW03_runlog.txt              # Program output
└── README.md                    # This file
```

**Note:** An interactive pygame version of the Wumpus World game is available in the `../interactive_game/` directory.

## Quick Start

### Option 1: Run All Programs with PowerShell Script (Windows)

```powershell
.\run_all.ps1
```

This will:
- Run all four main programs (Parts A, B, C, D)
- Display output on screen in real-time
- Save all output to `HW03_runlog.txt` for screenshots

### Option 2: Run Full Experiments

```bash
python run_experiments.py
```

This will:
- Test Part A: Propositional logic (equivalences, model checking)
- Test Part B: Horn inference on generic KB and Wumpus fragment
- Test Part C: Wumpus agent reasoning for 2 moves
- Test Part D: Resolution entailment
- Output formatted results ready for inclusion in report

### Option 3: Run Test Suite

```bash
python test_all.py
```

This runs automated tests on all components and reports pass/fail status.

### Option 4: Run Individual Programs

```bash
# Part A: Propositional logic
python propositional_logic.py

# Part B: Horn clause inference
python horn_inference.py

# Part C: Wumpus World agent
python wumpus_agent.py

# Part D: Resolution inference
python resolution.py
```

## Assignment Parts

### Part A: KB Agents & Propositional Logic (3.5/3.0 pts UG/Grad)

**File:** `propositional_logic.py`

Implements:
1. **KB Agent Overview** - Conceptual description of KB agents
2. **Logical Equivalences** - Truth tables for De Morgan and Contraposition
3. **Model Checking** - Propositional entailment by model enumeration

**Key algorithms referenced:**
- Model Checking: Russell & Norvig Figure 7.10, Lecture 8 Part I Slide 45

### Part B: Inference Engine (Horn) (4.0 pts both levels)

**File:** `horn_inference.py`

Implements Forward Chaining inference for Horn clauses.

**Algorithm:** Russell & Norvig Figure 7.15, Lecture 8 Part III Slide 80

**Key features:**
- Data-driven reasoning from known facts
- Linear time complexity O(n)
- Maintains count of satisfied premises
- Returns entailment result + inference trace

**Tests on two KBs:**
1. Generic KB (3-5 rules) - simple logical reasoning
2. Wumpus fragment - pit detection from breeze percepts

### Part C: Wumpus Reasoning Agent (5.0/3.5 pts UG/Grad)

**File:** `wumpus_agent.py`

Implements knowledge-based agent for Wumpus World (4×4 grid).

**Algorithm:** Russell & Norvig Section 7.7, Lecture 8 Part I Slides 25-40

**Key features:**
- Start at (1,1)
- Execute exactly 2 moves
- Use Horn inference to determine safe neighbors
- Log reasoning: percepts → KB additions → entailed facts → chosen move

**Rules implemented:**
- Breeze rules (pit detection)
- Stench rules (Wumpus detection) [optional]
- Safety rules (OK := ¬Pit ∧ ¬Wumpus)

### Part D: Resolution Entailment (2.0 pts Grad required; UG bonus)

**File:** `resolution.py`

Implements propositional resolution for CNF.

**Algorithm:** Russell & Norvig Figure 7.12, Section 7.5.2, Lecture 8 Part III Slides 95-110

**Key features:**
- CNF conversion (eliminate ⇒, move ¬ inward, distribute ∨)
- Resolution rule (clause merging)
- Proof by refutation (KB ∧ ¬query → contradiction)
- Clause trace showing derivation

**Tests:**
- Entailed query (derive empty clause)
- Non-entailed query (no contradiction)

## Understanding the Output

### Part A: Propositional Logic

```
=== Logical Equivalence: De Morgan ===

¬(P ∨ Q) ≡ (¬P) ∧ (¬Q)?

Truth Table:
P | Q | P∨Q | ¬(P∨Q) | ¬P | ¬Q | (¬P)∧(¬Q) | Equiv?
--+---+-----+--------+----+----+-----------+--------
T | T |  T  |   F    | F  | F  |     F     |   YES
T | F |  T  |   F    | F  | T  |     F     |   YES
F | T |  T  |   F    | T  | F  |     F     |   YES
F | F |  F  |   T    | T  | T  |     T     |   YES

Result: EQUIVALENT (De Morgan's Law confirmed)
```

### Part B: Horn Inference

```
=== Forward Chaining: Generic KB ===

KB Facts: A, B
KB Rules:
  1. A ∧ B ⇒ C
  2. C ⇒ D

Query: D

Forward Chaining Trace:
  Iteration 1: Derive C (from A, B using rule 1)
  Iteration 2: Derive D (from C using rule 2)
  Query D found!

Result: ENTAILED
Elapsed: 0.0001s
```

### Part C: Wumpus Agent

```
=== Wumpus World Agent ===

Starting position: (1, 1)

--- Move 1 ---
Current position: (1, 1)
Percepts: Breeze=False, Stench=False

KB additions:
  - not_B_1_1 (no breeze at 1,1)
  - not_S_1_1 (no stench at 1,1)

Inference:
  Query: not_P_2_1? → TRUE (from not_B_1_1)
  Query: not_P_1_2? → TRUE (from not_B_1_1)
  Query: not_W_2_1? → TRUE (from not_S_1_1)
  Query: not_W_1_2? → TRUE (from not_S_1_1)

Safe neighbors: [(2,1), (1,2)]
Chosen move: (2,1) [unvisited]

--- Move 2 ---
Current position: (2, 1)
Percepts: Breeze=True, Stench=False

KB additions:
  - B_2_1 (breeze at 2,1)
  - not_S_2_1 (no stench at 2,1)

Inference:
  Pit detected in one of: (3,1), (1,1), (2,2)
  Query: not_P_1_1? → TRUE (visited, still alive)
  Query: not_W_3_1? → TRUE (from not_S_2_1)

Safe neighbors: [(1,1)]
All safe neighbors already visited
Agent stops.

=== Final World State (4×4 grid) ===

   1   2   3   4
1  V   S   ?   ?
2  V   ?   ?   ?
3  ?   ?   ?   ?
4  ?   ?   ?   ?

Legend:
  V = Visited
  S = Safe (entailed, not visited)
  ? = Unknown
```

### Part D: Resolution

```
=== Resolution Test: Entailed Query ===

KB:
  P ⇒ Q
  Q ⇒ R
  P

Query: R

Step 1: Convert KB to CNF
  1. ¬P ∨ Q    (from P ⇒ Q)
  2. ¬Q ∨ R    (from Q ⇒ R)
  3. P         (from P)

Step 2: Negate query and add to KB
  4. ¬R        (from ¬query)

Step 3: Apply resolution
  Resolve [¬Q ∨ R] and [¬R] → [¬Q]
  Resolve [¬P ∨ Q] and [¬Q] → [¬P]
  Resolve [P] and [¬P] → [] (empty clause)

Result: ENTAILED (contradiction found)
```

## Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Model Checking | O(2^n) | O(n) | n = number of symbols |
| Forward Chaining | O(kn) | O(n) | k = max rule size, linear! |
| Resolution | Exponential worst | O(2^n) | Can generate many clauses |

## Safety Features

All implementations include:

1. **Input Validation**
   - Check for valid propositional sentences
   - Validate Horn clause structure
   - Verify grid boundaries in Wumpus World

2. **Clear Error Messages**
   - Explain parsing failures
   - Report invalid queries
   - Show which rules failed to apply

3. **Comprehensive Traces**
   - Show inference steps
   - Log KB modifications
   - Display reasoning process

## Troubleshooting

**"Module not found" error:**
- All code uses only Python standard library
- Make sure you're using Python 3.7 or higher

**Programs running slowly:**
- Model checking is exponential (2^n) - keep symbol sets small
- Forward chaining is linear - should be fast even on larger KBs
- Resolution can be slow on complex KBs

**Tests failing:**
- Check Python version (need 3.7+)
- Verify no modifications to KB structure
- Run individual tests to isolate issues

## Code Documentation

All Python files include:
- Detailed algorithm explanations
- References to textbook (Russell & Norvig) and lecture slides
- Line-by-line comments
- Complexity analysis
- Design decisions and tradeoffs

Read the source code to understand:
- How each algorithm works
- Why it works
- When to use each approach
- Implementation details

## Assignment Deliverables

Based on this code, the assignment requires:

1. **Code submission:** All .py files in this directory
2. **Report (AAAI format):** LaTeX writeup with:
   - Part A: Truth tables and model checking results
   - Part B: Horn inference traces on 2 KBs
   - Part C: Wumpus agent 2-move log with 4×4 grid
   - Part D: Resolution traces for 2 queries
   - Discussion: Algorithm comparisons and insights

3. **Submission format:** One PDF + all source code

## References

- Russell, S. & Norvig, P. "Artificial Intelligence: A Modern Approach" (4th Edition)
  - Chapter 7: Logical Agents
  - Section 7.4: Propositional Logic
  - Section 7.5: Propositional Theorem Proving
  - Section 7.7: Agents Based on Propositional Logic (Wumpus World)

- CS 4820/5820 Lecture Slides:
  - Lecture 8 Part I: Introduction to Logical Agents, Wumpus World
  - Lecture 8 Part II: Propositional Logic Syntax and Semantics
  - Lecture 8 Part III: Inference Methods (Forward/Backward Chaining, Resolution)
  - Lecture 8 Part IV: Summary and Applications

## Contact

For questions about this implementation:
- Josh Manchester
- josh.manchester@uccs.edu

## License

This code is for educational purposes as part of CS 4820/5820 coursework.
Individual work only - do not share or copy.
