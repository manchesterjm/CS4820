# CS 4820/5820 Homework 3 - Implementation Plan

**Student:** Josh Manchester
**Email:** josh.manchester@uccs.edu
**Assignment:** Logical Agents and Propositional Inference
**Total Points:** 12.5 pts (both levels)
**Date Created:** November 15, 2025

---

## Assignment Overview

### Point Allocation
| Part | Description | CS 4820 (UG) | CS 5820 (Grad) |
|------|-------------|--------------|----------------|
| A | KB Agents & Propositional Logic | 3.5 | 3.0 |
| B | Inference Engine (Horn) | 4.0 | 4.0 |
| C | Wumpus Reasoning (2 moves) | 5.0 | 3.5 |
| D | Graduate Extension | Bonus +2.0 | 2.0 (required) |
| **Total** | | **12.5** | **12.5** |

### Key Requirements
- ✅ Implement from scratch (no SAT/logic libraries)
- ✅ Python implementation following CS4820_STYLE_GUIDE.md
- ✅ AAAI-formatted PDF writeup following CS4820_WRITING_GUIDE.md
- ✅ Keep KBs small (4×4 grids)
- ✅ Pylint score ≥ 9.0/10 target
- ✅ Individual work only

---

## Directory Structure (Mirroring HW02)

```
HW03/
├── HW03_code/
│   ├── propositional_logic.py       # Part A: Model checking, equivalences
│   ├── horn_inference.py            # Part B: Forward/Backward Chaining
│   ├── wumpus_agent.py              # Part C: Wumpus World reasoning
│   ├── resolution.py                # Part D Option 1: Resolution
│   ├── scaling_study.py             # Part D Option 2: Scaling analysis
│   ├── knowledge_base.py            # KB data structures and utilities
│   ├── test_all.py                  # Comprehensive test suite
│   ├── run_experiments.py           # Generate all results
│   ├── run_all.ps1                  # PowerShell batch script
│   ├── requirements.txt             # Python dependencies (none expected)
│   ├── README.md                    # How to run everything
│   └── HW03_runlog.txt             # Program output
├── writeup/
│   ├── assignment_writeup.tex       # AAAI-formatted report
│   ├── aaai24.sty                   # AAAI style file
│   ├── aaai24.bst                   # AAAI bibliography style
│   └── references.bib               # Bibliography
├── Manchester_Josh_CS4820_HW03_Submission/  # Final submission
│   ├── Manchester_Josh_CS4820_HW03_Writeup.pdf
│   └── code/                        # All Python files
├── IMPLEMENTATION_PLAN.md           # This file
└── HW03_Extras_need_to_do.txt      # Original assignment notes
```

---

## Part A: KB Agents & Propositional Logic (3.5/3.0 pts)

### A1. KB Agent Overview (0.5 pt)

**Implementation:** Write conceptual explanation in `propositional_logic.py` docstring

**Coverage:**
- Knowledge Base (KB) structure
- Tell operation (add facts/rules)
- Ask operation (query entailment)
- Inference mechanisms
- Percepts → Actions flow

**Reference:** Russell & Norvig Section 7.1-7.2, Lecture 8 Part I Slides 1-15

### A2. Logical Equivalences (1.0 pt)

**File:** `propositional_logic.py`

**Functions to implement:**
```python
def check_equivalence_demorgan(show_table=True) -> bool:
    """
    Check if ¬(P ∨ Q) ≡ (¬P) ∧ (¬Q) using truth table.

    Based on Russell & Norvig Section 7.4.2 (Logical Equivalence).
    De Morgan's Law: ¬(A ∨ B) ≡ (¬A) ∧ (¬B)

    Returns:
        True if equivalent, False otherwise
    """
    pass

def check_equivalence_contraposition(show_table=True) -> bool:
    """
    Check if (P ⇒ Q) ≡ (¬Q ⇒ ¬P) using truth table.

    Based on Russell & Norvig Section 7.4.2 (Logical Equivalence).
    Contraposition: (A ⇒ B) ≡ (¬B ⇒ ¬A)

    Returns:
        True if equivalent, False otherwise
    """
    pass
```

**Output:** 4-row truth tables showing equivalence

### A3. Model Checking (2.0/1.5 pts)

**File:** `propositional_logic.py`

**Function to implement:**
```python
def model_check(kb: List[str], query: str, symbols: List[str]) -> bool:
    """
    Model checking algorithm for propositional logic.

    Based on Russell & Norvig Figure 7.10, Lecture 8 Part I Slide 45.
    Enumerates all possible truth assignments to symbols and checks
    if KB |= query (query is true in all models where KB is true).

    Args:
        kb: List of propositional sentences (e.g., ["P => Q", "Q => R", "P"])
        query: Query sentence (e.g., "R")
        symbols: List of propositional symbols (e.g., ["P", "Q", "R"])

    Returns:
        True if KB entails query, False otherwise

    Time Complexity: O(2^n) where n = number of symbols
    Space Complexity: O(n) for recursion depth
    """
    pass
```

**Test case:**
- KB: `P ⇒ Q, Q ⇒ R, P`
- Symbols: `{P, Q, R}`
- Query: `R`
- Expected: `True` (entailed)

**Output:** Concise reasoning (≤6 lines showing which models satisfy KB and query)

---

## Part B: Inference Engine (Horn Clauses) (4.0 pts)

### B1. Implementation (2.5 pts)

**File:** `horn_inference.py`

**Choose ONE algorithm to implement:**

#### Option 1: Forward Chaining (Data-Driven) ⭐ RECOMMENDED

**Reference:** Russell & Norvig Figure 7.15, Lecture 8 Part III Slide 80

```python
def forward_chaining(kb: HornKB, query: str, verbose=True) -> Tuple[bool, List[str]]:
    """
    Forward chaining inference for Horn clauses.

    Based on Russell & Norvig Figure 7.15, Lecture 8 Part III Slide 80.
    Data-driven reasoning: start with known facts, apply rules until query
    is derived or no new facts can be inferred.

    Algorithm:
        1. Initialize count[rule] = number of premises for each rule
        2. Initialize inferred = {} (set of derived facts)
        3. Initialize agenda = queue of known facts
        4. While agenda not empty:
            - Pop fact from agenda
            - If fact == query, return True
            - For each rule where fact is in premises:
                - Decrement count[rule]
                - If count[rule] == 0, add conclusion to agenda
        5. Return False if agenda exhausted

    Args:
        kb: Horn clause knowledge base (facts + rules)
        query: Ground atom to prove
        verbose: If True, print inference trace

    Returns:
        Tuple of (entailed: bool, trace: List[str])
        - entailed: True if query is entailed
        - trace: List of derived facts in order

    Time Complexity: O(n) where n = KB size (linear time!)
    Space Complexity: O(n) for count and inferred structures
    """
    pass
```

#### Option 2: Backward Chaining (Goal-Driven)

**Reference:** Russell & Norvig Figure 7.16, Lecture 8 Part III Slide 85

```python
def backward_chaining(kb: HornKB, query: str, verbose=True) -> Tuple[bool, List[str]]:
    """
    Backward chaining inference for Horn clauses.

    Based on Russell & Norvig Figure 7.16, Lecture 8 Part III Slide 85.
    Goal-driven reasoning: start with query, work backwards to find
    supporting facts.

    Algorithm:
        1. If query is a known fact, return True
        2. For each rule with query as conclusion:
            - Recursively prove all premises
            - If all premises proven, return True
        3. Return False if no supporting rules

    Args:
        kb: Horn clause knowledge base (facts + rules)
        query: Ground atom to prove
        verbose: If True, print goal stack expansion

    Returns:
        Tuple of (entailed: bool, trace: List[str])
        - entailed: True if query is entailed
        - trace: List of goals expanded (stack trace)

    Time Complexity: O(n) for well-structured KBs
    Space Complexity: O(d) where d = proof depth
    """
    pass
```

### B2. Test on Two KBs (1.5 pts)

**Test 1: Generic KB (3-5 rules)**

Example KB:
```python
GENERIC_KB = {
    'facts': ['A', 'B'],
    'rules': [
        (['A', 'B'], 'C'),  # A ∧ B ⇒ C
        (['C'], 'D'),        # C ⇒ D
        (['D', 'E'], 'F'),   # D ∧ E ⇒ F
    ]
}

# Queries to test:
# - Ask('C'): Should return True (derived from A, B)
# - Ask('D'): Should return True (derived from C)
# - Ask('F'): Should return False (E not known)
```

**Test 2: Wumpus Fragment**

KB from assignment:
```python
WUMPUS_KB = {
    'facts': ['not_B_1_1', 'B_2_1', 'B_1_2'],
    'rules': [
        # Breezy if pit in adjacent cell
        # B(x,y) ⇐ P(x+1,y) ∨ P(x-1,y) ∨ P(x,y+1) ∨ P(x,y-1)
        (['P_2_1'], 'B_1_1'),
        (['P_1_2'], 'B_1_1'),
        # Not breezy means no adjacent pits
        # ¬B(x,y) ⇒ ¬P(x+1,y) ∧ ¬P(x-1,y) ∧ ¬P(x,y+1) ∧ ¬P(x,y-1)
        (['not_B_1_1'], 'not_P_2_1'),
        (['not_B_1_1'], 'not_P_1_2'),
        # Add rules for (2,1) and (1,2) neighbors
    ]
}

# Queries to test (from assignment):
# - Ask('not_P_1_2'): Should return True (no breeze at 1,1)
# - Ask('not_P_2_1'): Should return True (no breeze at 1,1)
```

**Output format:**
```
=== Test: Generic KB ===
Query: C
Result: True (entailed)
Trace:
  1. Start with facts: A, B
  2. Apply rule (A ∧ B ⇒ C): derive C
  3. Query C found in derived facts
Elapsed: 0.0001s

Query: F
Result: False (not entailed)
Trace:
  1. Start with facts: A, B
  2. Apply rule (A ∧ B ⇒ C): derive C
  3. Apply rule (C ⇒ D): derive D
  4. Cannot apply rule (D ∧ E ⇒ F): E not known
  5. No more rules to apply
Elapsed: 0.0001s
```

---

## Part C: Wumpus Reasoning Agent (5.0/3.5 pts)

### C1. Horn Rules (0.5 pt)

**File:** `wumpus_agent.py`

**Rules to implement:**

```python
# Wumpus World Rules (4×4 grid)
# Based on Russell & Norvig Section 7.7, Lecture 8 Part I Slides 25-40

# BREEZE RULES:
# If there's a breeze at (x,y), then there's a pit in an adjacent cell
# B(x,y) ⇒ P(x+1,y) ∨ P(x-1,y) ∨ P(x,y+1) ∨ P(x,y-1)

# If there's no breeze at (x,y), then no pits in adjacent cells
# ¬B(x,y) ⇒ ¬P(x+1,y) ∧ ¬P(x-1,y) ∧ ¬P(x,y+1) ∧ ¬P(x,y-1)

# STENCH RULES (optional):
# If there's a stench at (x,y), then the Wumpus is in an adjacent cell
# S(x,y) ⇒ W(x+1,y) ∨ W(x-1,y) ∨ W(x,y+1) ∨ W(x,y-1)

# If there's no stench at (x,y), then no Wumpus in adjacent cells
# ¬S(x,y) ⇒ ¬W(x+1,y) ∧ ¬W(x-1,y) ∧ ¬W(x,y+1) ∧ ¬W(x,y-1)

# SAFETY RULE:
# A cell is safe if it has no pit and no Wumpus
# OK(x,y) := ¬P(x,y) ∧ ¬W(x,y)
```

**Implementation note:** Convert disjunctions to multiple Horn clauses or use simplified representation

### C2. Agent Loop (3.0/2.0 pts)

**File:** `wumpus_agent.py`

```python
class WumpusAgent:
    """
    Knowledge-based agent for Wumpus World.

    Based on Russell & Norvig Section 7.7, Lecture 8 Part I Slides 25-40.
    Uses Horn clause inference to determine safe moves.
    """

    def __init__(self, grid_size=4):
        """Initialize agent at (1,1) with empty KB."""
        self.kb = HornKB()
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.grid_size = grid_size
        self.move_log = []

    def sense(self, percepts: Dict[str, bool]) -> None:
        """
        Add percepts to KB.

        Args:
            percepts: Dict with keys 'breeze', 'stench', etc.
        """
        pass

    def infer_safe_neighbors(self) -> List[Tuple[int, int]]:
        """
        Use inference engine to determine which neighbors are safe.

        Returns:
            List of (x, y) coordinates that are entailed safe
        """
        pass

    def choose_move(self, safe_neighbors: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Choose an unvisited safe neighbor.

        Args:
            safe_neighbors: List of cells entailed to be safe

        Returns:
            Next move (x, y) or None if no safe unvisited neighbors
        """
        pass

    def make_move(self, new_position: Tuple[int, int]) -> None:
        """Update position and visited set."""
        pass

    def run_two_steps(self, world: WumpusWorld) -> None:
        """
        Execute exactly two moves with reasoning.

        For each step:
            1. Sense percepts at current cell
            2. Add percepts to KB
            3. Run inference to find safe neighbors
            4. Choose and make move to safe unvisited neighbor
            5. Log the reasoning process
        """
        pass
```

**Output format:**
```
=== Wumpus World Agent ===

Step 1:
  Position: (1, 1)
  Percepts: {'breeze': False, 'stench': False}
  KB additions: not_B_1_1, not_S_1_1
  Inference results:
    - not_P_2_1: True (entailed from not_B_1_1)
    - not_P_1_2: True (entailed from not_B_1_1)
    - not_W_2_1: True (entailed from not_S_1_1)
    - not_W_1_2: True (entailed from not_S_1_1)
  Safe neighbors: [(2, 1), (1, 2)]
  Chosen move: (2, 1) [unvisited]

Step 2:
  Position: (2, 1)
  Percepts: {'breeze': True, 'stench': False}
  KB additions: B_2_1, not_S_2_1
  Inference results:
    - Pit in one of: (3,1), (1,1), (2,2) [from B_2_1]
    - not_P_1_1: True (we're alive, visited there)
    - not_W_3_1: True (entailed from not_S_2_1)
    - not_W_1_1: True (entailed from not_S_2_1)
    - not_W_2_2: True (entailed from not_S_2_1)
  Safe neighbors: [(1, 1)] [already visited]
  No safe unvisited neighbors found
  Agent stops.
```

### C3. Mini Table (1.5/1.0 pts)

**Output:** 4×4 grid showing visited/safe/unknown cells

```
=== World State After 2 Moves ===

   1   2   3   4
1  V   S   ?   ?
2  V   ?   ?   ?
3  ?   ?   ?   ?
4  ?   ?   ?   ?

Legend:
  V = Visited
  S = Safe (entailed, not visited)
  ? = Unknown
  X = Unsafe (pit or wumpus detected)
```

---

## Part D: Graduate Extension (2.0 pts required for Grad; bonus for UG)

### Option 1: Resolution Entailment ⭐ RECOMMENDED

**File:** `resolution.py`

**Reference:** Russell & Norvig Section 7.5.2, Figure 7.12, Lecture 8 Part III Slides 95-110

```python
def convert_to_cnf(sentence: str) -> List[List[str]]:
    """
    Convert propositional sentence to CNF.

    Based on Russell & Norvig Section 7.5.2.

    Steps:
        1. Eliminate implications: (A ⇒ B) becomes (¬A ∨ B)
        2. Move negations inward (De Morgan's laws)
        3. Distribute ∨ over ∧
        4. Flatten to list of clauses

    Args:
        sentence: Propositional sentence string

    Returns:
        List of clauses, where each clause is a list of literals

    Example:
        Input: "(P => Q) & (Q => R)"
        Output: [['-P', 'Q'], ['-Q', 'R']]
    """
    pass

def resolve(clause1: List[str], clause2: List[str]) -> Optional[List[str]]:
    """
    Resolution rule: resolve two clauses if possible.

    Based on Russell & Norvig Figure 7.12.

    If clause1 contains literal L and clause2 contains ¬L,
    then resolvent = (clause1 - {L}) ∪ (clause2 - {¬L})

    Args:
        clause1: List of literals
        clause2: List of literals

    Returns:
        Resolvent clause or None if no resolution possible
    """
    pass

def resolution_entailment(kb: List[List[str]], query: str) -> Tuple[bool, List[str]]:
    """
    Resolution-based inference for propositional logic.

    Based on Russell & Norvig Figure 7.12, Lecture 8 Part III Slide 110.

    Algorithm:
        1. Convert KB ∧ ¬query to CNF
        2. Repeatedly apply resolution to derive new clauses
        3. If empty clause derived, return True (contradiction found)
        4. If no new clauses, return False

    Args:
        kb: Knowledge base in CNF (list of clauses)
        query: Query sentence

    Returns:
        Tuple of (entailed: bool, trace: List[str])
    """
    pass
```

**Test cases:**
1. **Entailed query:** KB = {P ⇒ Q, Q ⇒ R, P}, Query = R
   - Expected: True (derive empty clause)

2. **Not entailed:** KB = {P ⇒ Q, Q}, Query = P
   - Expected: False (no contradiction)

**Output:**
```
=== Resolution Test 1: Entailed Query ===
KB (CNF):
  1. ['-P', 'Q']  (from P => Q)
  2. ['-Q', 'R']  (from Q => R)
  3. ['P']        (from P)

Negated query: ['-R']

Resolution trace:
  Step 1: Resolve ['-Q', 'R'] and ['-R'] => ['-Q']
  Step 2: Resolve ['-P', 'Q'] and ['-Q'] => ['-P']
  Step 3: Resolve ['P'] and ['-P'] => [] (empty clause)

Result: ENTAILED (contradiction found)
```

### Option 2: Scaling Study

**File:** `scaling_study.py`

```python
def generate_kb(base_kb: HornKB, k: int) -> HornKB:
    """
    Generate larger KB by adding k neutral facts/rules.

    Args:
        base_kb: Original knowledge base
        k: Number of neutral facts/rules to add

    Returns:
        Extended knowledge base
    """
    pass

def run_scaling_experiment(k_values: List[int] = [5, 10, 15]) -> pd.DataFrame:
    """
    Measure inference runtime vs. KB size.

    For each k:
        1. Generate KB with k extra facts/rules
        2. Run inference 10 times
        3. Measure average runtime and number of inferences

    Returns:
        DataFrame with columns: k, avg_time, num_inferences
    """
    pass

def plot_scaling_results(results: pd.DataFrame) -> None:
    """Generate plot of runtime vs. k."""
    pass
```

---

## Code Structure and Standards

### Knowledge Base Data Structure

**File:** `knowledge_base.py`

```python
class HornKB:
    """
    Horn clause knowledge base.

    Based on Russell & Norvig Section 7.5.3.

    Supports:
        - Facts (ground atoms)
        - Horn clauses (conjunctions => single atom)
    """

    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str]] = []

    def tell_fact(self, fact: str) -> None:
        """Add a ground fact to KB."""
        self.facts.add(fact)

    def tell_rule(self, premises: List[str], conclusion: str) -> None:
        """Add a Horn clause: premises => conclusion."""
        self.rules.append((premises, conclusion))

    def ask(self, query: str) -> bool:
        """Query KB using inference engine."""
        pass
```

### Style Guide Compliance

Following **CS4820_STYLE_GUIDE.md**:

- ✅ Module docstrings with algorithm references
- ✅ Function docstrings with Args/Returns/Complexity
- ✅ Type hints for all functions
- ✅ Algorithm references in comments (R&N chapters, lecture slides)
- ✅ Complexity analysis documented
- ✅ No external logic libraries
- ✅ Timeout protection (where applicable)
- ✅ ASCII-only output (no Unicode)
- ✅ Lines ≤ 100 characters
- ✅ Pylint score ≥ 9.0/10

### Testing Standards

**File:** `test_all.py`

```python
def test_propositional_equivalences():
    """Test Part A logical equivalences."""
    assert check_equivalence_demorgan() == True
    assert check_equivalence_contraposition() == True

def test_model_checking():
    """Test Part A model checking."""
    kb = ["P => Q", "Q => R", "P"]
    assert model_check(kb, "R", ["P", "Q", "R"]) == True

def test_forward_chaining_generic():
    """Test Part B forward chaining on generic KB."""
    pass

def test_forward_chaining_wumpus():
    """Test Part B forward chaining on Wumpus KB."""
    pass

def test_wumpus_agent_two_moves():
    """Test Part C Wumpus agent makes 2 safe moves."""
    pass

def test_resolution_entailment():
    """Test Part D resolution on sample queries."""
    pass
```

---

## Writeup Structure (AAAI Format)

Following **CS4820_WRITING_GUIDE.md**:

### Required Sections

```latex
\documentclass[letterpaper]{article}
\usepackage{aaai24}

\title{Logical Agents and Propositional Inference}
\author{Josh Manchester\\
University of Colorado Colorado Springs\\
josh.manchester@uccs.edu}

\begin{document}

\maketitle

\begin{abstract}
[Brief summary following writing guide: question-driven, results with numbers]
\end{abstract}

\section{Introduction}
[Question-driven opening: "How do knowledge-based agents reason..."]

\section{Part A: Propositional Logic}
\subsection{Knowledge-Based Agent Overview}
[Explain KB, Tell/Ask, inference, percepts → actions]

\subsection{Logical Equivalences}
[Truth tables for De Morgan and Contraposition]

\subsection{Model Checking}
[Show enumeration of models for P, Q, R]

\section{Part B: Horn Clause Inference}
\subsection{Forward Chaining Implementation}
[Algorithm description with reference to R&N Figure 7.15]

\subsection{Test Results}
[Generic KB and Wumpus KB results with traces]

\section{Part C: Wumpus World Agent}
\subsection{Horn Rules}
[List exact rules used]

\subsection{Agent Reasoning}
[Show 2-move trace with inference steps]

\subsection{World State}
[4×4 grid table]

\section{Part D: Resolution Entailment}
[CNF conversion and resolution trace]

\section{Conclusion}
[Summary of key findings]

\section*{AI Use Disclosure}
[Full transparency per CLAUDE.md]

\bibliographystyle{aaai24}
\bibliography{references}

\end{document}
```

---

## Timeline and Milestones

### Phase 1: Setup and Part A (Days 1-2)
- ✅ Create directory structure
- ✅ Implement `propositional_logic.py`
- ✅ Test equivalences and model checking
- ✅ Write Part A in report

### Phase 2: Part B Horn Inference (Days 3-4)
- ✅ Implement `horn_inference.py` (Forward Chaining)
- ✅ Create `knowledge_base.py` utilities
- ✅ Test on generic KB and Wumpus fragment
- ✅ Write Part B in report

### Phase 3: Part C Wumpus Agent (Days 5-6)
- ✅ Implement `wumpus_agent.py`
- ✅ Test 2-move reasoning
- ✅ Generate 4×4 grid output
- ✅ Write Part C in report

### Phase 4: Part D Extension (Day 7)
- ✅ Implement `resolution.py`
- ✅ Test on 2 queries
- ✅ Write Part D in report

### Phase 5: Testing and Quality (Day 8)
- ✅ Complete `test_all.py`
- ✅ Run pylint, achieve ≥ 9.0/10
- ✅ Create `run_experiments.py`
- ✅ Generate all output

### Phase 6: Writeup (Days 9-10)
- ✅ Complete AAAI-formatted report
- ✅ Add all figures, tables, traces
- ✅ Compile to PDF
- ✅ Final review

### Phase 7: Submission (Day 11)
- ✅ Create submission package
- ✅ Test code runs independently
- ✅ Submit to Canvas

---

## Key Algorithms and References

### Algorithm Sources

| Algorithm | Russell & Norvig | Lecture 8 |
|-----------|------------------|-----------|
| Model Checking | Section 7.5.1, Figure 7.10 | Part I, Slide 45 |
| Forward Chaining | Section 7.5.3, Figure 7.15 | Part III, Slide 80 |
| Backward Chaining | Section 7.5.3, Figure 7.16 | Part III, Slide 85 |
| Resolution | Section 7.5.2, Figure 7.12 | Part III, Slides 95-110 |
| Wumpus World | Section 7.7 | Part I, Slides 25-40 |

### Complexity Analysis

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| Model Checking | O(2^n) | O(n) | n = number of symbols |
| Forward Chaining | O(kn) | O(n) | k = max rule size, linear! |
| Backward Chaining | O(kn) | O(d) | d = proof depth |
| Resolution | Exponential worst | O(2^n) | Can be inefficient |

---

## Success Criteria

### Code Quality
- ✅ Pylint score ≥ 9.0/10
- ✅ All tests pass (100% success rate)
- ✅ No external logic libraries used
- ✅ Comprehensive comments with algorithm references
- ✅ Type hints on all functions
- ✅ README with clear run instructions

### Writeup Quality
- ✅ AAAI format with proper citations
- ✅ Josh's writing style (question-driven, parenthetical definitions)
- ✅ "According to" citation pattern
- ✅ All figures/tables with captions
- ✅ Truth tables clearly formatted
- ✅ Inference traces readable
- ✅ AI disclosure section

### Functionality
- ✅ Part A: Correct equivalence checks and model checking
- ✅ Part B: Working inference engine with traces
- ✅ Part C: Agent makes 2 safe moves with reasoning
- ✅ Part D: Resolution works or scaling study complete
- ✅ All output matches assignment requirements

---

## Questions and Clarifications

### Q: Forward Chaining vs Backward Chaining?
**A:** Choose Forward Chaining - easier to implement, linear time, easier to trace.

### Q: How to handle Wumpus World rules with disjunctions?
**A:** Convert to multiple Horn clauses or use simplified representation:
- `¬B(x,y) ⇒ ¬P(x+1,y)` (separate rule for each neighbor)
- Instead of: `¬B(x,y) ⇒ (¬P(x+1,y) ∧ ¬P(x-1,y) ∧ ...)`

### Q: What counts as "neutral" facts for scaling study?
**A:** Facts/rules that don't affect query results (e.g., `Z1, Z2, ..., Zk` with `Zi ⇒ Zi+1`)

### Q: CNF conversion - implement full algorithm?
**A:** For Part D resolution, implement basic CNF conversion (eliminate ⇒, move ¬ inward, distribute ∨)

---

**Next Step:** Start with directory structure and Part A implementation!
