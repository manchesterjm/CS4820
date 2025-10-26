# CS 4820/5820 Homework 1 - Problem Solving and Search

**Author:** Josh Manchester  
**Institution:** University of Colorado Colorado Springs  
**Email:** josh.manchester@uccs.edu

## Requirements

- Python 3.7 or higher
- No external libraries required (uses standard library only)
- PowerShell (for automated batch execution on Windows)

## File Structure

```
HW01_Code/
├── n_puzzle_BFS.py          # Breadth-First Search for n-puzzle
├── n_puzzle_Depth_Limited_DFS.py  # Depth-Limited DFS for n-puzzle
├── n_puzzle_IDS.py          # Iterative Deepening Search for n-puzzle
├── n_puzzle_BDS.py          # Bidirectional Search for n-puzzle
├── n_puzzle_ASTAR.py        # A* Search (misplaced vs Manhattan heuristics)
├── n_queens_SA.py           # Simulated Annealing for n-queens
├── n_queens_GA.py           # Genetic Algorithm for n-queens
├── run_all.ps1              # PowerShell script to run all programs
└── README.md                # This file
```

## Quick Start

### Option 1: Run All Programs (Automated)

The `run_all.ps1` script runs all programs sequentially and saves output to a log file.

**On Windows PowerShell:**
```powershell
.\run_all.ps1
```

**What it does:**
- Runs each Python program in order
- Captures all console output
- Saves results to `HW01_runlog.txt` with timestamp
- Shows progress as it runs

**If you get a permissions error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run_all.ps1
```

### Option 2: Run Individual Programs

Each program can be run independently:

```bash
# n-Puzzle Programs (3 trials each on 8-puzzle and 15-puzzle)
python n_puzzle_BFS.py
python n_puzzle_Depth_Limited_DFS.py
python n_puzzle_IDS.py
python n_puzzle_BDS.py

# A* Search (compares two heuristics on same puzzles)
python n_puzzle_ASTAR.py

# n-Queens Programs
python n_queens_SA.py    # Simulated Annealing (n=4 and n=8)
python n_queens_GA.py    # Genetic Algorithm (tests configs, then n=8)
```

## What Each Program Does

### n-Puzzle Programs
All n-puzzle programs run **3 successful trials** on both 8-puzzle (3×3) and 15-puzzle (4×4):

- **n_puzzle_BFS.py:** Breadth-First Search - guarantees optimal solutions
- **n_puzzle_Depth_Limited_DFS.py:** Depth-Limited DFS - finds solutions within depth limit
- **n_puzzle_IDS.py:** Iterative Deepening Search - optimal with low memory usage
- **n_puzzle_BDS.py:** Bidirectional Search - meets in the middle for efficiency
- **n_puzzle_ASTAR.py:** A* Search comparing two heuristics on the **same** puzzle instances

Each trial uses a **randomly scrambled** starting position.

### n-Queens Programs

- **n_queens_SA.py:** Simulated Annealing
  - Runs 3 trials on n=4 with T₀=1.0, cooling=0.995
  - Runs 3 trials on n=8 with T₀=2.0, cooling=0.998
  
- **n_queens_GA.py:** Genetic Algorithm
  - **Phase 1:** Tests 3 different parameter configurations on n=4
  - **Phase 2:** Uses best configuration for 3 trials on n=8

## Modifying Randomization

All programs use Python's `random` module. To get **reproducible results** (same puzzles every run), uncomment the `random.seed(0)` line at the bottom of each file.

### For n-Puzzle Programs (BFS, DFS, IDS, BDS, A*)

**Change scrambling difficulty:**

Look for these lines near the top of each file:

```python
# Example from n_puzzle_BFS.py
configs = [(3, 40), (4, 40)]  # Format: (board_size, scramble_steps)
```

- First number: board dimension (3 for 8-puzzle, 4 for 15-puzzle)
- Second number: how many random moves from goal state

**Examples:**
```python
configs = [(3, 20), (4, 20)]  # Easier puzzles (20 steps)
configs = [(3, 60), (4, 30)]  # Harder 8-puzzle, moderate 15-puzzle
configs = [(3, 100), (4, 50)] # Very hard puzzles (may take longer!)
```

**Enable reproducible results:**

At the bottom of each file, find:
```python
if __name__ == "__main__":
    # random.seed(0)  # Uncomment for reproducible results
    run_trials_auto()
```

Change to:
```python
if __name__ == "__main__":
    random.seed(0)  # Now uses same random puzzles every run
    run_trials_auto()
```

### For n-Queens Programs

**Simulated Annealing (n_queens_SA.py):**

Modify parameters near the top:
```python
# Current settings:
configs = [
    (4, 1.0, 0.995, 50000),    # (n, T0, cooling, max_iters)
    (8, 2.0, 0.998, 200000),
]
```

- `T0`: Initial temperature (higher = more exploration)
- `cooling`: Cooling rate (closer to 1.0 = slower cooling)
- `max_iters`: Maximum iterations before giving up

**Genetic Algorithm (n_queens_GA.py):**

Modify configurations being tested:
```python
# Three configurations to test: (pop_size, cx_rate, mut_rate, max_gens)
configs = [
    (40, 0.9, 0.10, 200),  # Config A: Small pop, moderate mutation
    (80, 0.9, 0.05, 300),  # Config B: Large pop, low mutation
    (60, 0.8, 0.15, 300),  # Config C: Medium pop, high mutation
]
```

- `pop_size`: Population size (20-100 typical)
- `cx_rate`: Crossover rate (0.6-0.9 typical)
- `mut_rate`: Mutation rate (0.01-0.2 typical)
- `max_gens`: Maximum generations

**Enable reproducible results for n-Queens:**
```python
if __name__ == "__main__":
    random.seed(0)  # Uncomment this line
    # ... rest of code
```

## Performance Notes

### Expected Runtimes

- **BFS, IDS, BDS:** < 1 second for 8-puzzle, < 1 minute for 15-puzzle
- **Depth-Limited DFS:** Variable, can take 1-60 seconds for 15-puzzle
- **A*:** Very fast, usually < 0.01 seconds for 8-puzzle
- **n-Queens SA:** < 0.01 seconds for n=4 and n=8
- **n-Queens GA:** < 0.1 seconds for n=8

### Safety Limits

Programs have built-in safeguards:

- **Time limits:** Most programs stop after 60-120 seconds
- **Node limits:** DFS stops at 60M nodes for 15-puzzle
- **Retry logic:** If a puzzle is too hard, programs retry with easier instances

You can adjust these limits by modifying constants at the top of each file:
```python
MAX_TIME_SEC = 60        # Time limit per search
MAX_NODES = 60000000     # Node expansion limit
```

## Understanding the Output

### n-Puzzle Programs Output Format

```
trial 1:
BFS on 3x3
start:
0 5 2
1 4 3
7 8 6
moves: 6            # Solution length (number of moves)
expanded: 61        # Nodes examined
time_s: 0.0001      # Execution time in seconds
```

### n-Queens Programs Output Format

**Simulated Annealing:**
```
SA on n=8
board:
. . . . . . Q .     # Q = queen, . = empty
. . Q . . . . .
...
conflicts: 0        # 0 = solution found
expanded: 709       # Number of neighbor evaluations
time_s: 0.0027
status: ok
```

**Genetic Algorithm:**
```
GA on n=8 attempt 1
. . Q . . . . .
...
conflicts: 0
gens: 214           # Generations needed
evals: 12900        # Total fitness evaluations
time_s: 0.0846
status: ok
```

## Troubleshooting

**"Python not found"**
- Make sure Python 3.7+ is installed and in your PATH
- Try `python3` instead of `python` on Mac/Linux

**"Cannot run script" (PowerShell)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Programs taking too long:**
- Reduce scramble steps for n-puzzle programs
- Reduce max_iterations for n-queens programs
- The programs will eventually timeout (60-120 seconds) automatically

**"No module named..." error:**
- These programs only use Python's standard library
- Make sure you're using Python 3.7 or higher

## Code Documentation

All Python files are **extensively commented** with:
- Algorithm explanations
- Time and space complexity analysis
- Design decisions and tradeoffs
- Line-by-line code documentation

You can read the source code to understand how each algorithm works!

## Assignment Details

This code implements solutions for CS 4820/5820 Homework 1, covering:

- **Part A:** Intelligent Agents (theoretical, no code)
- **Part B:** n-Puzzle with uninformed search (BFS, DFS, IDS, BDS)
- **Part C:** n-Queens with local search (SA, GA)
- **Part D:** n-Puzzle with informed search (A* with heuristics)