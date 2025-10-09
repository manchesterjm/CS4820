# CS 4820/5820 Homework 1 - Problem Solving and Search

**Author:** Josh Manchester  
**Institution:** University of Colorado Colorado Springs  
**Email:** josh.manchester@uccs.edu

## Requirements

- Python 3.7+
- No external libraries required (standard library only)

## File Structure

```
HW01_Code/
├── n_puzzle_BFS.py          # Breadth-First Search
├── n_puzzle_DFS.py          # Depth-First Search
├── n_puzzle_IDS.py          # Iterative Deepening Search
├── n_puzzle_BDS.py          # Bidirectional Search
├── n_queens_SA.py           # Simulated Annealing
├── n_queens_GA.py           # Genetic Algorithm
├── n_puzzle_ASTAR.py        # A* (misplaced tiles vs Manhattan distance)
├── run_all.ps1              # Batch script to run all programs
└── README.md
```

## How to Run

### Run All Programs

```powershell
.\run_all.ps1
```

Output is saved to `HW01_runlog.txt`

### Run Individual Programs

```powershell
python n_puzzle_BFS.py
python n_puzzle_DFS.py
python n_puzzle_IDS.py
python n_puzzle_BDS.py
python n_queens_SA.py
python n_queens_GA.py
python n_puzzle_ASTAR.py
```

## What Each Program Does

- **n-Puzzle programs:** Run 3 trials each on 8-puzzle and 15-puzzle with random initial states
- **n_queens_SA.py:** Runs 3 trials each for n=4 and n=8
- **n_queens_GA.py:** Tests 3 parameter configurations on n=4, then uses best config for n=8 (3 trials)
- **n_puzzle_ASTAR.py:** Compares two heuristics on 3 identical puzzle instances

All programs are fully automated and generate their own test cases.