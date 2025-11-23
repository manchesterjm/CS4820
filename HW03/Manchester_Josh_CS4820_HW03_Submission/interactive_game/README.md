# Wumpus World Interactive Game (Bonus Feature)

**Author:** Josh Manchester
**Institution:** University of Colorado Colorado Springs
**Course:** CS 4820/5820 - Artificial Intelligence

## Overview

This is a fully interactive implementation of the Wumpus World problem with a knowledge-based reasoning agent that uses propositional logic to survive and win.

**Note:** This is a bonus feature demonstrating the assignment concepts in an interactive environment. The required assignment code is in the `assignment_code/` directory.

## Requirements

- Python 3.7+
- **pygame-ce** (Pygame Community Edition)

```bash
pip install pygame-ce
```

## How to Run the Game

```bash
python wumpus_game_visual.py
```

## Game Features

### Agent Capabilities
- **Logical Reasoning**: Uses Horn clause inference and propositional logic
- **Wumpus Triangulation**: Confirms exact wumpus location from 2+ stench observations
- **Pit Detection**: Identifies confirmed pit locations via definite clause reasoning
- **Arrow Shooting**: Navigates to aligned position and shoots wumpus
- **Gold Finding**: Locates and retrieves gold
- **Path Planning**: Uses Manhattan distance heuristic for navigation
- **Oscillation Detection**: Prevents infinite loops with 6-move lookahead
- **Risk Calculation**: Takes calculated risks when needed, avoiding confirmed dangers

### Game Controls
- **SPACE**: Execute next step (manual mode)
- **A**: Toggle auto-play mode (agent runs automatically)
- **R**: Reset game with new random world
- **Q or ESC**: Quit game

### Visual Display
- **Left Panel**: 8×8 game board showing agent, pits, wumpus, gold, and visited cells
- **Right Panel**: AI reasoning display with KB facts, inferred knowledge, and decision-making process
- **Real-time Updates**: See agent's percepts, logical inferences, and chosen moves

### Victory Conditions
1. Find the gold
2. Confirm wumpus location via triangulation (2+ stenches)
3. Navigate to shooting position (aligned row or column)
4. Shoot and kill the wumpus
5. Return to start with gold (optional, but ideal)

## Testing the Agent

Three specialized test files verify agent correctness:

### Test Wumpus Triangulation
```bash
python test_triangulation.py
```
Runs 5 games with different seeds, verifies wumpus confirmation accuracy (expect 100%)

### Test Wumpus Shooting
```bash
python test_wumpus_shooting.py
```
Verifies agent finds gold, confirms wumpus, navigates to shooting position, and kills wumpus

### Test Parsing Bug Fix
```bash
python test_wumpus_bug.py
```
Reproduces and verifies fix for negative observation parsing (not_S_, not_W_ facts)

## Agent Performance

The agent successfully:
- **Confirms wumpus location**: 100% accuracy with 2+ stench observations
- **Avoids confirmed pits**: Never moves into logically confirmed dangerous cells
- **Kills wumpus**: Navigates to aligned shooting position and fires arrow
- **Wins games**: Typical victory in 16-70 steps depending on world layout
- **Explores efficiently**: Uses frontier-based global search

### Example Victory
```
Step 10: Wumpus confirmed at (1, 7) via triangulation
Step 14: Gold found at (3, 8) - GRABBED GOLD!
Step 15: Navigated to (2, 7) - aligned with wumpus (same column)
Step 16: Shot arrow from (2, 7) → killed wumpus at (1, 7)
Result: GAME WON in 16 steps!
```

## Files in This Directory

- `wumpus_game_visual.py` - Interactive pygame game
- `wumpus_agent.py` - Core agent logic (shared with assignment)
- `knowledge_base.py` - KB data structures (shared with assignment)
- `inference_engine_base.py` - Base inference engine (shared with assignment)
- `test_triangulation.py` - Triangulation verification tests
- `test_wumpus_bug.py` - Bug fix verification
- `test_wumpus_shooting.py` - Shooting logic verification

## Technical Implementation

**Key Algorithms:**
- Forward chaining on Horn clauses (O(n) inference)
- Model checking for propositional entailment
- Definite clause reasoning for pit/wumpus confirmation
- Set intersection for triangulation
- Manhattan distance pathfinding
- Oscillation detection with recent position tracking

**Based on:**
- Russell & Norvig Section 7.7: "Agents Based on Propositional Logic"
- CS 4820/5820 Lecture 8: Logical Agents (Parts I-IV)

## Game Log

Each game automatically saves a detailed log file:
- File format: `wumpus_game_log_YYYYMMDD_HHMMSS.txt`
- Contains: World setup, all agent steps, percepts, reasoning, and statistics
- Useful for debugging and analyzing agent performance
