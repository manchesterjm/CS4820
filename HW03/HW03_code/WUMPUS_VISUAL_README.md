# Wumpus World - Interactive Pygame Visualization

**Author:** Josh Manchester
**Course:** CS 4820/5820 - Artificial Intelligence
**Institution:** University of Colorado Colorado Springs

## Overview

This is an interactive pygame visualization of the Wumpus World agent with real-time AI reasoning display. The game shows both the visual game board and the logical reasoning process the AI uses to navigate safely.

## Features

### Two-Panel Display

**Left Panel: Game Board (600x600px)**
- 4×4 grid world with visual representation
- Agent (orange circle with face) that moves between squares
- Pits (black circles marked with "P")
- Wumpus (red triangle marked with "W")
- Visited squares are highlighted in light green
- Uncovered squares show percept letters:
  - **B** = Breeze (felt near pits)
  - **S** = Stench (smelled near Wumpus)
- Starting position (1,1) is marked with green background
- Smooth animation when agent moves between cells

**Right Panel: AI Reasoning Display (500x600px)**
- Current step number
- Agent's current position
- Percepts at current location (Breeze: YES/NO, Stench: YES/NO)
- Knowledge Base updates (facts being added)
- List of safe neighbors found by inference
- Decision-making reasoning
- Next move chosen by the agent

**Bottom Bar: Controls & Status (100px height)**
- Interactive controls display
- Game status (Ready / Auto-play / Game Over)
- Legend for symbols

## Requirements

### Python Dependencies

```bash
pip install pygame
```

### System Requirements
- Python 3.7+
- Display capable of 1100×700 window
- Keyboard for controls

## How to Run

### Quick Start

```bash
cd /home/user/CS4820/HW03/HW03_code
python wumpus_game_visual.py
```

The game window will open showing the initial state with the agent at position (1,1).

### Controls

| Key | Action |
|-----|--------|
| **SPACE** | Execute next step (agent reasons and moves) |
| **R** | Reset game to initial state |
| **A** | Toggle auto-play mode (agent moves automatically) |
| **Q** or **ESC** | Quit the game |

### Auto-Play Mode

Press **A** to enable auto-play mode. The agent will automatically execute steps with a 1.5-second delay between moves. This is useful for demonstrations or watching the AI navigate the world without manual input.

Press **A** again to pause auto-play and return to manual step-by-step mode.

## Game World Configuration

The default world is configured with:
- **Grid size:** 4×4
- **Agent start:** (1, 1) - bottom-left corner
- **Pits:** Located at (3, 1), (3, 3), (4, 4)
- **Wumpus:** Located at (1, 3)

### Coordinates
- Grid coordinates are 1-indexed
- (1,1) is the bottom-left corner
- (4,4) is the top-right corner
- Agent can move up, down, left, right (no diagonal moves)

## How the AI Works

The agent uses **Horn clause inference** with **forward chaining** to reason about the world:

1. **Sense Percepts:** Agent senses breeze and stench at current location
2. **Update KB:** Adds percept facts to knowledge base
   - Example: `not_B_1_1` (no breeze at 1,1)
3. **Apply Rules:** Wumpus World rules infer safety:
   - No breeze → No pits in adjacent cells
   - No stench → No Wumpus in adjacent cells
4. **Find Safe Neighbors:** Query KB for safe adjacent cells
5. **Choose Move:** Pick an unvisited safe neighbor
6. **Execute Move:** Move to chosen cell and repeat

### Knowledge Base Rules

**Breeze Rules:**
- `not_B_x_y => not_P_u_v` for each neighbor (u,v) of (x,y)
- "If there's no breeze at (x,y), then no pit at neighbor (u,v)"

**Stench Rules:**
- `not_S_x_y => not_W_u_v` for each neighbor (u,v) of (x,y)
- "If there's no stench at (x,y), then no Wumpus at neighbor (u,v)"

**Safety Rule:**
- Cell is safe if both `not_P_x_y` AND `not_W_x_y` are entailed

## Visualizations Explained

### Cell Colors
- **Light Gray:** Unvisited unknown cells
- **Light Green:** Visited cells (agent has been here)
- **Bright Green:** Starting position (1,1)

### Symbols on Board
- **Orange Circle (with face):** The AI agent
- **Black "P" in circle:** Pit (semi-transparent if not visited)
- **Red "W" in triangle:** Wumpus (semi-transparent if not visited)
- **Blue "B":** Breeze percept (shown in visited cells)
- **Blue "S":** Stench percept (shown in visited cells)

### Reasoning Panel Colors
- **Green text:** Safe/positive information
- **Red text:** Danger/negative information (percepts detected)
- **Blue text:** Next move decision
- **Black text:** Neutral information

## Code Architecture

The visualization integrates with the existing Wumpus World implementation:

```python
# Core components from wumpus_agent.py
WumpusWorld       # Environment simulator
WumpusAgent       # AI reasoning agent
AgentStep         # Immutable step record
Percept           # Immutable percept data

# From knowledge_base.py
WumpusKB          # Knowledge base with Horn clauses

# From horn_inference.py
forward_chaining  # Inference engine
```

### SOFA Principles Applied
- **Single Responsibility:** Visualization separate from game logic
- **Open/Closed:** Uses existing agent interface without modification
- **Functional:** Consumes immutable AgentStep and Percept records
- **Abstraction:** Clean interface to underlying AI system

## Customization

### Change World Configuration

Edit `reset_game()` method in `WumpusGameVisual` class:

```python
def reset_game(self):
    self.world = WumpusWorld(grid_size=GRID_SIZE)

    # Add custom pits
    self.world.add_pit(2, 2)
    self.world.add_pit(4, 1)

    # Add Wumpus
    self.world.add_wumpus(3, 3)

    # ... rest of setup
```

### Change Animation Speed

Modify `anim_speed` in `__init__`:

```python
self.anim_speed = 0.05  # Higher = faster animation (0.01 to 0.2 recommended)
```

### Change Auto-Play Delay

Modify `auto_play_delay` in `__init__`:

```python
self.auto_play_delay = 1.5  # Seconds between auto steps
```

### Change Grid Size

Modify `GRID_SIZE` constant at top of file:

```python
GRID_SIZE = 5  # For a 5×5 world
```

Note: Larger grids will make cells smaller. Consider adjusting `BOARD_SIZE` for better visibility.

### Change Colors

All colors are defined as constants at the top of the file. Example:

```python
AGENT_COLOR = (255, 165, 0)  # Orange
PIT_COLOR = (0, 0, 0)         # Black
WUMPUS_COLOR = (220, 50, 50)  # Red
```

## Example Play Session

1. **Initial State:**
   - Agent at (1,1)
   - No percepts (safe starting position)
   - KB knows: `not_B_1_1`, `not_S_1_1`

2. **Press SPACE - Step 1:**
   - Agent infers (2,1) and (1,2) are safe
   - Chooses to move to (2,1)
   - Smooth animation shows agent moving

3. **Step 2 (at position 2,1):**
   - Agent senses: Breeze=NO, Stench=NO
   - KB adds: `not_B_2_1`, `not_S_2_1`
   - Infers more safe neighbors
   - Continues exploration

4. **Game Over:**
   - Agent reaches a position with no safe unvisited neighbors
   - Status bar shows "GAME OVER"
   - Press R to reset and try again

## Troubleshooting

### "ModuleNotFoundError: No module named 'pygame'"

Install pygame:
```bash
pip install pygame
```

### "No video device" or display errors

If running on a server without display:
```bash
# Use virtual display (Linux)
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
python wumpus_game_visual.py
```

Or run on a local machine with a display.

### Window doesn't appear

- Check if another pygame window is open
- Verify display settings
- Try running from terminal (not IDE) for better display handling

### Agent moves too fast/slow

- Adjust `auto_play_delay` in code
- Use manual mode (turn off auto-play) and press SPACE at your own pace

### Cells are too small on large displays

Modify constants:
```python
BOARD_SIZE = 800  # Increase from 600
CELL_SIZE = BOARD_SIZE // GRID_SIZE
```

## Integration with Assignment

This visualization demonstrates all concepts from HW03 Part C:

- ✅ Knowledge-based agent reasoning
- ✅ Horn clause inference (forward chaining)
- ✅ Wumpus World environment
- ✅ Percept-based decision making
- ✅ Safety inference from logical rules
- ✅ Step-by-step execution with reasoning trace

The visualization enhances the text-based output by:
- Showing spatial relationships visually
- Animating agent movement
- Displaying real-time reasoning
- Making the AI's logic transparent and understandable

## Educational Value

This tool is excellent for:
- **Teaching:** Show students how knowledge-based agents work
- **Debugging:** Visualize why agent makes certain decisions
- **Demonstrations:** Present AI reasoning in an engaging way
- **Learning:** Understand propositional logic inference visually

## Future Enhancements (Ideas)

- [ ] Gold collection (agent searches for gold)
- [ ] Arrow shooting (agent can kill Wumpus)
- [ ] Score tracking
- [ ] Multiple difficulty levels
- [ ] Save/load game states
- [ ] Export reasoning trace to file
- [ ] Comparison mode (show different inference strategies)
- [ ] Interactive cell clicking to add obstacles
- [ ] Multiple agents competing

## References

- Russell & Norvig, "Artificial Intelligence: A Modern Approach", Section 7.7
- CS 4820/5820 Lecture 8: Logical Agents
- HW03 Implementation: `wumpus_agent.py`, `knowledge_base.py`, `horn_inference.py`

## Credits

**Implementation:** Josh Manchester
**AI Assistant:** Claude Code (Sonnet 4.5)
**Course:** CS 4820/5820 - Artificial Intelligence
**Instructor:** Professor Adham Atyabi
**Institution:** University of Colorado Colorado Springs

---

**Generated:** November 19, 2025
**Version:** 1.0
