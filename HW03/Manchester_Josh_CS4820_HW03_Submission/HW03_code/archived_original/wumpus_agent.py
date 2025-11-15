"""
Wumpus World reasoning agent using propositional logic.

This module implements a knowledge-based agent that uses Horn clause inference
to navigate safely through the Wumpus World.

WUMPUS WORLD OVERVIEW:
======================
The Wumpus World is a classic AI problem environment (Russell & Norvig Section 7.7).

Environment:
- Grid world (typically 4x4)
- Agent starts at (1,1)
- Contains: Gold, Wumpus (monster), Pits
- Agent goal: Find gold, avoid pits and Wumpus

Percepts:
- Breeze: Felt in squares adjacent to a pit
- Stench: Smelled in squares adjacent to the Wumpus
- Glitter: Visible when gold is in same square
- Bump: Felt when agent walks into wall
- Scream: Heard when Wumpus is killed

Agent knowledge:
- Can infer safe cells from percepts
- Uses logic to determine where pits/Wumpus cannot be
- Makes reasoned decisions about where to move

Based on Russell & Norvig:
- Section 7.2: The Wumpus World
- Section 7.7: Agents Based on Propositional Logic
- Figure 7.20: Wumpus World Inference Rules

Lecture References:
- Lecture 8 Part I: Wumpus World Introduction
- Slides 25-40: Wumpus World Reasoning

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Set, Tuple, Optional, Dict
from knowledge_base import WumpusKB
from horn_inference import forward_chaining


class WumpusWorld:
    """
    Wumpus World environment for testing agent.

    This is a simple Wumpus World simulator that provides percepts
    based on agent location.

    Attributes:
        grid_size: Dimension of square grid
        pits: Set of (x, y) coordinates with pits
        wumpus: (x, y) coordinate of Wumpus
        gold: (x, y) coordinate of gold (optional)
    """

    def __init__(self, grid_size: int = 4):
        """
        Initialize Wumpus World.

        Args:
            grid_size: Grid dimension (default 4 for 4x4)
        """
        self.grid_size = grid_size
        self.pits: Set[Tuple[int, int]] = set()
        self.wumpus: Optional[Tuple[int, int]] = None
        self.gold: Optional[Tuple[int, int]] = None

    def add_pit(self, x: int, y: int) -> None:
        """Add a pit at location (x, y)."""
        self.pits.add((x, y))

    def add_wumpus(self, x: int, y: int) -> None:
        """Add the Wumpus at location (x, y)."""
        self.wumpus = (x, y)

    def get_percepts(self, x: int, y: int) -> Dict[str, bool]:
        """
        Get percepts at location (x, y).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Dictionary with 'breeze' and 'stench' booleans
        """
        percepts = {'breeze': False, 'stench': False}

        # Check for breeze (adjacent to pit)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.pits:
                percepts['breeze'] = True
                break

        # Check for stench (adjacent to Wumpus)
        if self.wumpus:
            wx, wy = self.wumpus
            if abs(x - wx) + abs(y - wy) == 1:  # Manhattan distance = 1
                percepts['stench'] = True

        return percepts


class WumpusAgent:
    """
    Knowledge-based agent for Wumpus World.

    Uses Horn clause inference to reason about safe moves based on percepts.

    Based on Russell & Norvig Section 7.7, Lecture 8 Part I Slides 25-40.

    The agent maintains a knowledge base and uses forward chaining to infer
    which neighboring cells are safe to visit.

    Attributes:
        kb: Wumpus knowledge base (Horn clauses)
        position: Current (x, y) position
        visited: Set of visited locations
        grid_size: Grid dimension
        move_log: List of (position, percepts, inferences, move) tuples
    """

    def __init__(self, grid_size: int = 4):
        """
        Initialize Wumpus agent at (1,1).

        Args:
            grid_size: Grid dimension (default 4 for 4x4)
        """
        self.kb = WumpusKB(grid_size=grid_size)
        self.kb.add_wumpus_rules()  # Add inference rules
        self.position = (1, 1)
        self.visited: Set[Tuple[int, int]] = {(1, 1)}
        self.grid_size = grid_size
        self.move_log: List[Dict] = []

        # Mark starting position as safe
        self.kb.mark_safe(1, 1)

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighbors of (x, y) within grid boundaries.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of (x, y) tuples for adjacent cells
        """
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def sense(self, percepts: Dict[str, bool]) -> None:
        """
        Add percepts to knowledge base.

        Args:
            percepts: Dictionary with 'breeze' and 'stench' keys
        """
        x, y = self.position
        self.kb.add_percept(x, y, percepts['breeze'], percepts['stench'])

    def infer_safe_neighbors(self, verbose: bool = True) -> List[Tuple[int, int]]:
        """
        Use forward chaining to determine which neighbors are safe.

        A neighbor (x, y) is safe if we can infer both:
        - not_P_x_y (no pit at x,y)
        - not_W_x_y (no Wumpus at x,y)

        Args:
            verbose: If True, print inference details

        Returns:
            List of (x, y) coordinates that are inferred safe
        """
        safe_neighbors = []
        neighbors = self.get_neighbors(*self.position)

        if verbose:
            print(f"  Checking {len(neighbors)} neighbors for safety...")

        for nx, ny in neighbors:
            # Query: Is this neighbor safe?
            # Safe means both not_P_x_y and not_W_x_y are entailed

            # Check for no pit
            no_pit_query = f"not_P_{nx}_{ny}"
            no_pit, _, _ = forward_chaining(self.kb, no_pit_query, verbose=False)

            # Check for no Wumpus
            no_wumpus_query = f"not_W_{nx}_{ny}"
            no_wumpus, _, _ = forward_chaining(self.kb, no_wumpus_query,
                                               verbose=False)

            if no_pit and no_wumpus:
                safe_neighbors.append((nx, ny))
                if verbose:
                    print(f"    ({nx}, {ny}): SAFE " +
                          f"(inferred {no_pit_query} and {no_wumpus_query})")
            else:
                if verbose:
                    status = []
                    if not no_pit:
                        status.append("possible pit")
                    if not no_wumpus:
                        status.append("possible Wumpus")
                    print(f"    ({nx}, {ny}): UNSAFE ({', '.join(status)})")

        return safe_neighbors

    def choose_move(self, safe_neighbors: List[Tuple[int, int]],
                    verbose: bool = True) -> Optional[Tuple[int, int]]:
        """
        Choose an unvisited safe neighbor.

        Args:
            safe_neighbors: List of cells inferred to be safe
            verbose: If True, print reasoning

        Returns:
            Next move (x, y) or None if no safe unvisited neighbors
        """
        # Filter to unvisited cells
        unvisited_safe = [loc for loc in safe_neighbors if loc not in self.visited]

        if unvisited_safe:
            # Choose first unvisited safe neighbor
            chosen = unvisited_safe[0]
            if verbose:
                print(f"  Choosing move to {chosen} (unvisited, safe)")
            return chosen
        elif safe_neighbors:
            # All safe neighbors are visited - could backtrack
            if verbose:
                print("  All safe neighbors already visited")
                print(f"  Safe but visited: {safe_neighbors}")
            return None
        else:
            if verbose:
                print("  No safe neighbors found!")
            return None

    def make_move(self, new_position: Tuple[int, int]) -> None:
        """
        Update position and mark cell as visited.

        Args:
            new_position: (x, y) coordinates to move to
        """
        self.position = new_position
        self.visited.add(new_position)
        # Mark this position as safe (we're alive!)
        self.kb.mark_safe(*new_position)

    def run_two_steps(self, world: WumpusWorld, verbose: bool = True) -> None:
        """
        Execute exactly two moves with reasoning.

        For each step:
            1. Sense percepts at current cell
            2. Add percepts to KB
            3. Run inference to find safe neighbors
            4. Choose and make move to safe unvisited neighbor
            5. Log the reasoning process

        Args:
            world: Wumpus World environment
            verbose: If True, print detailed reasoning trace
        """
        if verbose:
            print("=" * 80)
            print("Wumpus World Agent - Two-Step Reasoning")
            print("=" * 80)
            print()
            print(f"Starting position: {self.position}")
            print(f"Grid size: {self.grid_size}x{self.grid_size}")
            print()

        for step in range(1, 3):  # Steps 1 and 2
            if verbose:
                print(f"{'='*40} STEP {step} {'='*40}")
                print()
                print(f"Current position: {self.position}")

            # 1. Sense percepts
            percepts = world.get_percepts(*self.position)
            if verbose:
                breeze_str = "YES" if percepts['breeze'] else "NO"
                stench_str = "YES" if percepts['stench'] else "NO"
                print(f"Percepts: Breeze={breeze_str}, Stench={stench_str}")
                print()

            # 2. Add percepts to KB
            self.sense(percepts)
            if verbose:
                print("KB additions:")
                x, y = self.position
                if percepts['breeze']:
                    print(f"  - B_{x}_{y} (breeze at {self.position})")
                else:
                    print(f"  - not_B_{x}_{y} (no breeze at {self.position})")
                if percepts['stench']:
                    print(f"  - S_{x}_{y} (stench at {self.position})")
                else:
                    print(f"  - not_S_{x}_{y} (no stench at {self.position})")
                print()

            # 3. Run inference to find safe neighbors
            if verbose:
                print("Running inference...")
            safe_neighbors = self.infer_safe_neighbors(verbose=verbose)
            if verbose:
                print()

            # 4. Choose move
            if verbose:
                print("Decision:")
            next_move = self.choose_move(safe_neighbors, verbose=verbose)

            # Log this step
            self.move_log.append({
                'step': step,
                'position': self.position,
                'percepts': percepts,
                'safe_neighbors': safe_neighbors,
                'move': next_move
            })

            if next_move:
                # 5. Make the move
                if verbose:
                    print()
                    print(f"MOVING: {self.position} -> {next_move}")
                self.make_move(next_move)
                if verbose:
                    print(f"New position: {self.position}")
                    print()
            else:
                if verbose:
                    print()
                    print("STOPPING: No safe unvisited neighbors available")
                    print()
                break

    def print_world_state(self) -> None:
        """
        Print a 4x4 grid showing visited/safe/unknown cells.

        Legend:
            V = Visited
            S = Safe (entailed, not visited)
            ? = Unknown
        """
        print("=" * 80)
        print("World State After Agent Moves")
        print("=" * 80)
        print()

        # Determine safe cells from KB
        safe_cells = set(self.visited)  # Visited cells are safe
        for x in range(1, self.grid_size + 1):
            for y in range(1, self.grid_size + 1):
                if (x, y) not in self.visited:
                    # Check if this cell is inferred safe
                    no_pit, _, _ = forward_chaining(self.kb, f"not_P_{x}_{y}",
                                                   verbose=False)
                    no_wumpus, _, _ = forward_chaining(self.kb, f"not_W_{x}_{y}",
                                                       verbose=False)
                    if no_pit and no_wumpus:
                        safe_cells.add((x, y))

        # Print grid (Y-axis inverted to match typical grid display)
        print("    " + "   ".join(str(x) for x in range(1, self.grid_size + 1)))
        for y in range(self.grid_size, 0, -1):  # Top to bottom
            row = f"{y}  "
            for x in range(1, self.grid_size + 1):
                if (x, y) in self.visited:
                    cell = " V "
                elif (x, y) in safe_cells:
                    cell = " S "
                else:
                    cell = " ? "
                row += cell
            print(row)

        print()
        print("Legend:")
        print("  V = Visited")
        print("  S = Safe (inferred, not visited)")
        print("  ? = Unknown")
        print()


def test_wumpus_agent() -> None:
    """
    Test Wumpus agent on a simple scenario.

    World configuration:
        - 4x4 grid
        - Pit at (3, 1)
        - Wumpus at (1, 3)
        - Agent starts at (1, 1)

    Expected behavior:
        - Step 1: At (1,1), no breeze, no stench -> Move to (2,1) or (1,2)
        - Step 2: Sense percepts, infer safety, make second move
    """
    print("=" * 80)
    print("Part C: Wumpus World Agent Test")
    print("=" * 80)
    print()

    # Create world
    world = WumpusWorld(grid_size=4)
    world.add_pit(3, 1)       # Pit at (3,1)
    world.add_wumpus(1, 3)    # Wumpus at (1,3)

    print("World configuration:")
    print("  - Pit at (3, 1)")
    print("  - Wumpus at (1, 3)")
    print()

    # Create agent
    agent = WumpusAgent(grid_size=4)

    # Run two steps
    agent.run_two_steps(world, verbose=True)

    # Print world state
    agent.print_world_state()

    print("=" * 80)
    print("Part C: Wumpus World Agent - COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    """
    Run Wumpus agent demonstration.
    """
    test_wumpus_agent()
