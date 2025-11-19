"""
Refactored Wumpus World reasoning agent using SOFA principles.

SOFA Improvements:
- Single Responsibility: Separated world simulation, agent logic, and presentation
- Open/Closed: Extensible through strategy pattern for movement policies
- Functional: Immutable percept records, pure helper functions
- Abstraction: Clear interfaces for world, agent, and decision-making

Based on Russell & Norvig Section 7.7: Agents Based on Propositional Logic

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from knowledge_base import WumpusKB
from horn_inference import forward_chaining
import time


# ============================================================================
# SOFA: Functional - Immutable percept and move records
# ============================================================================

@dataclass(frozen=True)
class Percept:
    """
    Immutable percept snapshot.

    SOFA: Functional - Frozen dataclass for immutability
    """
    position: Tuple[int, int]
    breeze: bool
    stench: bool

    def __str__(self) -> str:
        """String representation."""
        breeze_str = "YES" if self.breeze else "NO"
        stench_str = "YES" if self.stench else "NO"
        return f"Percept({self.position}): Breeze={breeze_str}, Stench={stench_str}"


@dataclass(frozen=True)
class AgentStep:
    """
    Immutable record of one agent step.

    SOFA: Functional - Complete snapshot of reasoning process
    """
    step_number: int
    position: Tuple[int, int]
    percept: Percept
    safe_neighbors: Tuple[Tuple[int, int], ...]
    chosen_move: Optional[Tuple[int, int]]
    reasoning: str

    def __str__(self) -> str:
        """String representation."""
        move_str = str(self.chosen_move) if self.chosen_move else "None"
        return (f"Step {self.step_number}: {self.position} -> {move_str} "
                f"(safe options: {len(self.safe_neighbors)})")


# ============================================================================
# SOFA: Single Responsibility - World simulation only
# ============================================================================

class WumpusWorld:
    """
    Wumpus World environment simulator.

    SOFA: Single Responsibility - Only simulates world, no agent logic

    The Wumpus World is a classic AI environment (Russell & Norvig Section 7.2):
    - Grid world (typically 4x4)
    - Contains: Pits, Wumpus (monster)
    - Percepts: Breeze (near pit), Stench (near Wumpus)
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

    def add_pit(self, x: int, y: int) -> None:
        """Add a pit at location (x, y)."""
        self.pits.add((x, y))

    def add_wumpus(self, x: int, y: int) -> None:
        """Add the Wumpus at location (x, y)."""
        self.wumpus = (x, y)

    def sense_percepts(self, x: int, y: int) -> Percept:
        """
        Get immutable percept at location (x, y).

        SOFA: Functional - Returns immutable Percept object

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Immutable Percept with breeze and stench information
        """
        breeze = self._has_adjacent_pit(x, y)
        stench = self._has_adjacent_wumpus(x, y)
        return Percept(position=(x, y), breeze=breeze, stench=stench)

    def _has_adjacent_pit(self, x: int, y: int) -> bool:
        """Pure function: Check if position is adjacent to pit."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (x + dx, y + dy) in self.pits:
                return True
        return False

    def _has_adjacent_wumpus(self, x: int, y: int) -> bool:
        """Pure function: Check if position is adjacent to Wumpus."""
        if self.wumpus:
            wx, wy = self.wumpus
            return abs(x - wx) + abs(y - wy) == 1
        return False


# ============================================================================
# SOFA: Functional - Pure helper functions
# ============================================================================

def get_valid_neighbors(position: Tuple[int, int], grid_size: int) -> List[Tuple[int, int]]:
    """
    Pure function: Get valid neighbors within grid boundaries.

    SOFA: Functional - No side effects, deterministic

    Args:
        position: Current (x, y) position
        grid_size: Grid dimension

    Returns:
        List of valid neighbor coordinates
    """
    x, y = position
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= grid_size and 1 <= ny <= grid_size:
            neighbors.append((nx, ny))
    return neighbors


def is_safe_cell(kb: WumpusKB, x: int, y: int) -> bool:
    """
    Pure function: Check if cell (x, y) is safe using forward chaining.

    SOFA: Functional - Pure query, no KB modification

    Args:
        kb: Knowledge base
        x: X coordinate
        y: Y coordinate

    Returns:
        True if both not_P_x_y and not_W_x_y are entailed
    """
    no_pit_query = f"not_P_{x}_{y}"
    no_wumpus_query = f"not_W_{x}_{y}"

    no_pit, _, _ = forward_chaining(kb, no_pit_query, verbose=False)
    no_wumpus, _, _ = forward_chaining(kb, no_wumpus_query, verbose=False)

    return no_pit and no_wumpus


# ============================================================================
# SOFA: Abstraction - Strategy interface for movement policies
# ============================================================================

class MovementStrategy(ABC):
    """
    Abstract strategy for choosing moves.

    SOFA:
    - Abstraction: Defines interface without implementation
    - Open/Closed: New strategies extend without modifying agent
    """

    @abstractmethod
    def choose_move(
        self,
        current_position: Tuple[int, int],
        safe_neighbors: List[Tuple[int, int]],
        visited: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Choose next move from safe neighbors.

        Args:
            current_position: Current location
            safe_neighbors: List of safe neighboring cells
            visited: Set of visited cells

        Returns:
            Chosen move or None
        """
        pass


class UnvisitedFirstStrategy(MovementStrategy):
    """
    Strategy: Explore unvisited cells, revisit when needed to reach new areas.

    SOFA: Open/Closed - Implements strategy interface

    Prefers unvisited safe neighbors, but will move to visited safe neighbors
    to continue exploration. Only stops when no safe neighbors exist.
    """

    def __init__(self):
        """Initialize with visit frequency tracking."""
        self.visit_count: Dict[Tuple[int, int], int] = {}

    def choose_move(
        self,
        current_position: Tuple[int, int],
        safe_neighbors: List[Tuple[int, int]],
        visited: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Choose move: unvisited cell > least visited cell > None.

        1. Prefer unvisited safe neighbors (exploration)
        2. If all visited, choose least-visited safe neighbor (avoid loops)
        3. If no safe neighbors, return None (stuck)
        """
        if not safe_neighbors:
            return None

        unvisited_safe = [loc for loc in safe_neighbors if loc not in visited]

        # Prefer unvisited cells
        if unvisited_safe:
            return unvisited_safe[0]

        # All neighbors visited - choose least visited to avoid tight loops
        visited_safe = [loc for loc in safe_neighbors if loc in visited]
        least_visited = min(visited_safe, key=lambda loc: self.visit_count.get(loc, 0))

        # Track visit count
        self.visit_count[least_visited] = self.visit_count.get(least_visited, 0) + 1

        # Only revisit if we haven't been there too many times
        if self.visit_count[least_visited] <= 3:
            return least_visited

        return None  # Visited this area too many times, give up


# ============================================================================
# SOFA: Single Responsibility - Agent reasoning (no I/O)
# ============================================================================

@dataclass
class WumpusAgentState:
    """
    Mutable agent state during execution.

    SOFA: Single Responsibility - Only tracks agent state
    Note: Mutable for performance, converted to immutable trace at end
    """
    kb: WumpusKB
    position: Tuple[int, int] = (1, 1)
    visited: Set[Tuple[int, int]] = field(default_factory=lambda: {(1, 1)})
    steps: List[AgentStep] = field(default_factory=list)
    grid_size: int = 4
    previous_position: Optional[Tuple[int, int]] = None  # Track where we came from


class WumpusAgent:
    """
    Knowledge-based Wumpus World agent.

    SOFA:
    - Single Responsibility: Only agent reasoning, no world simulation or printing
    - Open/Closed: Uses strategy pattern for movement
    - Abstraction: Hides KB details from clients

    Based on Russell & Norvig Section 7.7, Lecture 8 Part I Slides 25-40.
    """

    def __init__(self, grid_size: int = 4, strategy: MovementStrategy = None):
        """
        Initialize agent with strategy.

        Args:
            grid_size: Grid dimension
            strategy: Movement strategy (default: UnvisitedFirstStrategy)
        """
        self._state = WumpusAgentState(
            kb=WumpusKB(grid_size=grid_size),
            grid_size=grid_size
        )
        self._state.kb.add_wumpus_rules()
        self._state.kb.mark_safe(1, 1)
        self._strategy = strategy or UnvisitedFirstStrategy()

    def execute_step(self, world: WumpusWorld, step_number: int) -> AgentStep:
        """
        Execute one reasoning step.

        SOFA: Functional - Returns immutable AgentStep

        Args:
            world: Wumpus World environment
            step_number: Current step number

        Returns:
            Immutable record of this step
        """
        # 1. Sense percepts
        percept = world.sense_percepts(*self._state.position)

        # 2. Add percepts to KB
        self._state.kb.add_percept(
            percept.position[0],
            percept.position[1],
            percept.breeze,
            percept.stench
        )

        # 3. Find safe neighbors
        neighbors = get_valid_neighbors(self._state.position, self._state.grid_size)
        safe_neighbors = [n for n in neighbors if is_safe_cell(self._state.kb, n[0], n[1])]

        # 4. Choose move using strategy
        chosen_move = self._strategy.choose_move(
            self._state.position,
            safe_neighbors,
            self._state.visited
        )

        # 5. Generate reasoning explanation
        unvisited_safe = [n for n in safe_neighbors if n not in self._state.visited]

        if chosen_move:
            reasoning = f"Exploring unvisited safe cell {chosen_move} (from {len(unvisited_safe)} options)"
        else:
            if safe_neighbors:
                reasoning = f"No unvisited safe neighbors. Visited: {safe_neighbors}"
            else:
                reasoning = "No safe neighbors found - cannot prove adjacent cells are safe"

        # 6. Create immutable step record
        step = AgentStep(
            step_number=step_number,
            position=self._state.position,
            percept=percept,
            safe_neighbors=tuple(safe_neighbors),
            chosen_move=chosen_move,
            reasoning=reasoning
        )

        # 7. Update state if moving
        if chosen_move:
            self._state.position = chosen_move
            self._state.visited.add(chosen_move)
            self._state.kb.mark_safe(*chosen_move)

        self._state.steps.append(step)
        return step

    def run_n_steps(self, world: WumpusWorld, n: int = 2) -> Tuple[AgentStep, ...]:
        """
        Execute n reasoning steps.

        SOFA: Functional - Returns immutable tuple of steps

        Args:
            world: Wumpus World environment
            n: Number of steps to execute

        Returns:
            Tuple of immutable AgentStep records
        """
        for step_num in range(1, n + 1):
            step = self.execute_step(world, step_num)
            if step.chosen_move is None:
                break  # No more moves possible

        return tuple(self._state.steps)

    def get_position(self) -> Tuple[int, int]:
        """Get current position."""
        return self._state.position

    def get_visited(self) -> Set[Tuple[int, int]]:
        """Get visited cells."""
        return self._state.visited.copy()

    def get_kb(self) -> WumpusKB:
        """Get knowledge base."""
        return self._state.kb


# ============================================================================
# SOFA: Single Responsibility - Separate presentation
# ============================================================================

class WumpusAgentPrinter:
    """
    Formats and prints agent execution traces.

    SOFA: Single Responsibility - Only handles output formatting
    """

    @staticmethod
    def print_step(step: AgentStep) -> None:
        """Print details of one agent step."""
        print(f"{'='*40} STEP {step.step_number} {'='*40}")
        print()
        print(f"Current position: {step.position}")
        print(step.percept)
        print()
        print(f"KB additions: {step.percept.position}")
        print()
        print(f"Safe neighbors found: {list(step.safe_neighbors)}")
        print(f"Reasoning: {step.reasoning}")
        if step.chosen_move:
            print(f"MOVING: {step.position} -> {step.chosen_move}")
        else:
            print("STOPPING: No move available")
        print()

    @staticmethod
    def print_trace(steps: Tuple[AgentStep, ...], world: WumpusWorld) -> None:
        """Print complete agent trace."""
        print("=" * 80)
        print("Wumpus World Agent - Reasoning Trace")
        print("=" * 80)
        print()
        print(f"World configuration: {world.grid_size}x{world.grid_size} grid")
        print(f"  Pits: {list(world.pits)}")
        print(f"  Wumpus: {world.wumpus}")
        print()

        for step in steps:
            WumpusAgentPrinter.print_step(step)

    @staticmethod
    def print_world_state(agent: WumpusAgent, world: WumpusWorld) -> None:
        """Print grid visualization of world state."""
        print("=" * 80)
        print("World State After Agent Moves")
        print("=" * 80)
        print()

        # Determine safe cells
        kb = agent.get_kb()
        visited = agent.get_visited()
        safe_cells = set(visited)

        for x in range(1, world.grid_size + 1):
            for y in range(1, world.grid_size + 1):
                if (x, y) not in visited and is_safe_cell(kb, x, y):
                    safe_cells.add((x, y))

        # Print grid
        print("    " + "   ".join(str(x) for x in range(1, world.grid_size + 1)))
        for y in range(world.grid_size, 0, -1):
            row = f"{y}  "
            for x in range(1, world.grid_size + 1):
                if (x, y) in visited:
                    cell = " V "
                elif (x, y) in safe_cells:
                    cell = " S "
                else:
                    cell = " ? "
                row += cell
            print(row)

        print()
        print("Legend: V = Visited, S = Safe (inferred), ? = Unknown")
        print()


# ============================================================================
# SOFA: Facade - Simplified interface (backward compatible)
# ============================================================================

def test_wumpus_agent(verbose: bool = True) -> Tuple[AgentStep, ...]:
    """
    Test Wumpus agent on standard scenario.

    SOFA: Facade pattern - Simple interface to complex subsystem

    Args:
        verbose: Whether to print trace

    Returns:
        Tuple of immutable agent steps
    """
    # Create world
    world = WumpusWorld(grid_size=4)
    world.add_pit(3, 1)
    world.add_wumpus(1, 3)

    # Create agent with default strategy
    agent = WumpusAgent(grid_size=4)

    # Run two steps
    steps = agent.run_n_steps(world, n=2)

    # Print if verbose
    if verbose:
        print("=" * 80)
        print("Part C: Wumpus World Agent Test")
        print("=" * 80)
        print()
        WumpusAgentPrinter.print_trace(steps, world)
        WumpusAgentPrinter.print_world_state(agent, world)
        print("=" * 80)
        print("Part C: Wumpus World Agent - COMPLETE")
        print("=" * 80)

    return steps


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    """Demonstrate refactored Wumpus agent."""
    print("=" * 80)
    print("Refactored Wumpus Agent (SOFA Principles)")
    print("=" * 80)
    print()

    steps = test_wumpus_agent(verbose=True)

    print()
    print(f"Executed {len(steps)} steps successfully")
    print("All SOFA principles applied:")
    print("  - Single Responsibility: World, Agent, Printer separated")
    print("  - Open/Closed: Extensible through MovementStrategy")
    print("  - Functional: Immutable Percept and AgentStep records")
    print("  - Abstraction: Clear interfaces, hidden implementation")
