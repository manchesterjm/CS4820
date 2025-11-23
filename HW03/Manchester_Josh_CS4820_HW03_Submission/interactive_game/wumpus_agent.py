"""
Refactored Wumpus World reasoning agent.

Design improvements:
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
# Immutable percept and move records
# ============================================================================

@dataclass(frozen=True)
class Percept:
    """
    Immutable percept snapshot.

    Frozen dataclass for immutability.

    Percepts in Wumpus World (Russell & Norvig Section 7.7):
    - Breeze: Felt in squares adjacent to pit
    - Stench: Smelled in squares adjacent to Wumpus
    - Glitter: Visible when gold is in current square
    - Bump: Felt when walking into wall
    - Scream: Heard when Wumpus is killed
    """
    position: Tuple[int, int]
    breeze: bool
    stench: bool
    glitter: bool
    bump: bool
    scream: bool

    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.breeze:
            parts.append("Breeze")
        if self.stench:
            parts.append("Stench")
        if self.glitter:
            parts.append("GLITTER")
        if self.bump:
            parts.append("Bump")
        if self.scream:
            parts.append("SCREAM")

        if not parts:
            parts.append("Nothing")

        return f"Percept({self.position}): {', '.join(parts)}"


@dataclass(frozen=True)
class AgentStep:
    """
    Immutable record of one agent step.

    Complete snapshot of reasoning process.
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
# World simulation only
# ============================================================================

class WumpusWorld:
    """
    Wumpus World environment simulator.

    Only simulates world, no agent logic.

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
        self.gold: Optional[Tuple[int, int]] = None
        self.wumpus_alive = True
        self.last_scream = False  # True for one percept after wumpus killed

    def add_pit(self, x: int, y: int) -> None:
        """Add a pit at location (x, y)."""
        self.pits.add((x, y))

    def add_wumpus(self, x: int, y: int) -> None:
        """Add the Wumpus at location (x, y)."""
        self.wumpus = (x, y)

    def add_gold(self, x: int, y: int) -> None:
        """Add gold at location (x, y)."""
        self.gold = (x, y)

    def grab_gold(self, x: int, y: int) -> bool:
        """
        Attempt to grab gold at current position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if gold was at this location and grabbed, False otherwise
        """
        if self.gold == (x, y):
            self.gold = None  # Remove gold from world
            return True
        return False

    def shoot_arrow(self, from_pos: Tuple[int, int], direction: str) -> bool:
        """
        Shoot arrow in direction from position.

        Args:
            from_pos: Current position (x, y)
            direction: One of 'up', 'down', 'left', 'right'

        Returns:
            True if wumpus was hit and killed, False otherwise
        """
        if not self.wumpus or not self.wumpus_alive:
            return False

        x, y = from_pos
        wx, wy = self.wumpus

        # Check if wumpus is in the direction of shot
        hit = False
        if direction == 'up' and x == wx and y < wy:
            hit = True
        elif direction == 'down' and x == wx and y > wy:
            hit = True
        elif direction == 'right' and y == wy and x < wx:
            hit = True
        elif direction == 'left' and y == wy and x > wx:
            hit = True

        if hit:
            self.wumpus_alive = False
            self.last_scream = True
            return True
        return False

    def is_agent_dead(self, x: int, y: int) -> bool:
        """
        Check if agent died at this location.

        Agent dies if:
        - Falls into a pit
        - Enters square with alive Wumpus

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if agent is dead, False otherwise
        """
        # Check if in pit
        if (x, y) in self.pits:
            return True

        # Check if in square with alive Wumpus
        if self.wumpus_alive and self.wumpus == (x, y):
            return True

        return False

    def sense_percepts(self, x: int, y: int, bump: bool = False) -> Percept:
        """
        Get immutable percept at location (x, y).

        Returns immutable Percept object.

        Args:
            x: X coordinate
            y: Y coordinate
            bump: True if agent bumped into wall

        Returns:
            Immutable Percept with all sensory information
        """
        breeze = self._has_adjacent_pit(x, y)
        stench = self._has_adjacent_wumpus(x, y) if self.wumpus_alive else False
        glitter = (self.gold == (x, y)) if self.gold else False

        # Scream is only perceived once after wumpus is killed
        scream = self.last_scream
        if self.last_scream:
            self.last_scream = False  # Clear scream after one percept

        return Percept(
            position=(x, y),
            breeze=breeze,
            stench=stench,
            glitter=glitter,
            bump=bump,
            scream=scream
        )

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
# Pure helper functions
# ============================================================================

def get_valid_neighbors(position: Tuple[int, int], grid_size: int) -> List[Tuple[int, int]]:
    """
    Pure function: Get valid neighbors within grid boundaries.

    No side effects, deterministic.

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

    Pure query, no KB modification.

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
# Strategy interface for movement policies
# ============================================================================

class MovementStrategy(ABC):
    """
    Abstract strategy for choosing moves.

    Defines interface without implementation.
    New strategies extend without modifying agent.
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
    Strategy: Global exploration with pathfinding to frontier cells.

    Implements strategy interface.

    Finds cells with unexplored safe neighbors anywhere on the visited map
    and pathfinds toward them. This allows full exploration of the reachable
    safe region, not just local dead-ends.
    """

    def __init__(self):
        """Initialize with visit frequency tracking."""
        self.visit_count: Dict[Tuple[int, int], int] = {}
        self.grid_size: Optional[int] = None

    def choose_move(
        self,
        current_position: Tuple[int, int],
        safe_neighbors: List[Tuple[int, int]],
        visited: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Choose move using global frontier search:

        1. If current position has unvisited safe neighbors, explore one
        2. Otherwise, find ANY visited cell with unvisited safe neighbors (frontier)
        3. Move toward closest frontier cell
        4. Stop only when no frontier cells exist globally
        """
        if not safe_neighbors:
            return None

        unvisited_safe = [loc for loc in safe_neighbors if loc not in visited]

        # Case 1: Current position has unvisited safe neighbors - explore
        if unvisited_safe:
            return unvisited_safe[0]

        # Case 2: No local unvisited neighbors - find frontier cells globally
        # A frontier cell is a visited cell that has unvisited safe neighbors
        frontier_cells = self._find_frontier_cells(visited, safe_neighbors)

        if not frontier_cells:
            # No frontier cells found anywhere - exploration complete
            return None

        # Case 3: Move toward nearest frontier cell
        # Since we don't have access to KB here, just move toward any visited safe neighbor
        # that gets us closer to a frontier (simple heuristic)
        visited_safe = [loc for loc in safe_neighbors if loc in visited]

        if visited_safe:
            # Choose least-visited to avoid loops
            least_visited = min(visited_safe, key=lambda loc: self.visit_count.get(loc, 0))

            # Limit revisits to prevent infinite loops
            if self.visit_count.get(least_visited, 0) < 5:
                self.visit_count[least_visited] = self.visit_count.get(least_visited, 0) + 1
                return least_visited

        return None

    def _find_frontier_cells(
        self,
        visited: Set[Tuple[int, int]],
        current_safe_neighbors: List[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Find all visited cells that have unvisited safe neighbors.

        Note: This is a simplified check - in the actual agent, we need KB access
        to determine which cells are safe. This is called from the strategy which
        doesn't have KB access, so we return current_safe_neighbors as a proxy.

        The agent will need to provide frontier info or we need to restructure.
        """
        # This is a limitation - we can't check other cells' safe neighbors
        # without KB access. Return empty for now, will fix in agent code.
        return set()


# ============================================================================
# Agent reasoning (no I/O)
# ============================================================================

@dataclass
class WumpusAgentState:
    """
    Mutable agent state during execution.

    Only tracks agent state.
    Note: Mutable for performance, converted to immutable trace at end.
    """
    kb: WumpusKB
    position: Tuple[int, int] = (1, 1)
    visited: Set[Tuple[int, int]] = field(default_factory=lambda: {(1, 1)})
    steps: List[AgentStep] = field(default_factory=list)
    grid_size: int = 4
    previous_position: Optional[Tuple[int, int]] = None  # Track where we came from
    has_gold: bool = False
    has_arrow: bool = True
    alive: bool = True
    wumpus_killed: bool = False
    game_won: bool = False
    confirmed_wumpus_location: Optional[Tuple[int, int]] = None  # Triangulated wumpus
    confirmed_pit_locations: Set[Tuple[int, int]] = field(default_factory=set)  # Logically confirmed pits
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)  # Track recent moves for oscillation detection
    blocked_frontiers: Set[Tuple[int, int]] = field(default_factory=set)  # Frontier cells we've tried but can't reach


class WumpusAgent:
    """
    Knowledge-based Wumpus World agent.

    Only agent reasoning, no world simulation or printing.
    Uses strategy pattern for movement.
    Hides KB details from clients.

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

        Returns immutable AgentStep.

        Args:
            world: Wumpus World environment
            step_number: Current step number

        Returns:
            Immutable record of this step
        """
        # 1. Check if agent died (fell into pit or met wumpus)
        if world.is_agent_dead(*self._state.position):
            self._state.alive = False
            death_reason = "Fell into pit!" if self._state.position in world.pits else "Eaten by Wumpus!"
            reasoning = f"GAME OVER: {death_reason}"
            step = AgentStep(
                step_number=step_number,
                position=self._state.position,
                percept=Percept(self._state.position, False, False, False, False, False),
                safe_neighbors=tuple(),
                chosen_move=None,
                reasoning=reasoning
            )
            self._state.steps.append(step)
            return step

        # 2. Sense percepts
        percept = world.sense_percepts(*self._state.position)

        # 3. Handle gold pickup
        if percept.glitter and not self._state.has_gold:
            grabbed = world.grab_gold(*self._state.position)
            if grabbed:
                self._state.has_gold = True

        # 4. Handle wumpus scream (if heard)
        if percept.scream:
            self._state.wumpus_killed = True

        # 5. Add percepts to KB
        self._state.kb.add_percept(
            percept.position[0],
            percept.position[1],
            percept.breeze,
            percept.stench
        )

        # 5b. Use logical reasoning to find confirmed pit locations
        # Pass existing confirmed pits so the algorithm doesn't double-count
        confirmed_pits = self._state.kb.get_confirmed_pits(self._state.confirmed_pit_locations)
        for pit_x, pit_y in confirmed_pits:
            if (pit_x, pit_y) not in self._state.confirmed_pit_locations:
                # Newly confirmed pit! Add to our tracking set
                self._state.confirmed_pit_locations.add((pit_x, pit_y))
                # Important: Don't mark as safe!
                # The pit is confirmed to exist, so we want to AVOID it

        # 6. Try to triangulate wumpus exact location
        wumpus_location = self._infer_wumpus_location()
        if wumpus_location:
            self._state.confirmed_wumpus_location = wumpus_location

        # 7. If have gold and know wumpus location, try to get in position to shoot
        shot_taken = False
        if self._state.has_gold and not self._state.wumpus_killed and wumpus_location and self._state.has_arrow:
            # Check if we can shoot from current position
            shot_direction = self._get_shot_direction(wumpus_location)
            if shot_direction:
                # We're aligned! Shoot now!
                hit = world.shoot_arrow(self._state.position, shot_direction)
                if hit:
                    self._state.wumpus_killed = True
                    self._state.game_won = True  # Have gold + killed wumpus = WIN!
                self._state.has_arrow = False
                shot_taken = True
            # If not aligned, agent should navigate to shooting position
            # This will be handled in movement logic below

        # 8. Find safe neighbors (excluding current position)
        neighbors = get_valid_neighbors(self._state.position, self._state.grid_size)
        safe_neighbors = [
            n for n in neighbors
            if is_safe_cell(self._state.kb, n[0], n[1]) and n != self._state.position
        ]

        # 9. Priority: If have gold and know wumpus, navigate to shooting position
        chosen_move = None
        if (self._state.has_gold and wumpus_location and not self._state.wumpus_killed
            and self._state.has_arrow and not shot_taken):
            # Try to find a safe VISITED cell that's aligned with wumpus for shooting
            aligned_positions = self._find_aligned_shooting_positions(wumpus_location)
            if aligned_positions:
                # Path toward the closest aligned position
                chosen_move = self._move_toward(aligned_positions[0], safe_neighbors)
            else:
                # No aligned position in visited cells yet - try immediate neighbor
                shooting_position = self._find_shooting_position(wumpus_location, safe_neighbors)
                if shooting_position:
                    chosen_move = shooting_position

        # 10. Choose move using strategy (if not already choosing shooting position)
        if chosen_move is None:
            chosen_move = self._strategy.choose_move(
                self._state.position,
                safe_neighbors,
                self._state.visited
            )

        # 11. Detect oscillation and mark blocked frontiers
        oscillating = self._detect_oscillation()
        if oscillating:
            # We're stuck in a loop trying to reach a frontier
            # Find which frontier we're currently targeting and mark it as blocked
            current_frontier = self._find_frontier_cell()
            if current_frontier:
                self._state.blocked_frontiers.add(current_frontier)

        # 12. If no move chosen, try global frontier search
        frontier_cell = None
        if chosen_move is None and safe_neighbors:
            # Current position has no unvisited safe neighbors
            # Look for frontier cells (visited cells with unvisited safe neighbors)
            frontier_cell = self._find_frontier_cell()
            if frontier_cell:
                # Path toward frontier cell through safe visited neighbors
                chosen_move = self._move_toward(frontier_cell, safe_neighbors)

        # 13. If still no move and no frontier, try calculated risk
        took_risk = False
        if chosen_move is None and not frontier_cell:
            # Truly stuck - try taking a calculated risk
            risky_move = self._choose_risky_move(neighbors)
            if risky_move:
                chosen_move = risky_move
                took_risk = True

        # 14. Safety check: never move to current position
        if chosen_move == self._state.position:
            chosen_move = None

        # 15. Check win condition
        if self._state.has_gold and self._state.wumpus_killed:
            self._state.game_won = True

        # 16. Generate reasoning explanation
        unvisited_safe = [n for n in safe_neighbors if n not in self._state.visited]

        reasoning_parts = []

        # Add gold/wumpus status
        if percept.glitter and self._state.has_gold:
            reasoning_parts.append("GRABBED GOLD!")
        if percept.scream:
            reasoning_parts.append("Heard scream - WUMPUS KILLED!")
        if self._state.game_won:
            reasoning_parts.append("GAME WON! (Have gold + Wumpus dead)")
        if wumpus_location and self._state.has_gold and not self._state.has_arrow:
            reasoning_parts.append(f"Shot arrow at wumpus location {wumpus_location}")

        # Add movement reasoning
        if chosen_move:
            if took_risk:
                reasoning_parts.append(f"CALCULATED RISK: Taking educated guess on {chosen_move}")
            elif unvisited_safe and chosen_move in unvisited_safe:
                reasoning_parts.append(f"Exploring unvisited safe cell {chosen_move}")
            elif frontier_cell:
                reasoning_parts.append(f"Navigating toward frontier cell {frontier_cell}")
            else:
                reasoning_parts.append(f"Moving to {chosen_move}")
        else:
            if safe_neighbors:
                reasoning_parts.append(f"Dead end: All reachable safe cells explored")
            else:
                reasoning_parts.append("Stuck: No safe moves, risks too high")

        # Add confirmed locations info to reasoning
        if confirmed_pits:
            pit_list = ", ".join(str(p) for p in sorted(confirmed_pits))
            reasoning_parts.insert(0, f"PITS CONFIRMED at {pit_list}")
        if wumpus_location:
            reasoning_parts.insert(0, f"WUMPUS LOCATED at {wumpus_location}!")

        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Exploring"

        # 16. Create immutable step record
        step = AgentStep(
            step_number=step_number,
            position=self._state.position,
            percept=percept,
            safe_neighbors=tuple(safe_neighbors),
            chosen_move=chosen_move,
            reasoning=reasoning
        )

        # 17. Update state if moving
        if chosen_move:
            # Track position for oscillation detection (keep last 8)
            self._state.recent_positions.append(self._state.position)
            if len(self._state.recent_positions) > 8:
                self._state.recent_positions.pop(0)

            # Update previous position
            self._state.previous_position = self._state.position

            # Move to new position
            self._state.position = chosen_move
            self._state.visited.add(chosen_move)
            self._state.kb.mark_safe(*chosen_move)

        self._state.steps.append(step)
        return step

    def run_n_steps(self, world: WumpusWorld, n: int = 2) -> Tuple[AgentStep, ...]:
        """
        Execute n reasoning steps.

        Returns immutable tuple of steps.

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

    def _detect_oscillation(self) -> Optional[Tuple[int, int]]:
        """
        Detect if agent is stuck in an oscillation loop.

        Looks at last 6 positions to detect if we're cycling between
        the same 2-3 cells repeatedly.

        Returns:
            The frontier cell we're stuck trying to reach, or None
        """
        if len(self._state.recent_positions) < 6:
            return None

        # Check last 6 positions for simple 2-cell oscillation: A->B->A->B->A->B
        last_6 = self._state.recent_positions[-6:]
        if (last_6[0] == last_6[2] == last_6[4] and
            last_6[1] == last_6[3] == last_6[5] and
            last_6[0] != last_6[1]):
            # We're oscillating between two cells!
            # This usually means we're trying to reach a blocked frontier
            # Return current position as signal (caller should find frontier context)
            return self._state.position

        return None

    def _find_frontier_cell(self) -> Optional[Tuple[int, int]]:
        """
        Find a visited cell that has unvisited safe neighbors (frontier cell).

        Skips frontiers that we've already tried to reach but failed (blocked).

        Returns:
            First frontier cell found, or None if no frontiers exist
        """
        for visited_cell in self._state.visited:
            # Skip current position - we already checked its neighbors
            if visited_cell == self._state.position:
                continue

            # Skip frontiers we've already tried and failed to reach
            if visited_cell in self._state.blocked_frontiers:
                continue

            neighbors = get_valid_neighbors(visited_cell, self._state.grid_size)
            safe_neighbors = [n for n in neighbors if is_safe_cell(self._state.kb, n[0], n[1])]
            unvisited_safe = [n for n in safe_neighbors if n not in self._state.visited]

            if unvisited_safe:
                # This visited cell has unexplored safe neighbors
                return visited_cell

        return None

    def _move_toward(
        self,
        target: Tuple[int, int],
        safe_neighbors: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Choose a safe neighbor that moves toward target cell.

        Uses Manhattan distance as heuristic, avoiding previous position
        to prevent oscillation.

        Args:
            target: Target cell to move toward
            safe_neighbors: Available safe neighbors from current position

        Returns:
            Best neighbor to move toward target, or None
        """
        if not safe_neighbors:
            return None

        def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # Filter out previous position to avoid immediate backtracking
        candidates = [n for n in safe_neighbors if n != self._state.previous_position]

        # If all neighbors lead back to previous position, we're stuck
        # In this case, allow backtracking as last resort
        if not candidates:
            candidates = safe_neighbors

        # Choose safe neighbor closest to target
        best_neighbor = min(
            candidates,
            key=lambda n: manhattan_distance(n, target)
        )

        return best_neighbor

    def _choose_risky_move(self, neighbors: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Choose a calculated risk when no safe moves are available.

        Picks the neighbor with the lowest estimated risk based on:
        - How many breeze/stench observations point to it
        - Distance from known dangers

        CRITICAL: Never chooses cells that are confirmed to be dangerous
        via logical deduction (confirmed pits or confirmed wumpus).

        Args:
            neighbors: All valid neighbors (may be unsafe)

        Returns:
            Least risky unvisited neighbor, or None
        """
        unvisited_neighbors = [n for n in neighbors if n not in self._state.visited]

        if not unvisited_neighbors:
            return None

        # CRITICAL: Filter out confirmed dangerous locations
        # These are cells we KNOW are pits or wumpus via logical deduction
        safe_candidates = []
        for n in unvisited_neighbors:
            # Never move to confirmed pit
            if n in self._state.confirmed_pit_locations:
                continue
            # Never move to confirmed wumpus location
            if self._state.confirmed_wumpus_location and n == self._state.confirmed_wumpus_location:
                continue
            safe_candidates.append(n)

        if not safe_candidates:
            # All unvisited neighbors are confirmed dangerous - no risky move available
            return None

        # Get probable danger locations (not yet confirmed, but suspicious)
        probable_pits = self._state.kb.get_probable_pits()
        probable_wumpus = self._state.kb.get_probable_wumpus()

        # Calculate risk score for each safe candidate
        risk_scores = {}
        for nx, ny in safe_candidates:
            risk = 0

            # Count how many breezes point to this cell (pit risk)
            breeze_count = 0
            for visited_cell in self._state.visited:
                vx, vy = visited_cell
                percept = self._state.kb.facts
                if f"B_{vx}_{vy}" in percept:
                    # This cell has breeze, check if neighbor is adjacent
                    if abs(nx - vx) + abs(ny - vy) == 1:
                        breeze_count += 1

            # Count how many stenches point to this cell (wumpus risk)
            stench_count = 0
            for visited_cell in self._state.visited:
                vx, vy = visited_cell
                percept = self._state.kb.facts
                if f"S_{vx}_{vy}" in percept:
                    if abs(nx - vx) + abs(ny - vy) == 1:
                        stench_count += 1

            # Risk score = breeze_count + stench_count * 2 (wumpus is worse)
            risk = breeze_count + stench_count * 2

            # If in probable sets, increase risk
            if (nx, ny) in probable_pits:
                risk += 3
            if (nx, ny) in probable_wumpus:
                risk += 5

            risk_scores[(nx, ny)] = risk

        # Choose cell with lowest risk (if risk is reasonable)
        if risk_scores:
            best_risky_move = min(risk_scores, key=risk_scores.get)
            min_risk = risk_scores[best_risky_move]

            # Only take risk if it's somewhat reasonable (risk <= 5)
            # Allows: 1-2 breezes, or 1 breeze + probable pit, or 1 stench + probable wumpus
            # Rejects: Multiple stenches, or very high combined risk
            if min_risk <= 5:
                return best_risky_move

        return None

    def _infer_wumpus_location(self) -> Optional[Tuple[int, int]]:
        """
        Triangulate exact wumpus location from multiple stench observations.

        If we have 2+ cells with stenches, we can often narrow down to ONE
        location that is adjacent to all of them.

        Returns:
            Wumpus location if uniquely determined, None otherwise
        """
        # Find all cells with stench
        stench_cells = []
        for fact in self._state.kb.facts:
            if fact.startswith("S_") and not fact.startswith("not_S"):
                parts = fact.split("_")
                if len(parts) == 3:
                    x, y = int(parts[1]), int(parts[2])
                    stench_cells.append((x, y))

        if len(stench_cells) < 2:
            # Need at least 2 stenches to triangulate
            return None

        # Find intersection of all possible wumpus locations
        # Start with neighbors of first stench cell
        possible_wumpus = set(get_valid_neighbors(stench_cells[0], self._state.grid_size))

        # Intersect with neighbors of each additional stench cell
        for stench_cell in stench_cells[1:]:
            neighbors = set(get_valid_neighbors(stench_cell, self._state.grid_size))
            possible_wumpus = possible_wumpus.intersection(neighbors)

        # Remove cells we know DON'T have wumpus
        for fact in self._state.kb.facts:
            if fact.startswith("not_W_"):
                parts = fact.split("_")
                # Format: "not_W_x_y" splits into ["not", "W", "x", "y"] = 4 parts
                if len(parts) == 4:
                    x, y = int(parts[2]), int(parts[3])
                    possible_wumpus.discard((x, y))

        # IMPROVEMENT: Use negative stench observations to narrow down further
        # If we visited a cell with NO stench, the wumpus is NOT adjacent to it
        cells_without_stench = []
        for fact in self._state.kb.facts:
            if fact.startswith("not_S_"):
                parts = fact.split("_")
                # Format: "not_S_x_y" splits into ["not", "S", "x", "y"] = 4 parts
                if len(parts) == 4:
                    x, y = int(parts[2]), int(parts[3])
                    cells_without_stench.append((x, y))

        # For each no-stench cell, eliminate its neighbors from candidates
        for no_stench_cell in cells_without_stench:
            neighbors = set(get_valid_neighbors(no_stench_cell, self._state.grid_size))
            for neighbor in neighbors:
                possible_wumpus.discard(neighbor)

        # If exactly one location remains, that's the wumpus!
        if len(possible_wumpus) == 1:
            return possible_wumpus.pop()

        return None

    def _get_shot_direction(self, target: Tuple[int, int]) -> Optional[str]:
        """
        Determine direction to shoot arrow from current position to target.

        Agent can only shoot in straight lines: up, down, left, right.

        Args:
            target: Target (x, y) location

        Returns:
            Direction ('up', 'down', 'left', 'right') or None if not aligned
        """
        curr_x, curr_y = self._state.position
        target_x, target_y = target

        # Check if target is in straight line
        if curr_x == target_x:
            # Same column - shoot up or down
            if curr_y < target_y:
                return 'up'
            elif curr_y > target_y:
                return 'down'
        elif curr_y == target_y:
            # Same row - shoot left or right
            if curr_x < target_x:
                return 'right'
            elif curr_x > target_x:
                return 'left'

        # Target not aligned - can't shoot straight
        return None

    def _find_aligned_shooting_positions(self, target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find all safe VISITED cells that are aligned with target for shooting.

        Agent can shoot in straight lines (up/down/left/right), so we need
        a cell in the same row OR same column as the target.

        Args:
            target: Target wumpus location (x, y)

        Returns:
            List of visited safe cells aligned with target, sorted by distance
        """
        tx, ty = target
        aligned = []

        # Check all visited cells
        for cell in self._state.visited:
            x, y = cell
            # Skip the target itself
            if cell == target:
                continue
            # Check if aligned (same row OR same column)
            if x == tx or y == ty:
                # Also verify it's safe (no confirmed pit there)
                if cell not in self._state.confirmed_pit_locations:
                    aligned.append(cell)

        # Sort by Manhattan distance to target (closest first)
        aligned.sort(key=lambda c: abs(c[0] - tx) + abs(c[1] - ty))
        return aligned

    def _find_shooting_position(
        self,
        target: Tuple[int, int],
        safe_neighbors: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        Find a safe neighbor that allows shooting at target.

        Args:
            target: Target wumpus location
            safe_neighbors: Available safe neighbors from current position

        Returns:
            Safe neighbor aligned with target, or None
        """
        for neighbor in safe_neighbors:
            nx, ny = neighbor
            tx, ty = target

            # Check if this neighbor is aligned with target
            if nx == tx or ny == ty:
                # Aligned! This is a valid shooting position
                return neighbor

        return None


# ============================================================================
# Separate presentation
# ============================================================================

class WumpusAgentPrinter:
    """
    Formats and prints agent execution traces.

    Only handles output formatting.
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
# Simplified interface (backward compatible)
# ============================================================================

def test_wumpus_agent(verbose: bool = True) -> Tuple[AgentStep, ...]:
    """
    Test Wumpus agent on standard scenario.

    Facade pattern - Simple interface to complex subsystem.

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
    print("Refactored Wumpus Agent")
    print("=" * 80)
    print()

    steps = test_wumpus_agent(verbose=True)

    print()
    print(f"Executed {len(steps)} steps successfully")
    print("Design principles applied:")
    print("  - Single Responsibility: World, Agent, Printer separated")
    print("  - Open/Closed: Extensible through MovementStrategy")
    print("  - Functional: Immutable Percept and AgentStep records")
    print("  - Abstraction: Clear interfaces, hidden implementation")
