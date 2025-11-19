"""
Pygame visualization of Wumpus World with AI reasoning display.

Interactive game showing:
- Left panel: Visual game board with agent, pits, Wumpus
- Right panel: AI logic and reasoning display

Controls:
- SPACE: Execute next step
- R: Reset game
- Q/ESC: Quit
- A: Toggle auto-play mode

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
"""

import pygame
import sys
import time
import random
import datetime
from typing import Tuple, Set, Optional, List
from wumpus_agent import (
    WumpusWorld, WumpusAgent, AgentStep, Percept,
    UnvisitedFirstStrategy, is_safe_cell
)
from knowledge_base import WumpusKB

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_GRAY = (240, 240, 240)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 220)
YELLOW = (255, 215, 0)
ORANGE = (255, 165, 0)
PURPLE = (160, 32, 240)
DARK_GREEN = (0, 100, 0)
DARK_RED = (139, 0, 0)
GOLD = (255, 215, 0)

# Window dimensions
BOARD_SIZE = 800
PANEL_WIDTH = 500
WINDOW_WIDTH = BOARD_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = 900
INFO_HEIGHT = 100

# Grid settings
GRID_SIZE = 8
CELL_SIZE = BOARD_SIZE // GRID_SIZE

# Fonts
TITLE_FONT = pygame.font.Font(None, 32)
HEADER_FONT = pygame.font.Font(None, 24)
TEXT_FONT = pygame.font.Font(None, 20)
SMALL_FONT = pygame.font.Font(None, 16)
CELL_FONT = pygame.font.Font(None, 24)  # Larger for 8x8 grid


class WumpusGameVisual:
    """
    Pygame visualization for Wumpus World with AI reasoning.
    """

    def __init__(self):
        """Initialize the game visualization."""
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Wumpus World - AI Agent Visualization")
        self.clock = pygame.time.Clock()

        # Game state
        self.world = None
        self.agent = None
        self.current_step = 0
        self.max_steps = 100  # Sufficient for 8x8 grid
        self.steps_history = []
        self.game_over = False
        self.auto_play = False
        self.auto_play_delay = 0.3  # Faster for smaller grid
        self.last_auto_step = 0

        # Animation state
        self.animating = False
        self.anim_start_pos = None
        self.anim_end_pos = None
        self.anim_progress = 0.0
        self.anim_speed = 0.05

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"wumpus_game_log_{timestamp}.txt"
        self.log_file = None

        # Initialize game
        self.reset_game()

    def reset_game(self):
        """Reset the game to initial state with randomized obstacles."""
        # Create world with obstacles
        self.world = WumpusWorld(grid_size=GRID_SIZE)

        # Randomize pit and wumpus locations
        # Avoid starting position (1,1)
        available_positions = [
            (x, y) for x in range(1, GRID_SIZE + 1)
            for y in range(1, GRID_SIZE + 1)
            if (x, y) != (1, 1)
        ]

        # Add random pits (approximately 8% of grid for better exploration)
        # Lower density allows agent to explore larger connected safe regions
        num_pits = max(3, int(GRID_SIZE * GRID_SIZE * 0.08))
        pit_positions = random.sample(available_positions, num_pits)
        for x, y in pit_positions:
            self.world.add_pit(x, y)

        # Add wumpus at random position (not on a pit, not at start)
        wumpus_positions = [pos for pos in available_positions if pos not in pit_positions]
        wumpus_pos = random.choice(wumpus_positions)
        self.world.add_wumpus(wumpus_pos[0], wumpus_pos[1])

        # Add gold at random SAFE position (not on pit/wumpus, and not adjacent to any pits)
        # This ensures agent can actually reach and detect the gold
        def is_safe_for_gold(pos):
            x, y = pos
            # Not on pit or wumpus or start
            if pos in pit_positions or pos == wumpus_pos:
                return False
            # Not adjacent to any pit (agent would sense breeze and avoid)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_x, adj_y = x + dx, y + dy
                if (adj_x, adj_y) in pit_positions:
                    return False
            # Not adjacent to wumpus (agent would sense stench and avoid)
            if wumpus_pos:
                wx, wy = wumpus_pos
                if abs(x - wx) + abs(y - wy) == 1:
                    return False
            return True

        gold_positions = [pos for pos in available_positions if is_safe_for_gold(pos)]
        if gold_positions:
            gold_pos = random.choice(gold_positions)
            self.world.add_gold(gold_pos[0], gold_pos[1])
        else:
            # Fallback: place near start if no safe positions (very rare)
            self.world.add_gold(2, 1)

        # Create agent
        self.agent = WumpusAgent(grid_size=GRID_SIZE, strategy=UnvisitedFirstStrategy())

        # Reset state
        self.current_step = 0
        self.steps_history = []
        self.game_over = False
        self.animating = False

        # Start logging
        self.log_file = open(self.log_filename, 'w', encoding='utf-8')
        self.log_file.write(f"=== WUMPUS WORLD GAME LOG ===\n")
        self.log_file.write(f"Timestamp: {datetime.datetime.now()}\n")
        self.log_file.write(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}\n\n")
        self.log_file.write(f"=== WORLD SETUP ===\n")
        self.log_file.write(f"Pits: {sorted(list(self.world.pits))}\n")
        self.log_file.write(f"Wumpus: {self.world.wumpus}\n")
        self.log_file.write(f"Gold: {self.world.gold}\n")
        self.log_file.write(f"\n=== AGENT EXECUTION ===\n\n")
        self.log_file.flush()

    def grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """
        Convert grid coordinates to screen pixel coordinates (center of cell).

        Grid coords are 1-indexed, (1,1) is bottom-left.
        Screen coords: (0,0) is top-left.
        """
        screen_x = (grid_x - 1) * CELL_SIZE + CELL_SIZE // 2
        screen_y = BOARD_SIZE - (grid_y * CELL_SIZE) + CELL_SIZE // 2
        return screen_x, screen_y

    def screen_to_grid(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to grid coordinates."""
        grid_x = (screen_x // CELL_SIZE) + 1
        grid_y = GRID_SIZE - (screen_y // CELL_SIZE)
        return grid_x, grid_y

    def draw_grid(self):
        """Draw the game board grid."""
        # Fill background
        board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, LIGHT_GRAY, board_rect)

        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen, DARK_GRAY,
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, BOARD_SIZE),
                2
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen, DARK_GRAY,
                (0, i * CELL_SIZE),
                (BOARD_SIZE, i * CELL_SIZE),
                2
            )

    def draw_cell_content(self, grid_x: int, grid_y: int, visited: Set[Tuple[int, int]]):
        """Draw contents of a cell (pits, wumpus, percepts)."""
        screen_x, screen_y = self.grid_to_screen(grid_x, grid_y)
        cell_rect = pygame.Rect(
            (grid_x - 1) * CELL_SIZE + 2,
            BOARD_SIZE - grid_y * CELL_SIZE + 2,
            CELL_SIZE - 4,
            CELL_SIZE - 4
        )

        pos = (grid_x, grid_y)

        # Get agent's inferred probable locations
        kb = self.agent.get_kb()
        probable_pits = kb.get_probable_pits()
        probable_wumpus = kb.get_probable_wumpus()
        confirmed_wumpus = self.agent._state.confirmed_wumpus_location
        confirmed_pits = self.agent._state.confirmed_pit_locations

        # Highlight CONFIRMED locations with bright colors
        if confirmed_wumpus and pos == confirmed_wumpus:
            # Confirmed wumpus - bright orange-red
            pygame.draw.rect(self.screen, (255, 100, 0), cell_rect)
        elif pos in confirmed_pits:
            # Confirmed pit - dark red
            pygame.draw.rect(self.screen, (180, 0, 0), cell_rect)

        # Highlight probable danger cells (if not visited and not confirmed)
        elif pos not in visited:
            if pos in probable_pits and pos in probable_wumpus:
                # Both pit and wumpus possible - orange background
                pygame.draw.rect(self.screen, (255, 200, 150), cell_rect)
            elif pos in probable_pits:
                # Probable pit - light red background
                pygame.draw.rect(self.screen, (255, 220, 220), cell_rect)
            elif pos in probable_wumpus:
                # Probable wumpus - light purple background
                pygame.draw.rect(self.screen, (255, 220, 255), cell_rect)

        # Highlight visited cells
        if pos in visited:
            pygame.draw.rect(self.screen, (220, 255, 220), cell_rect)

        # Draw starting position marker
        if pos == (1, 1):
            pygame.draw.rect(self.screen, (200, 255, 200), cell_rect)

        # Show actual world contents (pits and wumpus) - semi-transparent if not visited
        if pos in self.world.pits:
            alpha = 255 if pos in visited else 80
            pit_radius = max(8, CELL_SIZE // 4)
            pit_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(pit_surf, (*BLACK, alpha), (CELL_SIZE // 2, CELL_SIZE // 2), pit_radius)
            self.screen.blit(pit_surf, (screen_x - CELL_SIZE // 2, screen_y - CELL_SIZE // 2))
            label = CELL_FONT.render("P", True, (*BLACK, alpha))
            self.screen.blit(label, (screen_x - 6, screen_y - 8))

        if self.world.wumpus and pos == self.world.wumpus and self.world.wumpus_alive:
            alpha = 255 if pos in visited else 80
            wumpus_size = max(12, CELL_SIZE // 3)
            wumpus_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(
                wumpus_surf, (*RED, alpha),
                [(CELL_SIZE // 2, CELL_SIZE // 2 - wumpus_size // 2),
                 (CELL_SIZE // 2 - wumpus_size // 2, CELL_SIZE // 2 + wumpus_size // 2),
                 (CELL_SIZE // 2 + wumpus_size // 2, CELL_SIZE // 2 + wumpus_size // 2)]
            )
            self.screen.blit(wumpus_surf, (screen_x - CELL_SIZE // 2, screen_y - CELL_SIZE // 2))
            label = CELL_FONT.render("W", True, (*RED, alpha))
            self.screen.blit(label, (screen_x - 7, screen_y - 8))

        # Draw gold (only if not grabbed by agent)
        if self.world.gold and pos == self.world.gold:
            gold_radius = max(6, CELL_SIZE // 5)
            pygame.draw.circle(self.screen, GOLD, (screen_x, screen_y), gold_radius)
            pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), gold_radius, 1)
            label = CELL_FONT.render("G", True, BLACK)
            self.screen.blit(label, (screen_x - 6, screen_y - 8))

        # Draw percepts for visited cells
        if pos in visited:
            percept = self.world.sense_percepts(grid_x, grid_y)
            percept_text = []
            if percept.breeze:
                percept_text.append("B")
            if percept.stench:
                percept_text.append("S")

            if percept_text:
                percept_str = ",".join(percept_text)
                label = SMALL_FONT.render(percept_str, True, BLUE)
                self.screen.blit(label, (screen_x - 8, screen_y - CELL_SIZE // 3))

    def draw_agent(self, position: Tuple[int, int]):
        """Draw the agent at given position."""
        screen_x, screen_y = self.grid_to_screen(position[0], position[1])
        agent_radius = max(8, CELL_SIZE // 4)

        # Draw agent as a circle with direction indicator
        pygame.draw.circle(self.screen, ORANGE, (screen_x, screen_y), agent_radius)
        pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), agent_radius, 2)

        # Draw eyes (proportional to agent size)
        eye_offset_x = agent_radius // 3
        eye_offset_y = agent_radius // 4
        eye_radius = max(2, agent_radius // 6)
        pygame.draw.circle(self.screen, BLACK, (screen_x - eye_offset_x, screen_y - eye_offset_y), eye_radius)
        pygame.draw.circle(self.screen, BLACK, (screen_x + eye_offset_x, screen_y - eye_offset_y), eye_radius)

        # Draw smile
        smile_width = agent_radius
        smile_height = agent_radius // 2
        pygame.draw.arc(self.screen, BLACK,
                       (screen_x - smile_width // 2, screen_y - smile_height // 2,
                        smile_width, smile_height), 3.14, 6.28, 2)

    def draw_animated_agent(self):
        """Draw agent with animation between positions."""
        if not self.animating or self.anim_start_pos is None or self.anim_end_pos is None:
            return

        # Interpolate position
        start_screen = self.grid_to_screen(*self.anim_start_pos)
        end_screen = self.grid_to_screen(*self.anim_end_pos)

        current_x = start_screen[0] + (end_screen[0] - start_screen[0]) * self.anim_progress
        current_y = start_screen[1] + (end_screen[1] - start_screen[1]) * self.anim_progress

        # Draw agent at interpolated position (same sizing as static agent)
        agent_radius = max(8, CELL_SIZE // 4)
        eye_offset_x = agent_radius // 3
        eye_offset_y = agent_radius // 4
        eye_radius = max(2, agent_radius // 6)
        smile_width = agent_radius
        smile_height = agent_radius // 2

        pygame.draw.circle(self.screen, ORANGE, (int(current_x), int(current_y)), agent_radius)
        pygame.draw.circle(self.screen, BLACK, (int(current_x), int(current_y)), agent_radius, 2)
        pygame.draw.circle(self.screen, BLACK, (int(current_x) - eye_offset_x, int(current_y) - eye_offset_y), eye_radius)
        pygame.draw.circle(self.screen, BLACK, (int(current_x) + eye_offset_x, int(current_y) - eye_offset_y), eye_radius)
        pygame.draw.arc(self.screen, BLACK,
                       (int(current_x) - smile_width // 2, int(current_y) - smile_height // 2,
                        smile_width, smile_height), 3.14, 6.28, 2)

    def draw_reasoning_panel(self):
        """Draw the AI reasoning panel on the right side."""
        panel_x = BOARD_SIZE
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.line(self.screen, BLACK, (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)

        y_offset = 20

        # Title
        title = TITLE_FONT.render("AI Reasoning", True, BLACK)
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 50

        # Current step info
        step_text = f"Step: {self.current_step}"
        step_surf = HEADER_FONT.render(step_text, True, BLACK)
        self.screen.blit(step_surf, (panel_x + 20, y_offset))
        y_offset += 35

        # Agent position
        pos = self.agent.get_position()
        pos_text = f"Position: {pos}"
        pos_surf = TEXT_FONT.render(pos_text, True, DARK_GREEN)
        self.screen.blit(pos_surf, (panel_x + 20, y_offset))
        y_offset += 30

        # Current percepts
        if self.steps_history:
            last_step = self.steps_history[-1]
            percept = last_step.percept

            percept_header = HEADER_FONT.render("Percepts:", True, BLACK)
            self.screen.blit(percept_header, (panel_x + 20, y_offset))
            y_offset += 30

            breeze_text = f"  Breeze: {'YES' if percept.breeze else 'NO'}"
            stench_text = f"  Stench: {'YES' if percept.stench else 'NO'}"
            breeze_color = RED if percept.breeze else GREEN
            stench_color = RED if percept.stench else GREEN

            breeze_surf = TEXT_FONT.render(breeze_text, True, breeze_color)
            self.screen.blit(breeze_surf, (panel_x + 20, y_offset))
            y_offset += 25

            stench_surf = TEXT_FONT.render(stench_text, True, stench_color)
            self.screen.blit(stench_surf, (panel_x + 20, y_offset))
            y_offset += 25

            glitter_text = f"  Glitter: {'YES' if percept.glitter else 'NO'}"
            glitter_color = GOLD if percept.glitter else GRAY
            glitter_surf = TEXT_FONT.render(glitter_text, True, glitter_color)
            self.screen.blit(glitter_surf, (panel_x + 20, y_offset))
            y_offset += 35

            # Agent status
            agent_state = self.agent._state
            status_header = HEADER_FONT.render("Agent Status:", True, BLACK)
            self.screen.blit(status_header, (panel_x + 20, y_offset))
            y_offset += 30

            gold_status = "Has Gold: YES" if agent_state.has_gold else "Has Gold: NO"
            gold_color = GOLD if agent_state.has_gold else GRAY
            gold_surf = SMALL_FONT.render(f"  {gold_status}", True, gold_color)
            self.screen.blit(gold_surf, (panel_x + 20, y_offset))
            y_offset += 20

            arrow_status = "Has Arrow: YES" if agent_state.has_arrow else "Has Arrow: NO"
            arrow_color = BLACK if agent_state.has_arrow else GRAY
            arrow_surf = SMALL_FONT.render(f"  {arrow_status}", True, arrow_color)
            self.screen.blit(arrow_surf, (panel_x + 20, y_offset))
            y_offset += 20

            wumpus_status = "Wumpus Killed: YES" if agent_state.wumpus_killed else "Wumpus Killed: NO"
            wumpus_color = GREEN if agent_state.wumpus_killed else GRAY
            wumpus_surf = SMALL_FONT.render(f"  {wumpus_status}", True, wumpus_color)
            self.screen.blit(wumpus_surf, (panel_x + 20, y_offset))
            y_offset += 35

            # KB additions
            kb_header = HEADER_FONT.render("KB Updates:", True, BLACK)
            self.screen.blit(kb_header, (panel_x + 20, y_offset))
            y_offset += 30

            if percept.breeze:
                kb_surf = SMALL_FONT.render(f"  Added: B_{pos[0]}_{pos[1]}", True, BLACK)
            else:
                kb_surf = SMALL_FONT.render(f"  Added: not_B_{pos[0]}_{pos[1]}", True, BLACK)
            self.screen.blit(kb_surf, (panel_x + 20, y_offset))
            y_offset += 20

            if percept.stench:
                kb_surf = SMALL_FONT.render(f"  Added: S_{pos[0]}_{pos[1]}", True, BLACK)
            else:
                kb_surf = SMALL_FONT.render(f"  Added: not_S_{pos[0]}_{pos[1]}", True, BLACK)
            self.screen.blit(kb_surf, (panel_x + 20, y_offset))
            y_offset += 35

            # Safe neighbors
            safe_header = HEADER_FONT.render("Safe Neighbors:", True, BLACK)
            self.screen.blit(safe_header, (panel_x + 20, y_offset))
            y_offset += 30

            if last_step.safe_neighbors:
                for neighbor in last_step.safe_neighbors:
                    safe_surf = SMALL_FONT.render(f"  {neighbor}", True, DARK_GREEN)
                    self.screen.blit(safe_surf, (panel_x + 20, y_offset))
                    y_offset += 20
            else:
                safe_surf = SMALL_FONT.render("  None found", True, RED)
                self.screen.blit(safe_surf, (panel_x + 20, y_offset))
                y_offset += 20

            y_offset += 15

            # Reasoning
            reason_header = HEADER_FONT.render("Decision:", True, BLACK)
            self.screen.blit(reason_header, (panel_x + 20, y_offset))
            y_offset += 30

            # Word wrap reasoning text
            reasoning_words = last_step.reasoning.split()
            line = ""
            for word in reasoning_words:
                test_line = line + word + " "
                if TEXT_FONT.size(test_line)[0] > PANEL_WIDTH - 60:
                    reason_surf = SMALL_FONT.render(line, True, BLACK)
                    self.screen.blit(reason_surf, (panel_x + 25, y_offset))
                    y_offset += 20
                    line = word + " "
                else:
                    line = test_line
            if line:
                reason_surf = SMALL_FONT.render(line, True, BLACK)
                self.screen.blit(reason_surf, (panel_x + 25, y_offset))
                y_offset += 20

            # Next move
            if last_step.chosen_move:
                y_offset += 10
                move_surf = TEXT_FONT.render(f"Next move: {last_step.chosen_move}", True, BLUE)
                self.screen.blit(move_surf, (panel_x + 20, y_offset))
        else:
            # Initial state
            ready_surf = TEXT_FONT.render("Press SPACE to start", True, GRAY)
            self.screen.blit(ready_surf, (panel_x + 20, y_offset))

    def draw_info_bar(self):
        """Draw information bar at bottom."""
        info_rect = pygame.Rect(0, BOARD_SIZE, WINDOW_WIDTH, INFO_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, info_rect)
        pygame.draw.line(self.screen, BLACK, (0, BOARD_SIZE), (WINDOW_WIDTH, BOARD_SIZE), 2)

        y = BOARD_SIZE + 15

        # Controls
        controls = [
            "SPACE: Next Step",
            "R: Reset",
            "A: Auto-play",
            "Q/ESC: Quit"
        ]

        x_offset = 20
        for control in controls:
            surf = SMALL_FONT.render(control, True, WHITE)
            self.screen.blit(surf, (x_offset, y))
            x_offset += 140

        # Game status
        y += 25
        agent_state = self.agent._state if self.agent else None
        if self.game_over and agent_state:
            if agent_state.game_won:
                status = "GAME WON! Agent got gold and killed Wumpus!"
                status_surf = TEXT_FONT.render(status, True, GOLD)
            elif not agent_state.alive:
                if self.world.is_agent_dead(*agent_state.position):
                    if agent_state.position in self.world.pits:
                        status = "GAME OVER - Agent fell into pit!"
                    else:
                        status = "GAME OVER - Agent eaten by Wumpus!"
                    status_surf = TEXT_FONT.render(status, True, RED)
                else:
                    status = "GAME OVER - Agent stopped"
                    status_surf = TEXT_FONT.render(status, True, YELLOW)
            else:
                status = "GAME OVER - Agent stopped"
                status_surf = TEXT_FONT.render(status, True, YELLOW)
        elif self.auto_play:
            status = "AUTO-PLAY MODE (press A to stop)"
            status_surf = TEXT_FONT.render(status, True, GREEN)
        else:
            status = "Ready for next step"
            status_surf = TEXT_FONT.render(status, True, WHITE)

        self.screen.blit(status_surf, (20, y))

        # Legend
        y += 30
        legend = "Legend: P=Pit, W=Wumpus, G=Gold, B=Breeze, S=Stench | Pink=Probable Pit, DARK RED=Confirmed Pit!, ORANGE=Confirmed Wumpus!"
        legend_surf = SMALL_FONT.render(legend, True, WHITE)
        self.screen.blit(legend_surf, (20, y))

    def execute_step(self):
        """Execute one agent reasoning step."""
        if self.game_over or self.current_step >= self.max_steps:
            return

        # Execute step
        step = self.agent.execute_step(self.world, self.current_step + 1)
        self.steps_history.append(step)
        self.current_step += 1

        # Log this step
        self.log_step(step)

        # Check if agent died or won
        agent_state = self.agent._state
        if not agent_state.alive or agent_state.game_won:
            self.game_over = True
            self.log_game_end(agent_state)
            return

        # Check if agent can continue
        if step.chosen_move is None:
            self.game_over = True
            self.log_game_end(agent_state)
        else:
            # Start animation
            self.animating = True
            self.anim_start_pos = step.position
            self.anim_end_pos = step.chosen_move
            self.anim_progress = 0.0

    def log_step(self, step: AgentStep):
        """Log a single agent step to file."""
        if not self.log_file:
            return

        agent_state = self.agent._state
        self.log_file.write(f"Step {step.step_number}:\n")
        self.log_file.write(f"  Position: {step.position}\n")
        self.log_file.write(f"  Percepts: {step.percept}\n")
        self.log_file.write(f"  Safe neighbors: {list(step.safe_neighbors)}\n")
        self.log_file.write(f"  Chosen move: {step.chosen_move}\n")
        self.log_file.write(f"  Reasoning: {step.reasoning}\n")
        self.log_file.write(f"  Agent state: has_gold={agent_state.has_gold}, ")
        self.log_file.write(f"has_arrow={agent_state.has_arrow}, ")
        self.log_file.write(f"wumpus_confirmed={agent_state.confirmed_wumpus_location}, ")
        self.log_file.write(f"pits_confirmed={sorted(list(agent_state.confirmed_pit_locations))}\n")
        self.log_file.write(f"  Cells visited so far: {len(agent_state.visited)}\n")
        self.log_file.write(f"\n")
        self.log_file.flush()

    def log_game_end(self, agent_state):
        """Log game end statistics."""
        if not self.log_file:
            return

        self.log_file.write(f"\n=== GAME END ===\n")
        self.log_file.write(f"Total steps: {self.current_step}\n")
        self.log_file.write(f"Cells explored: {len(agent_state.visited)}/{GRID_SIZE*GRID_SIZE}\n")
        self.log_file.write(f"Coverage: {len(agent_state.visited)/(GRID_SIZE*GRID_SIZE)*100:.1f}%\n")
        self.log_file.write(f"Has gold: {agent_state.has_gold}\n")
        self.log_file.write(f"Wumpus killed: {agent_state.wumpus_killed}\n")
        self.log_file.write(f"Game won: {agent_state.game_won}\n")
        self.log_file.write(f"Agent alive: {agent_state.alive}\n")
        self.log_file.write(f"\n=== LOGICAL REASONING STATS ===\n")
        self.log_file.write(f"Pits confirmed by logic: {len(agent_state.confirmed_pit_locations)}/{len(self.world.pits)}\n")
        self.log_file.write(f"Confirmed pit locations: {sorted(list(agent_state.confirmed_pit_locations))}\n")
        self.log_file.write(f"Actual pit locations: {sorted(list(self.world.pits))}\n")
        correct_pits = agent_state.confirmed_pit_locations.intersection(self.world.pits)
        self.log_file.write(f"Correctly identified pits: {len(correct_pits)}\n")
        if len(agent_state.confirmed_pit_locations) > 0:
            accuracy = len(correct_pits) / len(agent_state.confirmed_pit_locations) * 100
            self.log_file.write(f"Pit identification accuracy: {accuracy:.1f}%\n")

        if agent_state.game_won:
            self.log_file.write(f"\nRESULT: VICTORY!\n")
        elif not agent_state.alive:
            if agent_state.position in self.world.pits:
                self.log_file.write(f"\nRESULT: DEFEAT - Fell into pit at {agent_state.position}\n")
            else:
                self.log_file.write(f"\nRESULT: DEFEAT - Eaten by Wumpus at {agent_state.position}\n")
        else:
            self.log_file.write(f"\nRESULT: INCOMPLETE - Agent stopped exploring\n")

        self.log_file.write(f"\nLog saved to: {self.log_filename}\n")
        self.log_file.flush()
        self.log_file.close()
        self.log_file = None

    def update_animation(self):
        """Update animation state."""
        if self.animating:
            self.anim_progress += self.anim_speed
            if self.anim_progress >= 1.0:
                self.animating = False
                self.anim_progress = 1.0

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE and not self.animating:
                    self.execute_step()

                elif event.key == pygame.K_r:
                    self.reset_game()

                elif event.key == pygame.K_a:
                    self.auto_play = not self.auto_play
                    self.last_auto_step = time.time()

        return True

    def run(self):
        """Main game loop."""
        running = True

        while running:
            # Handle events
            running = self.handle_events()

            # Auto-play
            if self.auto_play and not self.animating and not self.game_over:
                current_time = time.time()
                if current_time - self.last_auto_step >= self.auto_play_delay:
                    self.execute_step()
                    self.last_auto_step = current_time

            # Update animation
            self.update_animation()

            # Draw everything
            self.screen.fill(WHITE)

            # Draw game board
            self.draw_grid()
            visited = self.agent.get_visited()
            for x in range(1, GRID_SIZE + 1):
                for y in range(1, GRID_SIZE + 1):
                    self.draw_cell_content(x, y, visited)

            # Draw agent
            if self.animating:
                self.draw_animated_agent()
            else:
                self.draw_agent(self.agent.get_position())

            # Draw reasoning panel
            self.draw_reasoning_panel()

            # Draw info bar
            self.draw_info_bar()

            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()


def main():
    """Run the Wumpus World visualization."""
    game = WumpusGameVisual()
    game.run()


if __name__ == "__main__":
    main()
