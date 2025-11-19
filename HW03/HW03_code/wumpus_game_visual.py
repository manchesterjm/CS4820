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
BOARD_SIZE = 600
PANEL_WIDTH = 500
WINDOW_WIDTH = BOARD_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = 700
INFO_HEIGHT = 100

# Grid settings
GRID_SIZE = 4
CELL_SIZE = BOARD_SIZE // GRID_SIZE

# Fonts
TITLE_FONT = pygame.font.Font(None, 32)
HEADER_FONT = pygame.font.Font(None, 24)
TEXT_FONT = pygame.font.Font(None, 20)
SMALL_FONT = pygame.font.Font(None, 16)
CELL_FONT = pygame.font.Font(None, 40)


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
        self.max_steps = 10
        self.steps_history = []
        self.game_over = False
        self.auto_play = False
        self.auto_play_delay = 1.5  # seconds between auto steps
        self.last_auto_step = 0

        # Animation state
        self.animating = False
        self.anim_start_pos = None
        self.anim_end_pos = None
        self.anim_progress = 0.0
        self.anim_speed = 0.05

        # Initialize game
        self.reset_game()

    def reset_game(self):
        """Reset the game to initial state."""
        # Create world with obstacles
        self.world = WumpusWorld(grid_size=GRID_SIZE)
        self.world.add_pit(3, 1)
        self.world.add_pit(3, 3)
        self.world.add_pit(4, 4)
        self.world.add_wumpus(1, 3)

        # Create agent
        self.agent = WumpusAgent(grid_size=GRID_SIZE, strategy=UnvisitedFirstStrategy())

        # Reset state
        self.current_step = 0
        self.steps_history = []
        self.game_over = False
        self.animating = False

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

        # Draw coordinate labels
        for i in range(1, GRID_SIZE + 1):
            # X coordinates (bottom)
            label = SMALL_FONT.render(str(i), True, BLACK)
            x, y = self.grid_to_screen(i, 1)
            self.screen.blit(label, (x - 5, BOARD_SIZE + 5))

            # Y coordinates (left)
            label = SMALL_FONT.render(str(i), True, BLACK)
            x, y = self.grid_to_screen(1, i)
            self.screen.blit(label, (5, y - 10))

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

        # Highlight visited cells
        if pos in visited:
            pygame.draw.rect(self.screen, (220, 255, 220), cell_rect)

        # Draw starting position marker
        if pos == (1, 1):
            pygame.draw.rect(self.screen, (200, 255, 200), cell_rect)
            start_label = SMALL_FONT.render("START", True, DARK_GREEN)
            self.screen.blit(start_label, (screen_x - 20, screen_y + 50))

        # Show actual world contents (pits and wumpus) - semi-transparent if not visited
        if pos in self.world.pits:
            alpha = 255 if pos in visited else 80
            pit_surf = pygame.Surface((CELL_SIZE - 20, CELL_SIZE - 20), pygame.SRCALPHA)
            pygame.draw.circle(pit_surf, (*BLACK, alpha), (CELL_SIZE // 2 - 10, CELL_SIZE // 2 - 10), 30)
            self.screen.blit(pit_surf, (screen_x - CELL_SIZE // 2 + 10, screen_y - CELL_SIZE // 2 + 10))
            label = CELL_FONT.render("P", True, (*BLACK, alpha))
            self.screen.blit(label, (screen_x - 12, screen_y - 20))

        if self.world.wumpus and pos == self.world.wumpus:
            alpha = 255 if pos in visited else 80
            wumpus_surf = pygame.Surface((CELL_SIZE - 20, CELL_SIZE - 20), pygame.SRCALPHA)
            pygame.draw.polygon(
                wumpus_surf, (*RED, alpha),
                [(CELL_SIZE // 2 - 10, 10), (CELL_SIZE // 2 - 30, CELL_SIZE - 30), (CELL_SIZE // 2 + 10, CELL_SIZE - 30)]
            )
            self.screen.blit(wumpus_surf, (screen_x - CELL_SIZE // 2 + 10, screen_y - CELL_SIZE // 2 + 10))
            label = CELL_FONT.render("W", True, (*RED, alpha))
            self.screen.blit(label, (screen_x - 15, screen_y - 20))

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
                label = TEXT_FONT.render(percept_str, True, BLUE)
                self.screen.blit(label, (screen_x - 10, screen_y - 40))

    def draw_agent(self, position: Tuple[int, int]):
        """Draw the agent at given position."""
        screen_x, screen_y = self.grid_to_screen(position[0], position[1])

        # Draw agent as a circle with direction indicator
        pygame.draw.circle(self.screen, ORANGE, (screen_x, screen_y), 20)
        pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), 20, 2)

        # Draw eyes
        pygame.draw.circle(self.screen, BLACK, (screen_x - 7, screen_y - 5), 3)
        pygame.draw.circle(self.screen, BLACK, (screen_x + 7, screen_y - 5), 3)

        # Draw smile
        pygame.draw.arc(self.screen, BLACK, (screen_x - 10, screen_y - 5, 20, 15), 3.14, 6.28, 2)

    def draw_animated_agent(self):
        """Draw agent with animation between positions."""
        if not self.animating or self.anim_start_pos is None or self.anim_end_pos is None:
            return

        # Interpolate position
        start_screen = self.grid_to_screen(*self.anim_start_pos)
        end_screen = self.grid_to_screen(*self.anim_end_pos)

        current_x = start_screen[0] + (end_screen[0] - start_screen[0]) * self.anim_progress
        current_y = start_screen[1] + (end_screen[1] - start_screen[1]) * self.anim_progress

        # Draw agent at interpolated position
        pygame.draw.circle(self.screen, ORANGE, (int(current_x), int(current_y)), 20)
        pygame.draw.circle(self.screen, BLACK, (int(current_x), int(current_y)), 20, 2)
        pygame.draw.circle(self.screen, BLACK, (int(current_x) - 7, int(current_y) - 5), 3)
        pygame.draw.circle(self.screen, BLACK, (int(current_x) + 7, int(current_y) - 5), 3)
        pygame.draw.arc(self.screen, BLACK, (int(current_x) - 10, int(current_y) - 5, 20, 15), 3.14, 6.28, 2)

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
        if self.game_over:
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
        legend = "Legend: P=Pit, W=Wumpus, B=Breeze, S=Stench"
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

        # Check if agent can continue
        if step.chosen_move is None:
            self.game_over = True
        else:
            # Start animation
            self.animating = True
            self.anim_start_pos = step.position
            self.anim_end_pos = step.chosen_move
            self.anim_progress = 0.0

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
