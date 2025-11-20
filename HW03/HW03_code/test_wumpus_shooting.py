"""
Test case to verify wumpus shooting after getting gold.

Setup: Wumpus at (2, 7), Gold at (5, 2)
Expected: Agent should find gold, confirm wumpus, navigate to shooting position, kill wumpus
"""

from wumpus_agent import WumpusWorld, WumpusAgent, UnvisitedFirstStrategy

# Create world matching the scenario
world = WumpusWorld(grid_size=8)
world.add_pit(4, 1)
world.add_pit(5, 6)
world.add_pit(7, 2)
world.add_pit(8, 1)
world.add_pit(8, 6)
world.add_wumpus(2, 7)
world.add_gold(5, 2)

# Create agent
strategy = UnvisitedFirstStrategy()
agent = WumpusAgent(grid_size=8, strategy=strategy)

print("=== TEST: Wumpus Shooting After Getting Gold ===")
print(f"Wumpus location: {world.wumpus}")
print(f"Gold location: {world.gold}")
print(f"Pits: {sorted(world.pits)}")
print()

# Run game
step_count = 0
max_steps = 150

gold_found_step = None
wumpus_confirmed_step = None
wumpus_killed_step = None

while step_count < max_steps:
    step_count += 1
    step = agent.execute_step(world, step_count)

    # Track key events
    if agent._state.has_gold and gold_found_step is None:
        gold_found_step = step_count
        print(f"Step {step_count}: GOLD FOUND at {step.position}")

    if agent._state.confirmed_wumpus_location and wumpus_confirmed_step is None:
        wumpus_confirmed_step = step_count
        print(f"Step {step_count}: WUMPUS CONFIRMED at {agent._state.confirmed_wumpus_location}")

    if agent._state.wumpus_killed and wumpus_killed_step is None:
        wumpus_killed_step = step_count
        print(f"Step {step_count}: WUMPUS KILLED at position {step.position}")
        print(f"  Agent shot from: {step.position}")
        print(f"  Wumpus was at: {world.wumpus}")
        break

    if agent._state.game_won:
        print(f"\n*** GAME WON at step {step_count}! ***")
        break

    if not agent._state.alive:
        print(f"Step {step_count}: Agent died")
        break

    if step.chosen_move is None:
        print(f"Step {step_count}: Agent stopped - {step.reasoning}")
        break

print(f"\n=== RESULTS ===")
print(f"Gold found: Step {gold_found_step if gold_found_step else 'NOT FOUND'}")
print(f"Wumpus confirmed: Step {wumpus_confirmed_step if wumpus_confirmed_step else 'NOT CONFIRMED'}")
print(f"Wumpus killed: Step {wumpus_killed_step if wumpus_killed_step else 'NOT KILLED'}")
print(f"Game won: {agent._state.game_won}")
print(f"Total steps: {step_count}")

if not agent._state.wumpus_killed:
    print(f"\nFAILURE: Wumpus was not killed!")
    print(f"  Has gold: {agent._state.has_gold}")
    print(f"  Has arrow: {agent._state.has_arrow}")
    print(f"  Wumpus confirmed: {agent._state.confirmed_wumpus_location}")
else:
    print(f"\nSUCCESS: Wumpus killed and game won!")
