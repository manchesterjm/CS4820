"""
Test case to reproduce wumpus triangulation bug from log 20251119_170152.

Wumpus at (4, 6)
Stenches at (5, 6) and (4, 7)
Agent visited (5, 7) with no stench
Should confirm wumpus at (4, 6)
"""

from wumpus_agent import WumpusWorld, WumpusAgent, UnvisitedFirstStrategy

# Create world matching the log
world = WumpusWorld(grid_size=8)
world.add_pit(2, 1)
world.add_pit(2, 7)
world.add_pit(3, 7)
world.add_pit(4, 5)
world.add_pit(6, 8)
world.add_wumpus(4, 6)
world.add_gold(6, 2)

# Create agent
strategy = UnvisitedFirstStrategy()
agent = WumpusAgent(grid_size=8, strategy=strategy)

print("=== TEST: Wumpus Triangulation Bug ===")
print(f"Wumpus actual location: {world.wumpus}")
print(f"Pits: {sorted(world.pits)}")
print()

# Run game until we get 2 stenches
step_count = 0
max_steps = 85

while step_count < max_steps:
    step_count += 1
    step = agent.execute_step(world, step_count)

    # Show when we get stenches
    if step.percept.stench:
        print(f"Step {step_count}: At {step.position}, STENCH")

    # Show wumpus confirmation
    if agent._state.confirmed_wumpus_location:
        print(f"\n*** WUMPUS CONFIRMED AT STEP {step_count} ***")
        print(f"*** Confirmed: {agent._state.confirmed_wumpus_location} ***")
        print(f"*** Actual: {world.wumpus} ***")
        print(f"*** Match: {agent._state.confirmed_wumpus_location == world.wumpus} ***\n")
        break

    if not agent._state.alive:
        print(f"Agent died at step {step_count}")
        break

    if step.chosen_move is None:
        print(f"\nAgent stopped at step {step_count}: {step.reasoning}")
        break

print(f"\nFinal result:")
print(f"  Wumpus confirmed: {agent._state.confirmed_wumpus_location}")
print(f"  Wumpus actual: {world.wumpus}")
print(f"  Match: {agent._state.confirmed_wumpus_location == world.wumpus if agent._state.confirmed_wumpus_location else False}")
