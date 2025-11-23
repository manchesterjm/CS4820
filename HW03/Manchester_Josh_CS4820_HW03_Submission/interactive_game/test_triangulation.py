"""
Quick test script to verify wumpus triangulation logic.

Runs multiple games and captures debug output to verify that 2+ stenches
correctly triangulate to confirm wumpus location.
"""

import random
from wumpus_agent import WumpusWorld, WumpusAgent, UnvisitedFirstStrategy

def run_test_game(seed: int, max_steps: int = 100) -> dict:
    """Run a single test game with given seed."""
    random.seed(seed)

    # Create world
    grid_size = 8
    world = WumpusWorld(grid_size=grid_size)

    # Add random obstacles
    available_positions = [
        (x, y) for x in range(1, grid_size + 1)
        for y in range(1, grid_size + 1)
        if (x, y) != (1, 1)
    ]

    # Add pits
    num_pits = max(3, int(grid_size * grid_size * 0.08))
    pit_positions = random.sample(available_positions, num_pits)
    for x, y in pit_positions:
        world.add_pit(x, y)

    # Add wumpus
    wumpus_positions = [pos for pos in available_positions if pos not in pit_positions]
    wumpus_pos = random.choice(wumpus_positions)
    world.add_wumpus(wumpus_pos[0], wumpus_pos[1])

    # Add gold
    gold_positions = [pos for pos in available_positions if pos not in pit_positions and pos != wumpus_pos]
    if gold_positions:
        gold_pos = random.choice(gold_positions)
        world.add_gold(gold_pos[0], gold_pos[1])

    # Create agent
    strategy = UnvisitedFirstStrategy()
    agent = WumpusAgent(grid_size=world.grid_size, strategy=strategy)

    print(f"\n{'='*60}")
    print(f"GAME SEED: {seed}")
    print(f"Grid: {world.grid_size}x{world.grid_size}")
    print(f"Wumpus at: {world.wumpus}")
    print(f"Pits at: {sorted(world.pits)}")
    print(f"Gold at: {world.gold}")
    print(f"{'='*60}\n")

    # Run game
    step_count = 0
    wumpus_confirmed = False

    while step_count < max_steps:
        step_count += 1
        step = agent.execute_step(world, step_count)

        # Check if wumpus was confirmed this step
        if agent._state.confirmed_wumpus_location and not wumpus_confirmed:
            wumpus_confirmed = True
            print(f"\n*** WUMPUS CONFIRMED AT STEP {step_count} ***")
            print(f"*** Location: {agent._state.confirmed_wumpus_location} ***")
            print(f"*** Actual location: {world.wumpus} ***")
            print(f"*** MATCH: {agent._state.confirmed_wumpus_location == world.wumpus} ***\n")

        if not agent._state.alive:
            print(f"\nAgent died at step {step_count}")
            break

        if agent._state.game_won:
            print(f"\nAgent won at step {step_count}!")
            break

        if step.chosen_move is None:
            print(f"\nAgent stopped: {step.reasoning}")
            break

    return {
        'seed': seed,
        'steps': step_count,
        'wumpus_actual': world.wumpus,
        'wumpus_confirmed': agent._state.confirmed_wumpus_location,
        'match': agent._state.confirmed_wumpus_location == world.wumpus if agent._state.confirmed_wumpus_location else None,
        'alive': agent._state.alive
    }


if __name__ == '__main__':
    print("\n" + "="*60)
    print("WUMPUS TRIANGULATION TEST")
    print("Testing if 2+ stenches correctly confirm wumpus location")
    print("="*60)

    # Run multiple games with different seeds
    results = []
    num_games = 5

    for i in range(num_games):
        seed = 1000 + i
        result = run_test_game(seed, max_steps=150)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    confirmed_count = sum(1 for r in results if r['wumpus_confirmed'] is not None)
    correct_count = sum(1 for r in results if r['match'] is True)

    for r in results:
        status = "CORRECT" if r['match'] else ("NOT_FOUND" if r['wumpus_confirmed'] is None else "WRONG")
        print(f"Seed {r['seed']}: Actual={r['wumpus_actual']}, "
              f"Confirmed={r['wumpus_confirmed']}, "
              f"Steps={r['steps']}, "
              f"Alive={r['alive']} [{status}]")

    print(f"\nGames with confirmed wumpus: {confirmed_count}/{num_games}")
    print(f"Correct confirmations: {correct_count}/{confirmed_count if confirmed_count > 0 else 1}")
    print("="*60)
