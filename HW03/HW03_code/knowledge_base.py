"""
Knowledge Base data structures for propositional and Horn clause logic.

This module provides KB representations supporting:
- Propositional logic (arbitrary sentences)
- Horn clauses (restricted format for efficient inference)
- Tell/Ask interface for KB agents

Based on Russell & Norvig Chapter 7: Logical Agents
- Section 7.1: Knowledge-Based Agents
- Section 7.4: Propositional Logic
- Section 7.5.3: Horn Clauses and Definite Clauses

Author: Josh Manchester
Course: CS 4820/5820 - Artificial Intelligence
Institution: University of Colorado Colorado Springs
"""

from typing import List, Set, Tuple


class PropositionalKB:
    """
    Knowledge base for general propositional logic.

    Supports arbitrary propositional sentences with operators:
    - Negation: ¬ or 'not'
    - Conjunction: ∧ or '&' or 'and'
    - Disjunction: ∨ or '|' or 'or'
    - Implication: ⇒ or '=>'
    - Biconditional: ⇔ or '<=>'

    Based on Russell & Norvig Section 7.4.

    Attributes:
        sentences: List of propositional sentences in string form
    """

    def __init__(self):
        """Initialize empty propositional KB."""
        self.sentences: List[str] = []

    def tell(self, sentence: str) -> None:
        """
        Add a sentence to the knowledge base.

        Args:
            sentence: Propositional logic sentence (e.g., "P => Q")
        """
        if sentence and sentence.strip():
            self.sentences.append(sentence.strip())

    def sentences_list(self) -> List[str]:
        """Return all sentences in KB."""
        return self.sentences.copy()

    def __len__(self) -> int:
        """Number of sentences in KB."""
        return len(self.sentences)

    def __str__(self) -> str:
        """String representation of KB."""
        if not self.sentences:
            return "KB: (empty)"
        return "KB:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.sentences))


class HornKB:
    """
    Knowledge base for Horn clauses.

    A Horn clause has the form:
        P1 ∧ P2 ∧ ... ∧ Pn ⇒ Q

    Where P1, ..., Pn are positive literals (premises) and Q is a positive
    literal (conclusion). Facts are Horn clauses with no premises.

    Based on Russell & Norvig Section 7.5.3.

    Horn clauses allow for efficient inference:
    - Forward chaining: O(n) time where n = KB size
    - Backward chaining: O(n) time for well-structured KBs

    Why Horn clauses?
    - Tractable inference (polynomial time)
    - Expressive enough for many real-world problems
    - Used in logic programming (Prolog) and expert systems

    Attributes:
        facts: Set of ground positive literals
        rules: List of (premises, conclusion) tuples
    """

    def __init__(self):
        """Initialize empty Horn clause KB."""
        self.facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str]] = []

    def tell_fact(self, fact: str) -> None:
        """
        Add a ground fact to the KB.

        A fact is a Horn clause with no premises (atomic sentence).

        Args:
            fact: Positive literal (e.g., "P", "B_1_1", "not_P_2_1")

        Note: Use naming convention "not_X" for negated facts to stay
        within Horn clause restrictions (only positive literals).
        """
        if fact and fact.strip():
            self.facts.add(fact.strip())

    def tell_rule(self, premises: List[str], conclusion: str) -> None:
        """
        Add a Horn clause rule to the KB.

        Rule format: premises[0] ∧ premises[1] ∧ ... ⇒ conclusion

        Args:
            premises: List of positive literals (body of rule)
            conclusion: Single positive literal (head of rule)

        Example:
            tell_rule(['A', 'B'], 'C')  # A ∧ B ⇒ C
            tell_rule(['C'], 'D')        # C ⇒ D
        """
        if conclusion and conclusion.strip():
            # Filter out empty premises
            valid_premises = [p.strip() for p in premises if p and p.strip()]
            self.rules.append((valid_premises, conclusion.strip()))

    def get_facts(self) -> Set[str]:
        """Return all known facts."""
        return self.facts.copy()

    def get_rules(self) -> List[Tuple[List[str], str]]:
        """Return all rules."""
        return self.rules.copy()

    def __len__(self) -> int:
        """Total KB size (facts + rules)."""
        return len(self.facts) + len(self.rules)

    def __str__(self) -> str:
        """String representation of KB."""
        lines = []
        if self.facts:
            lines.append("Facts:")
            for fact in sorted(self.facts):
                lines.append(f"  {fact}")
        if self.rules:
            lines.append("Rules:")
            for i, (premises, conclusion) in enumerate(self.rules, 1):
                if premises:
                    premise_str = " AND ".join(premises)
                    lines.append(f"  {i}. {premise_str} => {conclusion}")
                else:
                    lines.append(f"  {i}. {conclusion}")
        if not lines:
            return "HornKB: (empty)"
        return "\n".join(lines)


class WumpusKB(HornKB):
    """
    Specialized knowledge base for Wumpus World.

    Extends HornKB with Wumpus-specific utilities:
    - Percept handling (breeze, stench)
    - Grid coordinate tracking
    - Safety inference

    Based on Russell & Norvig Section 7.7: Agents Based on Propositional Logic.

    Wumpus World percepts:
    - Breeze: Felt in squares adjacent to pit
    - Stench: Smelled in squares adjacent to Wumpus
    - Glitter: Visible when gold is in same square
    - Bump: Felt when agent walks into wall
    - Scream: Heard when Wumpus is killed

    Propositional symbols:
    - P_x_y: Pit at location (x, y)
    - W_x_y: Wumpus at location (x, y)
    - B_x_y: Breeze perceived at (x, y)
    - S_x_y: Stench perceived at (x, y)
    - OK_x_y: Location (x, y) is safe (no pit, no Wumpus)

    Attributes:
        grid_size: Dimension of square grid (default 4 for 4x4)
    """

    def __init__(self, grid_size: int = 4):
        """
        Initialize Wumpus World KB.

        Args:
            grid_size: Grid dimension (4 for 4x4 world)
        """
        super().__init__()
        self.grid_size = grid_size

    def add_percept(self, x: int, y: int, breeze: bool, stench: bool) -> None:
        """
        Add percepts from location (x, y) to KB.

        Args:
            x: X coordinate (1-indexed)
            y: Y coordinate (1-indexed)
            breeze: True if breeze felt at (x, y)
            stench: True if stench smelled at (x, y)
        """
        if breeze:
            self.tell_fact(f"B_{x}_{y}")
        else:
            self.tell_fact(f"not_B_{x}_{y}")

        if stench:
            self.tell_fact(f"S_{x}_{y}")
        else:
            self.tell_fact(f"not_S_{x}_{y}")

    def add_wumpus_rules(self) -> None:
        """
        Add Wumpus World inference rules to KB.

        Rules for breeze/pit relationship:
        - If no breeze at (x,y), then no pits in adjacent cells
        - Implemented as: not_B_x_y => not_P_u_v for each neighbor (u,v)

        Rules for stench/Wumpus relationship:
        - If no stench at (x,y), then no Wumpus in adjacent cells
        - Implemented as: not_S_x_y => not_W_u_v for each neighbor (u,v)

        Based on Russell & Norvig Section 7.7, Figure 7.20.
        """
        # For each cell in grid
        for x in range(1, self.grid_size + 1):
            for y in range(1, self.grid_size + 1):
                neighbors = self._get_neighbors(x, y)

                # No breeze => no pits in neighbors
                for nx, ny in neighbors:
                    self.tell_rule([f"not_B_{x}_{y}"], f"not_P_{nx}_{ny}")

                # No stench => no Wumpus in neighbors
                for nx, ny in neighbors:
                    self.tell_rule([f"not_S_{x}_{y}"], f"not_W_{nx}_{ny}")

                # If cell is known safe (visited and alive), no pit there
                # This is added dynamically as agent moves

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighbors of (x, y) within grid boundaries.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of (x, y) tuples for adjacent cells

        Neighbors are in 4 directions: up, down, left, right.
        Diagonal moves not allowed in Wumpus World.
        """
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def mark_safe(self, x: int, y: int) -> None:
        """
        Mark location (x, y) as safe (visited and survived).

        Adds facts: not_P_x_y and not_W_x_y

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.tell_fact(f"not_P_{x}_{y}")
        self.tell_fact(f"not_W_{x}_{y}")

    def get_probable_pits(self) -> Set[Tuple[int, int]]:
        """
        Find cells that probably contain pits based on breeze observations.

        Returns cells adjacent to breezes that aren't proven safe.
        """
        probable_pits = set()

        # Find all cells with breezes
        breeze_cells = [fact for fact in self.facts if fact.startswith("B_")]

        for breeze_fact in breeze_cells:
            # Parse "B_x_y" to get coordinates
            parts = breeze_fact.split("_")
            if len(parts) == 3:
                x, y = int(parts[1]), int(parts[2])
                neighbors = self._get_neighbors(x, y)

                # Neighbors of breeze cells are probable pit locations
                # unless proven safe
                for nx, ny in neighbors:
                    if f"not_P_{nx}_{ny}" not in self.facts:
                        probable_pits.add((nx, ny))

        return probable_pits

    def get_probable_wumpus(self) -> Set[Tuple[int, int]]:
        """
        Find cells that probably contain Wumpus based on stench observations.

        Returns cells adjacent to stenches that aren't proven safe.
        """
        probable_wumpus = set()

        # Find all cells with stenches
        stench_cells = [fact for fact in self.facts if fact.startswith("S_")]

        for stench_fact in stench_cells:
            # Parse "S_x_y" to get coordinates
            parts = stench_fact.split("_")
            if len(parts) == 3:
                x, y = int(parts[1]), int(parts[2])
                neighbors = self._get_neighbors(x, y)

                # Neighbors of stench cells are probable wumpus locations
                # unless proven safe
                for nx, ny in neighbors:
                    if f"not_W_{nx}_{ny}" not in self.facts:
                        probable_wumpus.add((nx, ny))

        return probable_wumpus

    def get_confirmed_pits(self) -> Set[Tuple[int, int]]:
        """
        Find cells that MUST contain pits using logical deduction.

        Uses constraint satisfaction:
        - If a cell has breeze and all neighbors but one are safe,
          the pit MUST be in the remaining neighbor (definite clause).
        - Find intersection of possible pit locations from multiple breezes.

        Returns cells that are logically confirmed to have pits.
        """
        confirmed_pits = set()

        # Find all cells with breezes
        breeze_cells = []
        for fact in self.facts:
            if fact.startswith("B_") and not fact.startswith("not_B"):
                parts = fact.split("_")
                if len(parts) == 3:
                    x, y = int(parts[1]), int(parts[2])
                    breeze_cells.append((x, y))

        # For each breeze cell, find possible pit locations
        for bx, by in breeze_cells:
            neighbors = self._get_neighbors(bx, by)
            possible_pits = []

            for nx, ny in neighbors:
                # If we haven't proven this cell safe, it's a possible pit location
                if f"not_P_{nx}_{ny}" not in self.facts:
                    possible_pits.append((nx, ny))

            # Definite clause: if exactly ONE neighbor could have pit, it MUST be there
            if len(possible_pits) == 1:
                confirmed_pits.add(possible_pits[0])

        # Triangulation: Find pits that appear in ALL possible sets from multiple breezes
        # If we have multiple breeze observations that narrow down to a single cell
        if len(breeze_cells) >= 2:
            # Build possibility sets for each breeze
            possibility_sets = []
            for bx, by in breeze_cells:
                neighbors = self._get_neighbors(bx, by)
                possible = set()
                for nx, ny in neighbors:
                    if f"not_P_{nx}_{ny}" not in self.facts:
                        possible.add((nx, ny))
                if possible:
                    possibility_sets.append(possible)

            # Find cells that appear in multiple possibility sets
            # If a cell is the ONLY common element in intersecting sets, it's confirmed
            for i, set1 in enumerate(possibility_sets):
                for j, set2 in enumerate(possibility_sets[i+1:], i+1):
                    intersection = set1.intersection(set2)
                    # If intersection has exactly one cell, that's a confirmed pit
                    if len(intersection) == 1:
                        confirmed_pits.add(intersection.pop())

        return confirmed_pits


# Utility functions for parsing propositional logic

def extract_symbols(sentence: str) -> Set[str]:
    """
    Extract all propositional symbols from a sentence.

    Args:
        sentence: Propositional logic sentence

    Returns:
        Set of symbol names

    Example:
        extract_symbols("P => Q")  # {'P', 'Q'}
        extract_symbols("(A & B) | C")  # {'A', 'B', 'C'}
    """
    # Remove operators and parentheses, split on whitespace
    symbols = set()
    # Replace operators with spaces
    cleaned = sentence
    for op in ['=>', '<=>', '&', '|', 'and', 'or', 'not', '(', ')', '¬', '∧', '∨', '⇒', '⇔']:
        cleaned = cleaned.replace(op, ' ')
    # Extract words
    for word in cleaned.split():
        if word.strip() and word not in ['True', 'False', 'true', 'false']:
            symbols.add(word.strip())
    return symbols


def is_horn_clause(premises: List[str], conclusion: str) -> bool:
    """
    Check if a rule is a valid Horn clause.

    A Horn clause has:
    - Conjunction of positive literals as premises
    - Single positive literal as conclusion

    Args:
        premises: List of premise literals
        conclusion: Conclusion literal

    Returns:
        True if valid Horn clause, False otherwise
    """
    # Check that conclusion is a single positive literal
    if not conclusion or ' ' in conclusion.strip() or '=>' in conclusion:
        return False

    # Check that all premises are positive literals
    for premise in premises:
        if not premise or ' ' in premise.strip() or '=>' in premise:
            return False

    return True


if __name__ == "__main__":
    # Example usage
    print("=== Propositional KB Example ===")
    prop_kb = PropositionalKB()
    prop_kb.tell("P => Q")
    prop_kb.tell("Q => R")
    prop_kb.tell("P")
    print(prop_kb)
    print()

    print("=== Horn KB Example ===")
    horn_kb = HornKB()
    horn_kb.tell_fact("A")
    horn_kb.tell_fact("B")
    horn_kb.tell_rule(["A", "B"], "C")
    horn_kb.tell_rule(["C"], "D")
    print(horn_kb)
    print()

    print("=== Wumpus KB Example ===")
    wumpus_kb = WumpusKB(grid_size=4)
    wumpus_kb.add_percept(1, 1, breeze=False, stench=False)
    wumpus_kb.add_wumpus_rules()
    wumpus_kb.mark_safe(1, 1)
    print(wumpus_kb)
    print(f"\nKB size: {len(wumpus_kb)} (facts + rules)")
