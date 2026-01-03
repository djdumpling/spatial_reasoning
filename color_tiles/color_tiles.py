"""
Color Tiles Game Simulator

Game Rules:
- 15x23 grid with 200 colored tiles (10 colors, 20 of each)
- Colors represented as digits 0-9
- An action: pick an EMPTY cell
- Find the first non-empty cell in each of the 4 cardinal directions (up, right, down, left)
- Any tiles among those 4 that share the same color get cleared
- A legal move requires at least 2 tiles of the same color to clear
"""

import random
from collections import Counter
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


ROWS = 15
COLS = 23
NUM_COLORS = 10
TILES_PER_COLOR = 20
TOTAL_TILES = NUM_COLORS * TILES_PER_COLOR  # 200


@dataclass
class GameState:
    """Represents the state of a Color Tiles game."""
    grid: np.ndarray  # -1 for empty, 0-9 for colors
    tiles_remaining: int
    
    @classmethod
    def create(cls, seed: int) -> "GameState":
        """Create a new game with the given random seed."""
        rng = random.Random(seed)
        
        # Create grid filled with -1 (empty)
        grid = np.full((ROWS, COLS), -1, dtype=np.int8)
        
        # Generate all tile colors (20 of each color 0-9)
        tiles = []
        for color in range(NUM_COLORS):
            tiles.extend([color] * TILES_PER_COLOR)
        
        # Shuffle tiles
        rng.shuffle(tiles)
        
        # Get all cell positions and shuffle them
        all_positions = [(r, c) for r in range(ROWS) for c in range(COLS)]
        rng.shuffle(all_positions)
        
        # Place 200 tiles on the grid
        for i, (r, c) in enumerate(all_positions[:TOTAL_TILES]):
            grid[r, c] = tiles[i]
        
        return cls(grid=grid, tiles_remaining=TOTAL_TILES)
    
    def get_first_tile_in_direction(self, row: int, col: int, dr: int, dc: int) -> tuple[int, int, int] | None:
        """
        Get the first non-empty tile in a direction from (row, col).
        Returns (row, col, color) or None if no tile found.
        """
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS:
            if self.grid[r, c] != -1:
                return (r, c, self.grid[r, c])
            r += dr
            c += dc
        return None
    
    def get_tiles_in_cardinal_directions(self, row: int, col: int) -> list[tuple[int, int, int]]:
        """
        Get the first non-empty tile in each of the 4 cardinal directions.
        Returns list of (row, col, color) tuples.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        tiles = []
        for dr, dc in directions:
            tile = self.get_first_tile_in_direction(row, col, dr, dc)
            if tile is not None:
                tiles.append(tile)
        return tiles
    
    def get_clearable_tiles(self, row: int, col: int) -> list[tuple[int, int]]:
        """
        Get the tiles that would be cleared by clicking on (row, col).
        Returns list of (row, col) positions to clear.
        """
        if self.grid[row, col] != -1:
            return []  # Can only click on empty cells
        
        tiles = self.get_tiles_in_cardinal_directions(row, col)
        if len(tiles) < 2:
            return []
        
        # Count colors
        color_counts = Counter(t[2] for t in tiles)
        
        # Find colors that appear more than once
        matching_colors = {color for color, count in color_counts.items() if count >= 2}
        
        if not matching_colors:
            return []
        
        # Return positions of all tiles with matching colors
        return [(r, c) for r, c, color in tiles if color in matching_colors]
    
    def get_legal_moves(self) -> list[tuple[int, int, list[tuple[int, int]]]]:
        """
        Get all legal moves.
        Returns list of (row, col, tiles_to_clear) tuples.
        """
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r, c] == -1:  # Empty cell
                    clearable = self.get_clearable_tiles(r, c)
                    if clearable:
                        moves.append((r, c, clearable))
        return moves
    
    def apply_move(self, row: int, col: int, tiles_to_clear: list[tuple[int, int]]) -> int:
        """
        Apply a move: clear the specified tiles.
        Returns the number of tiles cleared.
        """
        for r, c in tiles_to_clear:
            self.grid[r, c] = -1
        
        cleared = len(tiles_to_clear)
        self.tiles_remaining -= cleared
        return cleared


def play_game(seed: int, strategy: str = "random", strategy_seed: int | None = None) -> tuple[int, int]:
    """
    Play a game with the given seed and strategy.
    Returns (tiles_cleared, moves_made).
    
    Args:
        seed: Random seed for board generation
        strategy: Strategy to use ("random", "avoid_3")
        strategy_seed: Optional separate seed for strategy randomness (defaults to seed + 1000000)
    """
    game = GameState.create(seed)
    rng = random.Random(strategy_seed if strategy_seed is not None else seed + 1000000)
    
    initial_tiles = game.tiles_remaining
    total_cleared = 0
    moves_made = 0
    
    while True:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break
        
        if strategy == "random":
            # Pick a random legal move
            row, col, tiles_to_clear = rng.choice(legal_moves)
        elif strategy == "avoid_3":
            # Filter out moves that clear exactly 3 tiles
            # Only allow moves that clear 2 or 4 tiles
            filtered_moves = [(r, c, tiles) for r, c, tiles in legal_moves if len(tiles) != 3]
            if filtered_moves:
                row, col, tiles_to_clear = rng.choice(filtered_moves)
            else:
                # If no valid moves under this policy, fall back to any legal move
                row, col, tiles_to_clear = rng.choice(legal_moves)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        cleared = game.apply_move(row, col, tiles_to_clear)
        total_cleared += cleared
        moves_made += 1
    
    return total_cleared, moves_made


def compute_statistics(results: list[int]) -> dict:
    """Compute statistics for the results."""
    arr = np.array(results)
    
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    return {
        "count": len(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "q1": q1,
        "median": np.median(arr),
        "q3": q3,
        "max": np.max(arr),
        "iqr": iqr,
        "perfect_clears": np.sum(arr == TOTAL_TILES),
    }


def main():
    """Run the simulation."""
    num_seeds = 200
    attempts_per_seed = 5
    strategy = "avoid_3"  # Options: "random", "avoid_3"
    
    print(f"Color Tiles Simulation")
    print(f"=" * 50)
    print(f"Grid size: {ROWS}x{COLS} ({ROWS * COLS} cells)")
    print(f"Tiles: {TOTAL_TILES} ({NUM_COLORS} colors, {TILES_PER_COLOR} each)")
    print(f"Seeds: {num_seeds}")
    print(f"Attempts per seed: {attempts_per_seed} (taking max)")
    print(f"Strategy: {strategy}")
    print()
    
    tiles_cleared_list = []
    moves_made_list = []
    
    for seed in tqdm(range(num_seeds), desc="Simulating games"):
        best_cleared = 0
        best_moves = 0
        for attempt in range(attempts_per_seed):
            # Use different strategy seeds for each attempt
            strategy_seed = seed * 1000000 + attempt
            tiles_cleared, moves_made = play_game(seed, strategy, strategy_seed)
            if tiles_cleared > best_cleared:
                best_cleared = tiles_cleared
                best_moves = moves_made
        tiles_cleared_list.append(best_cleared)
        moves_made_list.append(best_moves)
    
    print()
    print("=" * 50)
    print("RESULTS: Tiles Cleared")
    print("=" * 50)
    
    stats = compute_statistics(tiles_cleared_list)
    
    print(f"  Total games:     {stats['count']}")
    print(f"  Mean:            {stats['mean']:.2f}")
    print(f"  Std Dev:         {stats['std']:.2f}")
    print(f"  Min:             {stats['min']}")
    print(f"  Q1 (25%):        {stats['q1']:.2f}")
    print(f"  Median (50%):    {stats['median']:.2f}")
    print(f"  Q3 (75%):        {stats['q3']:.2f}")
    print(f"  Max:             {stats['max']}")
    print(f"  IQR:             {stats['iqr']:.2f}")
    print(f"  Perfect clears:  {stats['perfect_clears']} ({100*stats['perfect_clears']/stats['count']:.2f}%)")
    
    print()
    print("=" * 50)
    print("RESULTS: Moves Made")
    print("=" * 50)
    
    move_stats = compute_statistics(moves_made_list)
    
    print(f"  Mean:            {move_stats['mean']:.2f}")
    print(f"  Std Dev:         {move_stats['std']:.2f}")
    print(f"  Min:             {move_stats['min']}")
    print(f"  Max:             {move_stats['max']}")
    
    # Distribution of tiles cleared
    print()
    print("=" * 50)
    print("DISTRIBUTION (Tiles Cleared)")
    print("=" * 50)
    
    bins = [0, 50, 100, 150, 180, 190, 195, 200]
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            count = sum(1 for x in tiles_cleared_list if low <= x <= high)
            print(f"  {low:3d}-{high:3d}: {count:4d} ({100*count/num_seeds:.1f}%)")
        else:
            count = sum(1 for x in tiles_cleared_list if low <= x < high)
            print(f"  {low:3d}-{high-1:3d}: {count:4d} ({100*count/num_seeds:.1f}%)")


if __name__ == "__main__":
    main()

