"""
Queens Environment - Place exactly one queen per row, column, and region
with no queens touching (including diagonally).

Multi-turn environment where:
- Queens can be placed (but not removed)
- X marks can be placed to eliminate positions
- Correct queen placements auto-mark invalid positions
- Incorrect placements end the game immediately
"""

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State

# === Constants ===

GAME_RULES = textwrap.dedent(
    """
    # Queens Puzzle Rules
    
    You are playing Queens, a placement puzzle on a grid with colored regions.
    
    ## IMPORTANT: No Tools or Code Allowed
    You must solve this puzzle using ONLY logical reasoning. 
    Do NOT use any tools, code execution, Python scripts, or external computation. 
    Think through the puzzle step by step using deduction only.
    
    ## Objective
    Place exactly ONE queen in:
    - Every row
    - Every column  
    - Every region (numbered area)
    
    Queens CANNOT touch each other - not horizontally, vertically, or diagonally (8-neighborhood).
    
    ## Grid Format
    - Grid shows region numbers (0-9)
    - 'Q' = placed queen
    - 'X' = marked as invalid (cannot place queen here)
    - Numbers/letters = region IDs (available cells)
    
    ## Action Format
    Respond with ONLY a valid JSON action:
    
    Place a queen (0-indexed row and column):
    {"place": [{"r": 0, "c": 3}]}
    
    Mark cells as invalid (X) (0-indexed row and column):
    {"mark": [{"r": 1, "c": 2}, {"r": 1, "c": 4}]}
    
    You can combine both in one turn (0-indexed row and column):
    {"place": [{"r": 0, "c": 3}], "mark": [{"r": 1, "c": 2}]}
    
    ## Rules
    - You CANNOT remove queens once placed
    - When you correctly place a queen, cells in the same row, column, region, and adjacent cells are auto-marked as X
    - If you place a queen incorrectly, the game ends immediately
    - If you mark X on a cell where a queen should go, the game ends immediately
    
    ## Strategy Tips
    - Use X marks to eliminate impossible positions
    - Look for regions/rows/columns with only one valid cell remaining
    - Place queens only when you're certain of the position, otherwise use X marks to eliminate impossible positions
    """
).strip()

FOLLOW_UP = textwrap.dedent(
    """
    Continue solving! Place queens or mark X's. Output the same JSON format.
    """
).strip()


# === Helper Functions ===


def parse_json_from_text(content: str) -> Optional[Dict]:
    """Parse JSON from text, handling LLM extra text."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None


# === Game Logic Classes ===


@dataclass
class QueensPuzzle:
    """Represents a Queens puzzle state."""

    rows: int
    cols: int
    regions: List[List[int]]  # 2D array of region IDs
    solution: Set[Tuple[int, int]]  # Correct queen positions
    queens: Set[Tuple[int, int]] = field(default_factory=set)  # Placed queens
    marks: Set[Tuple[int, int]] = field(default_factory=set)  # X marks

    def copy(self) -> "QueensPuzzle":
        """Create a deep copy of the puzzle state."""
        return QueensPuzzle(
            rows=self.rows,
            cols=self.cols,
            regions=[row[:] for row in self.regions],
            solution=set(self.solution),
            queens=set(self.queens),
            marks=set(self.marks),
        )

    def in_bounds(self, r: int, c: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_region(self, r: int, c: int) -> int:
        """Get the region ID at position (r, c)."""
        return self.regions[r][c]

    def is_solution_position(self, r: int, c: int) -> bool:
        """Check if (r, c) is a correct queen position."""
        return (r, c) in self.solution

    def is_available(self, r: int, c: int) -> bool:
        """Check if cell is available (not queen, not marked)."""
        return (r, c) not in self.queens and (r, c) not in self.marks

    def get_cells_to_auto_mark(self, r: int, c: int) -> Set[Tuple[int, int]]:
        """Get all cells that should be auto-marked when queen placed at (r,c)."""
        to_mark = set()
        region_id = self.get_region(r, c)

        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) == (r, c) or (row, col) in self.queens:
                    continue

                should_mark = False

                # Same row or column
                if row == r or col == c:
                    should_mark = True
                # Same region
                elif self.get_region(row, col) == region_id:
                    should_mark = True
                # Adjacent (8-neighborhood)
                elif max(abs(row - r), abs(col - c)) <= 1:
                    should_mark = True

                if should_mark:
                    to_mark.add((row, col))

        return to_mark

    def place_queen(self, r: int, c: int) -> Tuple[bool, str, Set[Tuple[int, int]]]:
        """
        Attempt to place a queen at (r, c).
        Returns (success, error_message, auto_marked_cells).
        """
        if not self.in_bounds(r, c):
            return False, f"Position ({r}, {c}) out of bounds", set()

        if (r, c) in self.queens:
            return False, f"Queen already at ({r}, {c})", set()

        if (r, c) in self.marks:
            return False, f"Position ({r}, {c}) is marked as X", set()

        # Check if this is a correct position
        if not self.is_solution_position(r, c):
            return False, f"Wrong position! ({r}, {c}) is not a valid queen location", set()

        # Place the queen
        self.queens.add((r, c))

        # Auto-mark related cells
        auto_marked = self.get_cells_to_auto_mark(r, c)
        self.marks.update(auto_marked)

        return True, "", auto_marked

    def place_mark(self, r: int, c: int) -> Tuple[bool, str]:
        """
        Attempt to place an X mark at (r, c).
        Returns (success, error_message).
        """
        if not self.in_bounds(r, c):
            return False, f"Position ({r}, {c}) out of bounds"

        if (r, c) in self.queens:
            return False, f"Cannot mark ({r}, {c}) - queen already there"

        if (r, c) in self.marks:
            return False, f"Position ({r}, {c}) already marked"

        # Check if this position should have a queen (wrong mark)
        if self.is_solution_position(r, c):
            return False, f"Wrong mark! ({r}, {c}) should have a queen"

        # Place the mark
        self.marks.add((r, c))
        return True, ""

    def is_solved(self) -> bool:
        """Check if the puzzle is completely solved."""
        return self.queens == self.solution

    def to_ascii(self) -> str:
        """Generate ASCII representation of the puzzle."""
        lines = [f"QUEENS {self.rows}x{self.cols}"]

        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if (r, c) in self.queens:
                    row_str.append("Q")
                elif (r, c) in self.marks:
                    row_str.append("X")
                else:
                    row_str.append(str(self.regions[r][c]))
            lines.append(" ".join(row_str))

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Generate JSON representation of the puzzle."""
        return {
            "game": "queens",
            "size": {"rows": self.rows, "cols": self.cols},
            "regions": self.regions,
            "queens": [{"r": r, "c": c} for r, c in sorted(self.queens)],
            "marks": [{"r": r, "c": c} for r, c in sorted(self.marks)],
            "queens_remaining": len(self.solution) - len(self.queens),
        }


# === Environment Class ===


class QueensEnv(MultiTurnEnv):
    """Multi-turn environment for the Queens puzzle."""

    def __init__(self, max_turns: int = 50, *args, **kwargs):
        self.max_turns = max_turns
        super().__init__(*args, **kwargs)

    async def setup_state(self, state: State, **kwargs) -> State:
        """Initialize puzzle-specific state."""
        puzzle_data = state["info"]["puzzle"]
        solution_data = state["info"]["solution"]

        # Convert solution to set of tuples
        solution_set = {(pos[0], pos[1]) for pos in solution_data}

        puzzle = QueensPuzzle(
            rows=puzzle_data["rows"],
            cols=puzzle_data["cols"],
            regions=puzzle_data["regions"],
            solution=solution_set,
            queens=set(),
            marks=set(),
        )

        state["puzzle"] = puzzle
        state["is_solved"] = False
        state["game_over"] = False
        state["game_over_reason"] = ""
        state["queens_placed"] = 0
        state["total_queens"] = len(solution_set)
        state["marks_placed"] = 0

        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check if the episode should end."""
        parent_done = await super().is_completed(messages, state, **kwargs)
        if parent_done:
            return True

        if state.get("is_solved", False):
            return True

        if state.get("game_over", False):
            return True

        assistant_count = len([m for m in messages if m["role"] == "assistant"])
        if self.max_turns > 0 and assistant_count >= self.max_turns:
            return True

        return False

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        """Process agent action and return environment feedback."""
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        turn_num = len(assistant_messages)

        if turn_num == 0:
            return [], state

        # Parse action
        last_content = assistant_messages[-1]["content"]
        parsed = parse_json_from_text(last_content)

        puzzle: QueensPuzzle = state["puzzle"]

        if parsed is None:
            response = {
                "valid": False,
                "reason": "No valid JSON found in response",
                "ascii": puzzle.to_ascii(),
                "json": puzzle.to_json(),
            }
            return [{"role": "user", "content": json.dumps(response)}], state

        # Process actions
        placements = parsed.get("place", [])
        markings = parsed.get("mark", [])

        if not placements and not markings:
            response = {
                "valid": False,
                "reason": "Action must contain 'place' or 'mark' field",
                "ascii": puzzle.to_ascii(),
                "json": puzzle.to_json(),
            }
            return [{"role": "user", "content": json.dumps(response)}], state

        results = []
        game_ended = False
        end_reason = ""

        # Process queen placements first
        for placement in placements:
            if game_ended:
                break

            r, c = placement.get("r", -1), placement.get("c", -1)
            success, error, auto_marked = puzzle.place_queen(r, c)

            if success:
                state["queens_placed"] = len(puzzle.queens)
                state["marks_placed"] = len(puzzle.marks)
                results.append({
                    "action": "place",
                    "r": r,
                    "c": c,
                    "success": True,
                    "auto_marked": len(auto_marked),
                })
            else:
                # Wrong placement - game over
                game_ended = True
                end_reason = error
                results.append({
                    "action": "place",
                    "r": r,
                    "c": c,
                    "success": False,
                    "reason": error,
                })

        # Process X markings
        for marking in markings:
            if game_ended:
                break

            r, c = marking.get("r", -1), marking.get("c", -1)
            success, error = puzzle.place_mark(r, c)

            if success:
                state["marks_placed"] = len(puzzle.marks)
                results.append({
                    "action": "mark",
                    "r": r,
                    "c": c,
                    "success": True,
                })
            else:
                # Check if this is a fatal error (marking where queen should be)
                if "should have a queen" in error:
                    game_ended = True
                    end_reason = error
                results.append({
                    "action": "mark",
                    "r": r,
                    "c": c,
                    "success": False,
                    "reason": error,
                })

        # Check for game over
        if game_ended:
            state["game_over"] = True
            state["game_over_reason"] = end_reason
            response = {
                "valid": False,
                "done": True,
                "game_over": True,
                "reason": end_reason,
                "results": results,
                "queens_count": len(puzzle.queens),
                "ascii": puzzle.to_ascii(),
                "json": puzzle.to_json(),
            }
            return [{"role": "user", "content": json.dumps(response)}], state

        # Check if solved
        if puzzle.is_solved():
            state["is_solved"] = True
            response = {
                "valid": True,
                "done": True,
                "solved": True,
                "results": results,
                "queens_count": len(puzzle.queens),
                "message": "Puzzle solved!",
                "ascii": puzzle.to_ascii(),
                "json": puzzle.to_json(),
            }
            return [{"role": "user", "content": json.dumps(response)}], state

        # Continue playing
        any_success = any(r.get("success", False) for r in results)
        response = {
            "valid": any_success,
            "done": False,
            "results": results,
            "queens_count": len(puzzle.queens),
            "queens_remaining": state["total_queens"] - len(puzzle.queens),
            "marks_count": len(puzzle.marks),
            "ascii": puzzle.to_ascii(),
            "json": puzzle.to_json(),
        }

        if any_success:
            follow_up = f"Actions valid.\n\n{FOLLOW_UP}\n\n{json.dumps(response)}"
        else:
            follow_up = json.dumps(response)

        return [{"role": "user", "content": follow_up}], state


# === Reward Functions ===


def reward_queens(state: State, **kwargs) -> float:
    """Reward based on correctly placed queens / total queens."""
    total_queens = state.get("total_queens", 1)
    queens_placed = state.get("queens_placed", 0)
    return queens_placed / total_queens

# === Dataset Building ===


def build_dataset(
    dataset_name: str = "djdumpling/spatial_reasoning",
    dataset_file: str = "queens.json",
) -> Dataset:
    """Load Queens puzzles from HuggingFace dataset."""

    # Load from HuggingFace
    hf_dataset = load_dataset(dataset_name, data_files=dataset_file, split="train")

    data = []
    for row in hf_dataset:
        puzzle_id = row.get("puzzle_id", "unknown")
        rows = row["rows"]
        cols = row["cols"]
        regions = row["regions"]
        solution = row["solution"]

        # Create puzzle for initial observation
        solution_set = {(pos[0], pos[1]) for pos in solution}
        temp_puzzle = QueensPuzzle(
            rows=rows,
            cols=cols,
            regions=regions,
            solution=solution_set,
            queens=set(),
            marks=set(),
        )

        initial_obs = {"ascii": temp_puzzle.to_ascii(), "json": temp_puzzle.to_json()}

        prompt_content = (
            f"{GAME_RULES}\n\n"
            f"## Initial State\n```\n{initial_obs['ascii']}\n```\n\n"
            f"JSON State:\n```json\n{json.dumps(initial_obs['json'], indent=2)}\n```\n\n"
        )

        data.append({
            "prompt": [{"role": "user", "content": prompt_content}],
            "answer": json.dumps({"solution": solution}),
            "task": "queens",
            "info": {
                "puzzle_id": puzzle_id,
                "puzzle": {
                    "rows": rows,
                    "cols": cols,
                    "regions": regions,
                },
                "solution": solution,
            },
        })

    return Dataset.from_list(data)


# === Environment Loading ===


def load_environment(
    dataset_name: str = "djdumpling/spatial_reasoning",
    dataset_file: str = "queens.json",
    max_turns: int = 50,
) -> vf.Environment:
    """Load the Queens puzzle environment from HuggingFace dataset."""

    dataset = build_dataset(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
    )

    rubric = Rubric(
        funcs=[reward_queens],
        weights=[1.0],
    )

    env = QueensEnv(
        max_turns=max_turns,
        dataset=dataset,
        rubric=rubric,
    )

    return env
