# Queens

Queens is a placement puzzle where the agent must place exactly one queen in every row, column, and colored region. Queens cannot touch each other—not horizontally, vertically, or diagonally (8-neighborhood).

Based on LinkedIn's daily Queens puzzle.

## Game Mechanics

- **Place queens** at positions you believe are correct
- **Mark X** on cells to eliminate impossible positions
- **Auto-marking**: When a queen is correctly placed, all cells in the same row, column, region, and adjacent cells are automatically marked as X
- **Strict validation**: Placing a queen incorrectly OR marking X where a queen should go ends the game immediately
- **No removal**: Queens cannot be removed once placed

## Quickstart

```bash
uv run vf-install queens
uv run vf-eval queens
```

Configure model and sampling:

```bash
uv run vf-eval queens \
  -m gpt-4.1 \
  -n 10 -r 3 \
  -a '{"dataset_name": "djdumpling/spatial_reasoning", "dataset_file": "queens.json", "max_turns": 50}'
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"djdumpling/spatial_reasoning"` | HuggingFace dataset name |
| `dataset_file` | str | `"queens.json"` | Data file within the dataset |
| `max_turns` | int | `50` | Maximum moves per episode |

## Observation Format

### ASCII
```
QUEENS 7x7
0 1 1 1 1 1 3
0 1 2 Q X X 3
0 X X X X 3 3
0 1 1 1 1 1 5
0 0 4 4 4 1 5
6 1 4 4 4 1 5
6 1 1 1 1 1 5
```

- Numbers/letters = region IDs (available cells)
- `Q` = placed queen
- `X` = marked as invalid

### JSON
```json
{
  "game": "queens",
  "size": {"rows": 7, "cols": 7},
  "regions": [[0, 1, 1, ...], ...],
  "queens": [{"r": 1, "c": 3}],
  "marks": [{"r": 1, "c": 4}, {"r": 1, "c": 5}, ...],
  "queens_remaining": 6
}
```

## Action Format

Place a queen:
```json
{"place": [{"r": 0, "c": 3}]}
```

Mark cells as invalid:
```json
{"mark": [{"r": 1, "c": 2}, {"r": 1, "c": 4}]}
```

Combined action:
```json
{"place": [{"r": 0, "c": 3}], "mark": [{"r": 2, "c": 1}]}
```

## Rules

1. Exactly one queen per row
2. Exactly one queen per column
3. Exactly one queen per region
4. No two queens can be adjacent (including diagonals - 8-neighborhood)
5. **Incorrect queen placement ends the game**
6. **Marking X where a queen belongs ends the game**
7. **Queens cannot be removed once placed**

## Metrics

| Metric | Weight | Meaning |
| ------ | ------ | ------- |
| `reward_solved` | `1.0` | 1.0 if puzzle completely solved, else 0.0 |
| `reward_progress` | `0.4` | Fraction of queens correctly placed (penalized if game over) |
| `reward_no_errors` | `0.3` | 1.0 if no errors made, 0.0 if game ended due to error |

## Data Source

Puzzles are loaded from HuggingFace: `djdumpling/spatial_reasoning/queens.json`
