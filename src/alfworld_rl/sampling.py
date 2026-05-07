"""Stratified sampling of ALFWorld training games.

Both prepare_alfworld_data.py and alfworld_interaction.py call this so the
game_index -> game_file mapping is identical across processes.
"""

from __future__ import annotations

import random
from collections import defaultdict

ALFWORLD_TRAIN_TASK_TYPES: tuple[str, ...] = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)


def _classify(path: str) -> str | None:
    """Return the ALFWorld task-type prefix for a game-file path."""
    for t in ALFWORLD_TRAIN_TASK_TYPES:
        # paths look like .../train/<task_type>-Object-...-Recep-NN/trial_.../game.tw-pddl
        if f"/{t}-" in path:
            return t
    return None


def stratified_sample_train_games(
    game_files: list[str],
    n_per_type: int = 84,
    seed: int = 42,
) -> list[str]:
    """Group `game_files` by task-type prefix and sample n_per_type from each.

    Returns the union, sorted alphabetically (for cross-process determinism).
    Output size: at most `n_per_type * len(ALFWORLD_TRAIN_TASK_TYPES)`.
    With n_per_type=84 and 6 types: 504 games max.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    for path in game_files:
        t = _classify(path)
        if t is not None:
            by_type[t].append(path)

    rng = random.Random(seed)
    sampled: list[str] = []
    for t in ALFWORLD_TRAIN_TASK_TYPES:
        paths = sorted(by_type.get(t, []))
        if len(paths) <= n_per_type:
            sampled.extend(paths)
        else:
            sampled.extend(rng.sample(paths, n_per_type))
    return sorted(sampled)
