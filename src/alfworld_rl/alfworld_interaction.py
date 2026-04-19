"""ALFWorld interaction adapter for verl multi-turn GRPO.

Mirrors verl.interactions.gsm8k_interaction.Gsm8kInteraction. verl's AgentLoopWorker
calls start_interaction -> generate_response (once per assistant turn) -> finalize.

Test-run simplifications (to revisit for real training):
  - Each instance gets a fresh batched AlfredTWEnv. Reset cycles through games; we do
    not pin a specific game per data row yet, so within a GRPO group, different rollouts
    may see different games. Pipeline still runs, but GRPO advantages are noisier than
    they should be.
  - Action parsing is permissive: takes the last non-empty line, strips chain-of-thought
    wrappers if present.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

_DUMP_DIR = os.environ.get("ALFWORLD_TRAJECTORY_DUMP_DIR", "")
_DUMP_RUN_ID = os.environ.get("ALFWORLD_TRAJECTORY_DUMP_RUN", str(int(time.time())))


def _dump_event(instance_id: str, event: dict) -> None:
    if not _DUMP_DIR:
        return
    try:
        os.makedirs(f"{_DUMP_DIR}/{_DUMP_RUN_ID}", exist_ok=True)
        path = f"{_DUMP_DIR}/{_DUMP_RUN_ID}/{instance_id}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:  # never let dumping break training
        logging.getLogger(__name__).warning("trajectory dump failed: %s", e)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


_AW_CONFIG_CACHE: Optional[dict] = None
_AW_ENV_CACHE = None
_AW_N_GAMES: Optional[int] = None


def _get_alfworld_env(split: str):
    """Load AlfredTWEnv once per process (expensive game-graph build).

    Each caller still needs their own batched env via .init_env() so that the
    game-iterator cursor is independent per rollout. That's what lets us pin
    a specific game_index per Interaction instance.
    """
    global _AW_CONFIG_CACHE, _AW_ENV_CACHE, _AW_N_GAMES
    if _AW_ENV_CACHE is not None:
        return _AW_ENV_CACHE

    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config_path = os.environ.get(
        "ALFWORLD_BASE_CONFIG",
        "/root/alfworld_data/configs/base_config.yaml",
    )
    with open(config_path) as f:
        _AW_CONFIG_CACHE = yaml.safe_load(f)

    logger.info("Initializing AlfredTWEnv on split=%s", split)
    _AW_ENV_CACHE = AlfredTWEnv(_AW_CONFIG_CACHE, train_eval=split)
    # Sort game_files for deterministic index→game mapping across processes.
    # collect_game_files uses os.walk, whose order is filesystem-dependent.
    _AW_ENV_CACHE.game_files.sort()
    _AW_N_GAMES = len(getattr(_AW_ENV_CACHE, "game_files", []) or []) or 134
    logger.warning(
        "AlfredTWEnv loaded with %d games in split=%s (sorted). First game: %s",
        _AW_N_GAMES, split, _AW_ENV_CACHE.game_files[0] if _AW_ENV_CACHE.game_files else "?",
    )
    return _AW_ENV_CACHE


def _parse_action(raw: str, admissible: list[str]) -> str:
    """Extract a valid ALFWorld action string from the model's free-text output.

    Order of preference:
      1. Line starting with 'Action:' (ReAct convention)
      2. Last line that matches an admissible command
      3. Last non-empty line
      4. Fallback: 'look'
    """
    if not raw:
        return "look"

    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "look"

    for line in reversed(lines):
        m = re.match(r"action\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("`'\"")

    admissible_set = {a.lower() for a in admissible}
    for line in reversed(lines):
        cand = line.strip("`'\"").lower()
        if cand in admissible_set:
            return line.strip("`'\"")

    return lines[-1].strip("`'\"")


def _format_observation(obs: str, admissible: list[str], turn: int, max_turns: int) -> str:
    actions_str = ", ".join(admissible)
    return (
        f"{obs}\n"
        f"Valid actions ({len(admissible)}): {actions_str}\n"
        f"[turn {turn}/{max_turns}] Reply with exactly 'Action: <command>' "
        f"where <command> is copied verbatim from the valid actions list."
    )


class AlfworldInteraction(BaseInteraction):
    """verl BaseInteraction for ALFWorld text-game rollouts."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.split = config.get("split", "eval_out_of_distribution")
        self.max_turns = int(config.get("max_turns", 15))
        logger.warning(
            "AlfworldInteraction init: max_turns=%d, split=%s, raw config keys=%s",
            self.max_turns,
            self.split,
            list(config.keys()),
        )

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        game_index: int = 0,
        task: str = "",
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        env = _get_alfworld_env(self.split)
        # Each Interaction instance gets its own batched env -> its own game cursor.
        # Cycling reset N+1 times lands on game at position (N % n_games) in the
        # deterministic shuffle order used by AlfredTWEnv. Data-prep writes
        # game_index using the same reset order, so parquet and Interaction agree.
        batched = env.init_env(batch_size=1)
        n_games = _AW_N_GAMES or 134
        target_cycles = (game_index % n_games) + 1
        obs, info = None, None
        for _ in range(target_cycles):
            obs, info = batched.reset()

        admissible = info.get("admissible_commands", [[]])[0]
        self._instance_dict[instance_id] = {
            "env": batched,
            "obs": obs[0],
            "admissible": admissible,
            "turn": 0,
            "done": False,
            "task_success": False,
            "cumulative_reward": 0.0,
            "task": task,
            "game_index": game_index,
        }
        logger.info(
            "start instance=%s game_index=%d, %d admissible",
            instance_id,
            game_index,
            len(admissible),
        )
        _dump_event(instance_id, {
            "event": "start",
            "instance_id": instance_id,
            "game_index": game_index,
            "initial_obs": obs[0],
            "admissible_commands": admissible,
            "task_from_kwargs": task,
        })
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instance_dict[instance_id]

        assistant_content = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                assistant_content = item.get("content", "") or ""
                break

        action = _parse_action(assistant_content, state["admissible"])
        obs, scores, dones, info = state["env"].step([action])

        reward_this_turn = float(scores[0])
        done_this_turn = bool(dones[0])

        state["obs"] = obs[0]
        state["admissible"] = info.get("admissible_commands", [[]])[0]
        state["turn"] += 1
        state["done"] = done_this_turn
        state["cumulative_reward"] += reward_this_turn
        if done_this_turn and reward_this_turn > 0:
            state["task_success"] = True

        should_terminate = done_this_turn or state["turn"] >= self.max_turns
        if should_terminate:
            logger.warning(
                "TERMINATE instance=%s turn=%d/%d env_done=%s cum_reward=%.2f",
                instance_id,
                state["turn"],
                self.max_turns,
                done_this_turn,
                state["cumulative_reward"],
            )

        next_prompt = _format_observation(
            state["obs"],
            state["admissible"],
            state["turn"],
            self.max_turns,
        )

        extra = {
            "turn": state["turn"],
            "action_sent": action,
            "cumulative_reward": state["cumulative_reward"],
            "env_done": done_this_turn,
            "task_success": state["task_success"],
        }
        _dump_event(instance_id, {
            "event": "turn",
            "turn": state["turn"],
            "assistant_raw": assistant_content,
            "action_parsed": action,
            "reward": reward_this_turn,
            "cumulative_reward": state["cumulative_reward"],
            "env_done": done_this_turn,
            "new_obs": state["obs"],
            "n_admissible": len(state["admissible"]),
            "admissible_sample": state["admissible"][:10],
            "should_terminate": should_terminate,
        })
        return should_terminate, next_prompt, reward_this_turn, extra

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict.get(instance_id)
        if not state:
            return 0.0
        return 1.0 if state["task_success"] else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        state = self._instance_dict.get(instance_id, {})
        _dump_event(instance_id, {
            "event": "finalize",
            "final_turn": state.get("turn", 0),
            "cumulative_reward": state.get("cumulative_reward", 0.0),
            "task_success": state.get("task_success", False),
            "env_done": state.get("done", False),
        })
        self._instance_dict.pop(instance_id, None)
