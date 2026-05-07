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
        # Stamp every event with wall-clock so we can compute per-turn latencies
        # post-hoc. Without this we cannot diagnose throughput regressions.
        event.setdefault("ts", time.time())
        with open(path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:  # never let dumping break training
        logging.getLogger(__name__).warning("trajectory dump failed: %s", e)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


_AW_CONFIG_CACHE: Optional[dict] = None
# Cache one AlfredTWEnv per split so train and val rollouts route to the
# correct env. Key: split name ("train" / "eval_out_of_distribution"). Value: env.
_AW_ENV_CACHE: dict[str, Any] = {}
_AW_N_GAMES: dict[str, int] = {}


def _get_alfworld_env(split: str):
    """Load AlfredTWEnv per split (expensive game-graph build), cache by split."""
    global _AW_CONFIG_CACHE, _AW_ENV_CACHE, _AW_N_GAMES
    if split in _AW_ENV_CACHE:
        return _AW_ENV_CACHE[split]

    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    if _AW_CONFIG_CACHE is None:
        config_path = os.environ.get(
            "ALFWORLD_BASE_CONFIG",
            "/root/alfworld_data/configs/base_config.yaml",
        )
        with open(config_path) as f:
            _AW_CONFIG_CACHE = yaml.safe_load(f)

    logger.info("Initializing AlfredTWEnv on split=%s", split)
    env = AlfredTWEnv(_AW_CONFIG_CACHE, train_eval=split)
    # For TRAIN split: stratified subsample 84 games × 6 task types = 504.
    # This is critical: it caps the per-rollout game-cycling overhead from
    # ~3553×30ms (~100s) down to ~504×30ms (~15s). Must match
    # prepare_alfworld_data.py exactly so game_index in the parquet maps
    # to the same game in this process.
    # For VAL split: use all 134 valid_unseen games, sorted.
    if split == "train":
        from alfworld_rl.sampling import stratified_sample_train_games
        env.game_files = stratified_sample_train_games(env.game_files)
    else:
        env.game_files.sort()
    _AW_ENV_CACHE[split] = env
    _AW_N_GAMES[split] = len(env.game_files) or 134
    logger.warning(
        "AlfredTWEnv loaded with %d games in split=%s. First game: %s",
        _AW_N_GAMES[split], split, env.game_files[0] if env.game_files else "?",
    )
    return env


def _parse_action(raw: str, admissible: list[str]) -> tuple[str, str]:
    """Extract a valid ALFWorld action string from the model's free-text output.

    Returns (action, parse_path) where parse_path is one of:
      'empty_raw', 'empty_lines', 'react_action_prefix',
      'admissible_match_last', 'fallback_last_line'.
    Tracking the path lets us see whether the model is following ReAct format
    or whether the parser is rescuing it via the admissible-match fallback.
    """
    if not raw:
        return "look", "empty_raw"

    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "look", "empty_lines"

    for line in reversed(lines):
        m = re.match(r"action\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("`'\""), "react_action_prefix"

    admissible_set = {a.lower() for a in admissible}
    for line in reversed(lines):
        cand = line.strip("`'\"").lower()
        if cand in admissible_set:
            return line.strip("`'\""), "admissible_match_last"

    return lines[-1].strip("`'\""), "fallback_last_line"


def _format_observation(
    obs: str,
    admissible: list[str],
    turn: int,
    max_turns: int,
    show_action_list: bool = True,
) -> str:
    """Per-turn user prompt. Matches AgentGym ALFWorld format verbatim.
    show_action_list=True (L_per_step): obs + "\\nAVAILABLE ACTIONS: ..." (BabyAI-like).
    show_action_list=False (L_init / option F): just obs (Xi's actual ALFWorld setup)."""
    if show_action_list:
        return f"{obs}\nAVAILABLE ACTIONS: {','.join(admissible)}"
    return obs


class AlfworldInteraction(BaseInteraction):
    """verl BaseInteraction for ALFWorld text-game rollouts."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.split = config.get("split", "eval_out_of_distribution")
        self.max_turns = int(config.get("max_turns", 15))
        self.show_action_list_per_turn = bool(config.get("show_action_list_per_turn", True))
        logger.warning(
            "AlfworldInteraction init: max_turns=%d, split=%s, show_action_list_per_turn=%s, raw config keys=%s",
            self.max_turns,
            self.split,
            self.show_action_list_per_turn,
            list(config.keys()),
        )

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        game_index: int = 0,
        task: str = "",
        split: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        # Per-row split from parquet's interaction_kwargs takes precedence
        # over the YAML default (self.split). This is what lets the same
        # Interaction class handle both train and val rollouts correctly.
        actual_split = split or self.split
        env = _get_alfworld_env(actual_split)
        # Pin the target game directly via game_files mutation, then init_env once,
        # then reset() once. The previous cycle-N-times pattern was costing
        # 9-15 minutes per rollout (median ~250 resets × ~2 sec each). The
        # _full_game_files cache preserves the full sorted list across mutations.
        if not hasattr(env, "_full_game_files"):
            env._full_game_files = list(env.game_files)
        n_games = len(env._full_game_files)
        target_idx = game_index % n_games
        target_game = env._full_game_files[target_idx]

        t_init = time.time()
        env.game_files = [target_game]
        env.num_games = 1
        batched = env.init_env(batch_size=1)
        t_after_init = time.time()
        obs, info = batched.reset()
        t_after_resets = time.time()

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
            "split": actual_split,
            "initial_obs": obs[0],
            "admissible_commands": admissible,
            "task_from_kwargs": task,
            "init_env_sec": t_after_init - t_init,
            "reset_loop_sec": t_after_resets - t_after_init,
            "n_resets": 1,  # always 1 now (game pinned via game_files mutation)
            "target_idx": target_idx,
            "target_game": target_game,
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

        t_parse_start = time.time()
        action, parse_path = _parse_action(assistant_content, state["admissible"])
        t_parse_end = time.time()
        action_in_admissible = action.lower() in {a.lower() for a in state["admissible"]}

        t_step_start = time.time()
        obs, scores, dones, info = state["env"].step([action])
        t_step_end = time.time()

        reward_this_turn = float(scores[0])
        done_this_turn = bool(dones[0])

        prev_admissible = state["admissible"]
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
                "TERMINATE instance=%s turn=%d/%d env_done=%s cum_reward=%.2f success=%s",
                instance_id,
                state["turn"],
                self.max_turns,
                done_this_turn,
                state["cumulative_reward"],
                state["task_success"],
            )

        next_prompt = _format_observation(
            state["obs"],
            state["admissible"],
            state["turn"],
            self.max_turns,
            show_action_list=self.show_action_list_per_turn,
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
            "assistant_raw_chars": len(assistant_content),
            "action_parsed": action,
            "parse_path": parse_path,
            "action_in_admissible": action_in_admissible,
            "reward": reward_this_turn,
            "cumulative_reward": state["cumulative_reward"],
            "env_done": done_this_turn,
            "new_obs": state["obs"],
            "n_admissible_before": len(prev_admissible),
            "n_admissible_after": len(state["admissible"]),
            "admissible_before": prev_admissible,  # full list, not [:10]
            "admissible_after_sample": state["admissible"][:15],
            "parse_sec": t_parse_end - t_parse_start,
            "env_step_sec": t_step_end - t_step_start,
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
