"""BabyAI interaction adapter for verl multi-turn GRPO.

Mirrors src/alfworld_rl/alfworld_interaction.py. Wraps AgentGym's BabyAI gym
env (which provides high-level `go to X / pick up Y / open door Z` actions
over the underlying minigrid env, with text-rendered observations).

Each rollout instantiates a fresh `BabyAI` env with (game_name, seed) drawn
from the parquet's interaction_kwargs. The seed determines the room layout
deterministically, so parquet's seed maps to a specific game.
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

_DUMP_DIR = os.environ.get("BABYAI_TRAJECTORY_DUMP_DIR", "")
_DUMP_RUN_ID = os.environ.get("BABYAI_TRAJECTORY_DUMP_RUN", str(int(time.time())))


def _dump_event(instance_id: str, event: dict) -> None:
    if not _DUMP_DIR:
        return
    try:
        os.makedirs(f"{_DUMP_DIR}/{_DUMP_RUN_ID}", exist_ok=True)
        path = f"{_DUMP_DIR}/{_DUMP_RUN_ID}/{instance_id}.jsonl"
        event.setdefault("ts", time.time())
        with open(path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:
        logging.getLogger(__name__).warning("trajectory dump failed: %s", e)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _parse_action(raw: str, admissible: list[str]) -> tuple[str, str]:
    """Same parser shape as alfworld_interaction. Returns (action, parse_path).

    Handles both Xi's multi-line format ("Action:\\nactual_action") AND the
    same-line format ("Action: actual_action"), plus an admissible-list
    fallback when the model omits the `Action:` prefix entirely.
    """
    if not raw:
        return "check available actions", "empty_raw"

    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "check available actions", "empty_lines"

    # 1. Same-line "Action: ..."
    for line in reversed(lines):
        m = re.match(r"action\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            cand = m.group(1).strip().strip("`'\"")
            if cand:  # non-empty after the colon
                return cand, "react_action_prefix_inline"

    # 2. Multi-line "Action:" then action on a following line
    for i, line in enumerate(lines):
        if re.match(r"^\s*action\s*:\s*$", line, re.IGNORECASE):
            if i + 1 < len(lines):
                return lines[i + 1].strip("`'\""), "react_action_prefix_multiline"

    # 3. Admissible-list fallback
    admissible_set = {a.lower() for a in admissible}
    for line in reversed(lines):
        cand = line.strip("`'\"").lower()
        if cand in admissible_set:
            return line.strip("`'\""), "admissible_match_last"

    # 4. Last non-empty line as a last resort
    return lines[-1].strip("`'\""), "fallback_last_line"


def _format_observation(
    obs: str,
    admissible: list[str],
    turn: int,
    max_turns: int,
    show_action_list: bool = True,
) -> str:
    """Per-turn user message. Matches AgentGym BabyAI format:
    show_action_list=True: obs already has \\nAvailable actions: [...] appended (BabyAI default).
    show_action_list=False: just obs (rare for BabyAI but supported for ablations)."""
    if show_action_list:
        return f"{obs}\nAvailable actions: [{', '.join(repr(a) for a in admissible)}]"
    return obs


class BabyAIInteraction(BaseInteraction):
    """verl BaseInteraction for BabyAI multi-turn rollouts."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.default_game_name = config.get("default_game_name", "BabyAI-GoToRedBall-v0")
        self.max_turns = int(config.get("max_turns", 30))
        self.max_episode_steps = int(config.get("max_episode_steps", 50))
        self.show_action_list_per_turn = bool(config.get("show_action_list_per_turn", True))
        logger.warning(
            "BabyAIInteraction init: max_turns=%d, default_game=%s, "
            "show_action_list_per_turn=%s, raw config keys=%s",
            self.max_turns,
            self.default_game_name,
            self.show_action_list_per_turn,
            list(config.keys()),
        )

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        game_name: Optional[str] = None,
        seed: int = 0,
        task: str = "",
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        from agentenv_babyai.environment import BabyAI

        game = game_name or self.default_game_name
        t_init = time.time()
        env = BabyAI(
            max_episode_steps=self.max_episode_steps,
            game_name=game,
            seed=seed,
        )
        # BabyAI calls .reset() in __init__, so the env is ready immediately.
        t_after_init = time.time()

        admissible = env._get_action_space()
        initial_obs = env._get_obs()
        goal = env._get_goal()

        self._instance_dict[instance_id] = {
            "env": env,
            "obs": initial_obs,
            "admissible": admissible,
            "goal": goal,
            "turn": 0,
            "done": False,
            "task_success": False,
            "cumulative_reward": 0.0,
            "task": task,
            "game_name": game,
            "seed": seed,
        }
        logger.info(
            "start instance=%s game=%s seed=%d goal=%r n_admissible=%d",
            instance_id, game, seed, goal, len(admissible),
        )
        _dump_event(instance_id, {
            "event": "start",
            "instance_id": instance_id,
            "game_name": game,
            "seed": seed,
            "goal": goal,
            "initial_obs": initial_obs,
            "admissible_commands": admissible,
            "task_from_kwargs": task,
            "init_env_sec": t_after_init - t_init,
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
        obs, reward, done, infos = state["env"].step(action)
        t_step_end = time.time()

        reward_this_turn = float(reward) - float(state["cumulative_reward"])
        # BabyAI's wrapper returns the running max reward, not per-step delta.
        # We log the delta to keep our turn-scores aggregation consistent.
        done_this_turn = bool(done)

        prev_admissible = state["admissible"]
        state["obs"] = obs
        state["admissible"] = state["env"]._get_action_space()
        state["turn"] += 1
        state["done"] = done_this_turn
        state["cumulative_reward"] = float(reward)
        if done_this_turn and float(reward) > 0:
            state["task_success"] = True

        should_terminate = done_this_turn or state["turn"] >= self.max_turns
        if should_terminate:
            logger.warning(
                "TERMINATE instance=%s turn=%d/%d env_done=%s cum_reward=%.2f success=%s",
                instance_id, state["turn"], self.max_turns,
                done_this_turn, state["cumulative_reward"], state["task_success"],
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
            "action_is_valid": bool(infos.get("action_is_valid", False)),
            "reward_delta": reward_this_turn,
            "cumulative_reward": state["cumulative_reward"],
            "env_done": done_this_turn,
            "new_obs": state["obs"],
            "n_admissible_before": len(prev_admissible),
            "n_admissible_after": len(state["admissible"]),
            "admissible_before": prev_admissible,
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
        return float(state["cumulative_reward"])

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        state = self._instance_dict.get(instance_id, {})
        _dump_event(instance_id, {
            "event": "finalize",
            "final_turn": state.get("turn", 0),
            "cumulative_reward": state.get("cumulative_reward", 0.0),
            "task_success": state.get("task_success", False),
            "env_done": state.get("done", False),
        })
        # Try to clean up the gym env so subprocess handles don't leak.
        env = state.get("env")
        if env is not None:
            try:
                env.env.close()
            except Exception:
                pass
        self._instance_dict.pop(instance_id, None)
