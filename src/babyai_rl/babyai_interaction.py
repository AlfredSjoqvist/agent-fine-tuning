"""BabyAI interaction adapter for verl multi-turn GRPO.

Wraps AgentGym's BabyAI gym env (which provides high-level 
`go to X / pick up Y / open door Z` actions
over the underlying minigrid env, with text-rendered observations).

Each rollout instantiates a fresh `BabyAI` env with (game_name, seed) 
drawn from the parquet's interaction_kwargs. 
The seed determines the room layout deterministically, 
so parquet's seed maps to a specific game.
"""

from __future__ import annotations
from typing import Any, Optional
from uuid import uuid4
from verl.interactions.base import BaseInteraction
from agentenv_babyai.environment import BabyAI
import re 

def _build_prompt(
    obs: str,
    admissible: list[str],
    show_action_list: bool = True,
) -> str:
    if not show_action_list:
        return obs
    return f"{obs}\nAvailable actions: [{', '.join(repr(action) for action in admissible)}]"

def _normalize_action_for_env(action: str) -> str:
    action = action.strip()
    action = re.sub("^pick\s+up\s+", "pickup ", action, flags=re.IGNORECASE)
    return action

def _parse_action(raw: str,  admissible: list[str]) -> tuple[str, str]:
    if not raw:
        return "drink cola", "empty_raw"

    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "drink cola", "empty_lines"

    # strategy 1: action prefix inline 
    for line in reversed(lines):
        matched = re.match(r"action\s*:\s*(.+)$", line, re.IGNORECASE)
        if matched:
            candidate = matched.group(1).strip().strip("`'\"")
            return candidate, "react_action_prefix_inline"

    # strategy 2: action prefix, then action literal on the next line
    for i, line in enumerate(lines):
        if re.match(r"action\s*:\s*$", line, re.IGNORECASE):
            if i + 1 < len(lines):
                return lines[i+1].strip("`'\""), "react_action_prefix_next_line"

    # strategy 3: if any action mentioned matches with admissible list 
    raw_lower = raw.lower()
    for candidate in admissible:
        if candidate.lower() in raw_lower:
            return candidate, "admissible_substring_match"
    
    # strategy 4: last line fall back 
    return lines[-1].strip("`'\""), "fallback_last_line"
    


class BabyAIInteraction(BaseInteraction):
    def __init__(self, config: dict): 
        super().__init__(config)
        # outer dict maps instance_id to a rollout state dict
        # inner dict stores information for each rollout
        self._instance_dict: dict[str, dict[str,Any]] = {} 
        self.default_game_name = config.get("default_game_name", "BabyAI-GoToRedBall-v0")
        self.max_turns = int(config.get("max_turns", 30))
        self.max_episode_steps = int(config.get("max_episode_steps", 50))
        self.show_action_list_per_turn = bool(config.get("show_action_list_per_turn", True))
        self.block_check_available = bool(config.get("block_check_available", False))

    async def start_interaction(self, instance_id=None, **kwargs) -> str: 
        """Spin up env, store state, return instance_id."""
        # Generate instance_id = str(uuid4()) if one wasn't provided.
        if instance_id is None:
            instance_id = str(uuid4())

        game = kwargs.get("game_name") or self.default_game_name
        seed = kwargs.get("seed", 0)
        task = kwargs.get("task", "")
        # instantiate env     
        env = BabyAI(max_episode_steps=self.max_episode_steps, game_name=game, seed=seed)
        
        # snapshot the starting state
        initial_obs = env._get_obs()
        admissible = env._get_action_space()
        goal = env._get_goal()

        #
        self._instance_dict[instance_id] = {
            "env": env,                  # the live gym object
            "obs": initial_obs,          # current text obs
            "admissible": admissible,    # curr list of available actions
            "goal": goal,                # goal string, fixed for the episode
            "done": False,
            "task_reward": 0.0,
            "task_success": False,        # did the agent complete the goal
            "task": task,                # identical to goal, kept for logging/provenance 
            "turn": 0,
        }
        return instance_id

    
    async def generate_response(self, instance_id, messages, **kwargs
        ) -> tuple[bool, str, float, dict]:
        """Parse last assistant turn, step env, return
        (should_terminate, next_user_prompt, reward_delta, extra_info)."""
        state = self._instance_dict[instance_id]

        assistant_content = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                assistant_content = item.get("content", "") or ""
                break
        
        action, strategy = _parse_action(assistant_content, state["admissible"])
        action_env = _normalize_action_for_env(action)
        # following AgentGym return signature, step the env 
        obs, reward, done, infos = state["env"].step(action_env)
        state["obs"] = obs
        state["admissible"] =state["env"]._get_action_space()
        state["turn"] += 1 

        # block the action list check for ablation tests
        if self.block_check_available and obs.startswith("You can take the following actions"): 
            should_terminate = state["turn"] >= self.max_turns
            next_prompt = "That action is not available. Use your observation to construct a valid action."
            return should_terminate, next_prompt, 0.0, {
                "turn": state["turn"],
                "action_sent": action_env,
                "env_done": False,
                "task_success": state["task_success"],
            }

        success_this_turn = done and float(reward) > 0
        if success_this_turn:
            step_count = state["turn"]
            done = True
            state["task_success"] = True 
            state["task_reward"] = 1.0 - 0.9 * step_count / self.max_turns
    
        
        should_terminate = state["turn"] >= self.max_turns or done 
        next_user_prompt = _build_prompt(state["obs"], state["admissible"], show_action_list=self.show_action_list_per_turn)
        reward_delta = state["task_reward"]

        return should_terminate, next_user_prompt, reward_delta, {
            "turn": state["turn"],
            "action_sent": action_env,
            "env_done": done,
            "task_success": state["task_success"],
        }


    async def calculate_score(self, instance_id, **kwargs) -> float:
        """return final scalar reward for this rollout.
        This function is never called in this project.
        So it is just implemented for the interface convention sake"""
        state = self._instance_dict.get(instance_id)
        if not state:
            return 0.0
        return float(state.get("task_reward", 0.0))

    async def finalize_interaction(self, instance_id, **kwargs) -> None: 
        """Clean up env handle, remove from instance dict."""
        self._instance_dict.pop(instance_id, None)

