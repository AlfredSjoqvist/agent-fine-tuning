"""Custom reward function for verl's reward_manager dispatch.

verl's NaiveRewardManager calls compute_score(data_source, solution_str, ground_truth, extra_info, ...).
It also injects the Interaction's per-turn rewards into extra_info["rollout_reward_scores"].

For ALFWorld the env emits reward=1.0 only on task completion, 0.0 otherwise.
So `max(rollout_reward_scores) > 0` is a correct task-success signal.
"""

import logging

logger = logging.getLogger(__name__)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    if not extra_info:
        return 0.0

    scores = extra_info.get("rollout_reward_scores", None)
    turn_scores = extra_info.get("turn_scores", None)
    tool_rewards = extra_info.get("tool_rewards", None)

    # Try every candidate container; flatten and check for any positive reward.
    candidates = []
    for v in (scores, turn_scores, tool_rewards):
        if v is None:
            continue
        if isinstance(v, dict):
            candidates.extend(v.values())
        elif isinstance(v, (list, tuple)):
            candidates.extend(v)
        else:
            candidates.append(v)

    flat = []
    for v in candidates:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)

    numeric = []
    for v in flat:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            pass

    result = 1.0 if numeric and max(numeric) > 0 else 0.0
    if result > 0:
        print(f"[ALFWORLD_REWARD] SUCCESS reward=1.0 scores_sum={sum(numeric):.1f}", flush=True)
    return result
