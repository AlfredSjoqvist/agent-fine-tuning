"""Custom reward function for verl's reward_manager dispatch.

verl's NaiveRewardManager calls compute_score(data_source, solution_str, ground_truth, extra_info, ...).
It also injects the Interaction's per-turn rewards into extra_info["rollout_reward_scores"].

For ALFWorld the env emits reward=1.0 only on task completion, 0.0 otherwise.
So `max(rollout_reward_scores) > 0` is a correct task-success signal.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Process-local counters so we can periodically dump batch stats. Reset on
# every "report" interval so the log shows per-window success rates rather
# than ever-growing totals.
_LOCK = threading.Lock()
_N_CALLS = 0
_N_SUCCESS = 0
_REPORT_EVERY = int(os.environ.get("ALFWORLD_REWARD_REPORT_EVERY", "16"))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    global _N_CALLS, _N_SUCCESS

    if not extra_info:
        # Don't silently return 0 — log it. A burst of these means the
        # Interaction's per-turn rewards aren't reaching the reward manager.
        print("[ALFWORLD_REWARD] WARN: empty extra_info; returning 0.0", flush=True)
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

    with _LOCK:
        _N_CALLS += 1
        if result > 0:
            _N_SUCCESS += 1
        if _N_CALLS % _REPORT_EVERY == 0:
            print(
                f"[ALFWORLD_REWARD] window n={_REPORT_EVERY} "
                f"successes={_N_SUCCESS}/{_N_CALLS} "
                f"({100.0 * _N_SUCCESS / _N_CALLS:.1f}% cumulative since proc start)",
                flush=True,
            )

    if result > 0:
        print(
            f"[ALFWORLD_REWARD] SUCCESS reward=1.0 "
            f"scores_seen={len(numeric)} max={max(numeric):.2f} sum={sum(numeric):.2f} "
            f"keys={[k for k in (('rollout_reward_scores' if scores else None), ('turn_scores' if turn_scores else None), ('tool_rewards' if tool_rewards else None)) if k]}",
            flush=True,
        )
    return result
