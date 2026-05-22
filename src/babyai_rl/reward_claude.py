"""Custom reward function for verl's reward_manager dispatch.

verl's NaiveRewardManager calls compute_score(data_source, solution_str, ground_truth, extra_info, ...).
extra_info["rollout_reward_scores"] (or turn_scores / tool_rewards) holds the
per-turn rewards from the BabyAI Interaction.

BabyAI gives DENSE rewards (incremental progress toward goal). We aggregate by
taking the cumulative max — the highest reward state reached across the rollout.
This is faithful to BabyAI semantics where reward=1.0 means task completed.
"""

###
# For debugging: Optional but useful during Modal smokes 
# when you want to confirm rewards are flowing without opening TensorBoard. 
# Each Ray worker keeps its own counters 
# (not one global cluster total).
import os
import threading

_LOCK = threading.Lock()
_N_CALLS = 0
_N_SUCCESS = 0
_N_NONZERO = 0
_REPORT_EVERY = int(os.environ.get("BABYAI_REWARD_REPORT_EVERY", "16"))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    global _N_CALLS, _N_SUCCESS, _N_NONZERO

    if not extra_info:
        print("[BABYAI_REWARD] WARN: empty extra_info; returning 0.0", flush=True)
        return 0.0

    scores = extra_info.get("rollout_reward_scores", None) 
    turn_scores = extra_info.get("turn_scores", None)
    tool_rewards = extra_info.get("tool_rewards", None)

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

    # BabyAI returns reward DELTAS per turn (we made the Interaction emit deltas).
    # Cumulative reward = sum of deltas. Success = cumulative ≥ 1.0.
    cumulative = sum(numeric) if numeric else 0.0

    with _LOCK:
        _N_CALLS += 1
        if cumulative > 0:
            _N_NONZERO += 1
        if cumulative >= 1.0 - 1e-6:
            _N_SUCCESS += 1
        if _N_CALLS % _REPORT_EVERY == 0:
            print(
                f"[BABYAI_REWARD] window n={_REPORT_EVERY} "
                f"successes={_N_SUCCESS}/{_N_CALLS} "
                f"nonzero={_N_NONZERO}/{_N_CALLS} "
                f"({100.0 * _N_SUCCESS / _N_CALLS:.1f}% / "
                f"{100.0 * _N_NONZERO / _N_CALLS:.1f}% cumulative since proc start)",
                flush=True,
            )

    if cumulative >= 1.0 - 1e-6:
        print(
            f"[BABYAI_REWARD] SUCCESS cumulative={cumulative:.2f} n_deltas={len(numeric)}",
            flush=True,
        )
    return cumulative
