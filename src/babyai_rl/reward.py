"""
A single function verl calls after each rollout to turn environment feedback into one scalar reward for GRPO/PPO.

Return the sum of per-turn reward deltas, which verl already collected in extra_info["turn_scores"]
This part of code is not AI-generated. @Hana
"""

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None,
    **kwargs
) -> float:
    
    # Guard: if extra_info is missing or empty 
    if not extra_info:
        print("[BABYAI] WARN: empty extra_info, returning 0.0 ")
        return 0.0

    # get values 
    per_turn_scores = extra_info.get("turn_scores", [])


    # shape validation 
    # guard for per_turn_scores empty
    if not per_turn_scores:
        print("[BABYAI] WARN: empty turn_scores, rewards misssing, returning 0.0 ")
        return 0.0
    
    assert isinstance(per_turn_scores, list), \
        f"expected sequence of non-empty per-turn rewards, got {per_turn_scores!r}"
    assert len(per_turn_scores) <= 30, \
        f"sequence length {len(per_turn_scores)} exceed max_turns=30"
    
    # coercion loop converting all elements into float 
    scores = []
    for score in per_turn_scores:
        try: 
            scores.append(float(score))
        except (TypeError, ValueError):
            # TypeError: something float() can't even attempt
            # ValueError: a string that float() tries but fails on 
            pass # skip policy — bad entry is dropped silently

    # guard after coercion in case all entries bad and skipped, returning empty list
    if not scores:
        print("[BABYAI] WARN: empty scores, only bad entries, returning 0.0 ")
        return 0.0

    # final summation and return 
    cumulative = sum(scores)
    return cumulative 

    
