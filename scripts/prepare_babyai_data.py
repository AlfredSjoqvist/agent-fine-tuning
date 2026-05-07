"""Build train + val parquets for the BabyAI list-overfit ablation.

Two conditions, two parquets per run:
  - L_per_step: initial user msg includes "Available actions: [...]"
                (the BabyAI default; the per-turn list re-emission is
                set in the YAML interaction config, NOT here).
  - L_none:     initial user msg has NO action list at all.
                Per-turn behavior also has no list (set in YAML).

The system prompt is identical across both conditions — it teaches the
ACTION SCHEMA (turn left, go to <obj>, pick up <obj>, ...). The variable
under test is whether the concrete admissible-action LIST (with object
ids resolved against the current room) is shown. This isolates list
provision as the only thing that differs between conditions.

Multi-level training: tasks are mixed across several BabyAI levels so the
policy doesn't overfit to a single level. Train and val seed ranges are
disjoint within each level.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/prepare_babyai_data.py
"""

import modal

app = modal.App("cs224r-babyai-dataprep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget", "ca-certificates")
    .pip_install(
        "gym",
        "gymnasium<1.0.0",
        "minigrid<2.5.0",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "pyyaml",
        "pandas",
        "pyarrow",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/WooooDyy/AgentGym /opt/agentgym",
        "cd /opt/agentgym/agentenv-babyai && pip install --no-deps -e .",
    )
    .env({
        "PYTHONPATH": "/root/project/src",
    })
    .add_local_dir(
        "c:/Users/Alfred/Desktop/project-cs224r/src",
        remote_path="/root/project/src",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


# Verbatim AgentGym BabyAI system prompt (Xi et al. setup).
# Source: WooooDyy/AgentGym agentenv/agentenv/envs/babyai.py.
# Identical across L_per_step and L_none conditions: schema-level affordance
# information ("you can do <go to>, <pick up>, ..."), no per-instance list.
SYSTEM_PROMPT = (
    "You are an exploration master that wants to finish every goal you are given. "
    "Every round I will give you an observation, and you have to respond an action "
    "and your thought based on the observation to finish the given task. You are "
    "placed in a room and you need to accomplish the given goal with actions.\n\n"
    "You can use the following actions:\n"
    "- turn right\n"
    "- turn left\n"
    "- move forward\n"
    "- go to <obj> <id>\n"
    "- pick up <obj> <id>\n"
    "- go through <door> <id>: <door> must be an open door.\n"
    "- toggle and go through <door> <id>: <door> can be a closed door or a locked door. "
    "If you want to open a locked door, you need to carry a key that is of the same color "
    "as the locked door.\n"
    "- toggle: there is a special case that you can just toggle, e.g. when you need to "
    "open a box.\n\n"
    "Your response must follow this format:\n"
    "Thought:\n"
    "your thoughts.\n\n"
    "Action:\n"
    "your next action"
)


def _format_initial_user_with_list(obs: str, admissible: list[str]) -> str:
    """L_per_step initial user msg — matches AgentGym BabyAI observe() format."""
    return f"{obs}\nAvailable actions: [{', '.join(repr(a) for a in admissible)}]"


def _format_initial_user_no_list(obs: str) -> str:
    """L_none initial user msg — observation only, no list ever shown.

    NOTE: we intentionally do NOT add length-matched filler. Filler text would
    be unnatural and would itself be a confound. The length difference is a
    real consequence of removing the list and is part of the variable we test.
    """
    return obs


def _build_split_rows(
    levels: list[str],
    seeds_per_level: list[int],
    split_role: str,
    condition: str,  # "per_step" or "none"
) -> list[dict]:
    """Build rows by iterating each (level, seed) pair. Level order interleaved
    so train batches are balanced across levels."""
    from agentenv_babyai.environment import BabyAI

    assert condition in ("per_step", "none"), f"unknown condition {condition}"

    # Interleave: (level0, seed0), (level1, seed0), ..., (level0, seed1), (level1, seed1), ...
    pairs: list[tuple[str, int]] = []
    for s_idx in range(len(seeds_per_level)):
        for level in levels:
            pairs.append((level, seeds_per_level[s_idx]))

    rows: list[dict] = []
    for i, (level, seed) in enumerate(pairs):
        if i % 50 == 0:
            print(f"[{split_role}/{condition}] {i}/{len(pairs)}", flush=True)
        env = BabyAI(max_episode_steps=50, game_name=level, seed=seed)
        admissible = env._get_action_space()
        goal = env._get_goal()
        obs = env._get_obs()
        env.env.close()

        if condition == "per_step":
            initial_user = _format_initial_user_with_list(obs, admissible)
        else:  # "none"
            initial_user = _format_initial_user_no_list(obs)

        rows.append({
            "data_source": "babyai",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_user},
            ],
            "ability": "agentic",
            "reward_model": {"style": "rule", "ground_truth": "task_success"},
            "extra_info": {
                "split_role": split_role,
                "condition": condition,
                "game_name": level,
                "seed": seed,
                "index": i,
                "goal": goal,
                "interaction_kwargs": {
                    "name": "babyai",
                    "game_name": level,
                    "seed": seed,
                    "task": goal,
                },
            },
        })
    return rows


# Three BabyAI levels chosen to span the action vocabulary we care about:
#   GoToRedBall: navigation only ("go to <obj>")
#   Pickup:      navigation + manipulation ("go to" + "pick up")
#   Open:        navigation + door interaction ("go to" + "open"/"toggle")
# A held-out 4th level is used for E_held_out_level evaluation (not in training).
DEFAULT_TRAIN_LEVELS = (
    "BabyAI-GoToRedBall-v0",
    "BabyAI-Pickup-v0",
    "BabyAI-Open-v0",
)


@app.function(image=image, volumes={"/output": volume}, timeout=14400)
def build_dataset(
    levels: tuple[str, ...] = DEFAULT_TRAIN_LEVELS,
    n_train_per_level: int = 80,    # 80 × 3 = 240 train tasks
    n_val_per_level: int = 20,      # 20 × 3 = 60 val tasks
    output_root: str = "babyai_listablate_v1",
) -> dict:
    """Generate two parquet pairs (one per condition) for the L_per_step vs
    L_none ablation. Output paths:
      /output/data/{output_root}/per_step/{train,val}.parquet
      /output/data/{output_root}/none/{train,val}.parquet

    The Modal training script must be configured with the right parquet path
    AND the matching `show_action_list_per_turn` flag in the interaction YAML.
    """
    import os
    import pandas as pd

    levels = list(levels)
    print(f"=== levels={levels} n_train_per_level={n_train_per_level} "
          f"n_val_per_level={n_val_per_level} ===", flush=True)

    # Disjoint seed ranges:
    #   train: seeds 0..n_train_per_level-1 (per level)
    #   val:   seeds 100_000..100_000+n_val_per_level-1 (per level)
    train_seeds = list(range(0, n_train_per_level))
    val_seeds = list(range(100_000, 100_000 + n_val_per_level))

    summary: dict[str, dict] = {}
    for condition in ("per_step", "none"):
        print(f"\n=== building condition={condition} ===")

        train_rows = _build_split_rows(levels, train_seeds, "train", condition)
        val_rows = _build_split_rows(levels, val_seeds, "val", condition)

        out_dir = f"/output/data/{output_root}/{condition}"
        os.makedirs(out_dir, exist_ok=True)
        train_path = f"{out_dir}/train.parquet"
        val_path = f"{out_dir}/val.parquet"
        pd.DataFrame(train_rows).to_parquet(train_path, index=False)
        pd.DataFrame(val_rows).to_parquet(val_path, index=False)

        print(f"[{condition}] wrote {len(train_rows)} train -> {train_path}")
        print(f"[{condition}] wrote {len(val_rows)} val   -> {val_path}")
        print(f"[{condition}] sample first 3 train rows:")
        for r in train_rows[:3]:
            ei = r["extra_info"]
            print(f"  - level={ei['game_name']} seed={ei['seed']} goal={ei['goal']!r}")
            print(f"    initial_user[:160]: {r['prompt'][1]['content'][:160]!r}")

        summary[condition] = {
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "train_path": train_path,
            "val_path": val_path,
        }

    volume.commit()
    print("\n=== done ===")
    return summary


@app.local_entrypoint()
def main():
    print(build_dataset.remote(
        levels=DEFAULT_TRAIN_LEVELS,
        n_train_per_level=80,
        n_val_per_level=20,
        output_root="babyai_listablate_v1",
    ))
