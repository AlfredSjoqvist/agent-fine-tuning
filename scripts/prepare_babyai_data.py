"""Build train + val parquets for the BabyAI ablation experiment.

Two conditions, two parquets each:
  - per_step:  initial user msg includes "Available actions: [...]" and the
               list is re-emitted every turn (set in the YAML interaction config).
  - examples:  no per-instance action list; the system prompt contains a
               generic format block with example actions only.

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
        "/Users/hana/Desktop/project-cs224r/src",
        remote_path="/root/project/src",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


# Base system prompt — schema-level affordance info only (used by per_step).
SYSTEM_PROMPT_BASE = (
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
    "- drop\n"
    "- go through <door> <id>: <door> must be an open door.\n"
    "- toggle and go through <door> <id>: <door> can be a closed door or a locked door. "
    "If you want to open a locked door, you need to carry a key that is of the same color "
    "as the locked door.\n"
    "- toggle: there is a special case that you can just toggle, e.g. when you need to "
    "open a box.\n\n"
)

# Format-examples block — appended ONLY in the examples condition.
# Teaches the LLM what concrete action surface forms look like without giving
# game-specific affordance info per instance.
EXAMPLES_BLOCK = (
    "Examples of valid action surface forms (the placeholders <obj> <id> get "
    "filled in based on what is in your room — your specific room will have "
    "DIFFERENT objects, colors, and IDs than these examples):\n"
    "- 'turn left'\n"
    "- 'turn right'\n"
    "- 'move forward'\n"
    "- 'go to red ball 1'\n"
    "- 'go to grey closed door 1'\n"
    "- 'pick up yellow key 1'\n"
    "- 'pick up purple ball 1'\n"
    "- 'drop'\n"
    "- 'toggle and go through red closed door 1'\n"
    "- 'go through red open door 1'\n"
    "- 'toggle'\n\n"
    "Object IDs use the form '<color> <object> <integer>'. Each turn you will "
    "receive a description of what you can see; construct a valid action by "
    "combining the schema above with the objects in your observation.\n\n"
)

FORMAT_BLOCK = (
    "Your response must follow this format:\n"
    "Thought:\n"
    "your thoughts.\n\n"
    "Action:\n"
    "your next action"
)


def _system_prompt_for(condition: str) -> str:
    if condition == "examples":
        return SYSTEM_PROMPT_BASE + EXAMPLES_BLOCK + FORMAT_BLOCK
    return SYSTEM_PROMPT_BASE + FORMAT_BLOCK


def _rewrite_list_for_llm(admissible: list[str]) -> list[str]:
    """Rewrite AgentGym surface forms to natural English: 'pickup X' -> 'pick up X'."""
    import re as _re
    return [_re.sub(r"^pickup\s+", "pick up ", a) for a in admissible]


def _format_initial_user_with_list(obs: str, admissible: list[str], goal: str) -> str:
    rewritten = _rewrite_list_for_llm(admissible)
    return (
        f"Goal: {goal}\n\n"
        f"{obs}\n"
        f"Available actions: [{', '.join(repr(a) for a in rewritten)}]"
    )


def _format_initial_user_no_list(obs: str, goal: str) -> str:
    return f"Goal: {goal}\n\n{obs}"


def _build_split_rows(
    levels: list[str],
    seeds_per_level: list[int],
    split_role: str,
    condition: str,
) -> list[dict]:
    from agentenv_babyai.environment import BabyAI

    assert condition in ("per_step", "examples"), f"unknown condition {condition}"

    system_prompt = _system_prompt_for(condition)

    # Interleave seeds × levels so batches are balanced across levels.
    pairs: list[tuple[str, int]] = [
        (level, seed)
        for seed in seeds_per_level
        for level in levels
    ]

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
            initial_user = _format_initial_user_with_list(obs, admissible, goal)
        else:
            initial_user = _format_initial_user_no_list(obs, goal)

        rows.append({
            "data_source": "babyai",
            "prompt": [
                {"role": "system", "content": system_prompt},
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


# Three BabyAI levels spanning the action vocabulary:
#   GoToRedBall: navigation only
#   Pickup:      navigation + manipulation
#   Open:        navigation + door interaction
DEFAULT_TRAIN_LEVELS = (
    "BabyAI-GoToRedBall-v0",
    "BabyAI-Pickup-v0",
    "BabyAI-Open-v0",
)


@app.function(image=image, volumes={"/output": volume}, timeout=14400)
def build_dataset(
    levels: tuple[str, ...] = DEFAULT_TRAIN_LEVELS,
    n_train_per_level: int = 80,
    n_val_per_level: int = 20,
    output_root: str = "babyai_v1",
) -> dict:
    """Generate parquet pairs for each condition:
      /output/data/{output_root}/per_step/{train,val}.parquet
      /output/data/{output_root}/examples/{train,val}.parquet
    """
    import os
    import pandas as pd

    levels = list(levels)
    print(f"=== levels={levels} n_train={n_train_per_level} n_val={n_val_per_level} ===", flush=True)

    train_seeds = list(range(0, n_train_per_level))
    val_seeds = list(range(100_000, 100_000 + n_val_per_level))

    summary: dict[str, dict] = {}
    for condition in ("per_step", "examples"):
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
        output_root="babyai_v1",
    ))


@app.local_entrypoint()
def smoke():
    """Minimal data prep for smoke testing: 6 train + 2 val seeds per level.
    Run with: modal run scripts/prepare_babyai_data.py::smoke
    """
    print(build_dataset.remote(
        levels=DEFAULT_TRAIN_LEVELS,
        n_train_per_level=6,
        n_val_per_level=2,
        output_root="babyai_smoke",
    ))
