"""Build paper-faithful train + val parquets for ALFWorld GRPO training.

Train data is sampled from ALFWorld's `train` split (~3,300 games, all 6 task types).
Val data is sampled from `valid_unseen` (134 held-out games).
Each parquet row carries `interaction_kwargs.split` so the verl Interaction loads
the correct env per rollout.

System prompt teaches the ReAct paradigm (Thought + Action), matching Xi et al. (2026).

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/prepare_alfworld_data.py
"""

import modal

app = modal.App("cs224r-alfworld-dataprep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget", "ca-certificates")
    .pip_install(
        "alfworld[full]",
        "textworld",
        "pyyaml",
        "pandas",
        "pyarrow",
    )
    .env({"ALFWORLD_DATA": "/root/alfworld_data"})
    .run_commands(
        "alfworld-download -f",
        "git clone --depth 1 https://github.com/alfworld/alfworld.git /tmp/alfworld_repo",
        "mkdir -p /root/alfworld_data/configs",
        "cp /tmp/alfworld_repo/configs/base_config.yaml /root/alfworld_data/configs/base_config.yaml",
    )
)

volume = modal.Volume.from_name("224project-data", create_if_missing=True)


SYSTEM_PROMPT = (
    "You are an expert agent in the ALFWorld text-based household simulator. "
    "Each turn, you see a room observation and a list of valid actions. "
    "First reason briefly about what to do, then issue exactly one action.\n\n"
    "Format every response as:\n"
    "Thought: <1-2 sentences of planning>\n"
    "Action: <one command, copied verbatim from the valid actions list>\n\n"
    "Example:\n"
    "Thought: I need to find a peppershaker. Those are usually in cabinets "
    "or drawers. Let me check drawer 1 first.\n"
    "Action: go to drawer 1"
)


def _extract_task_from_obs(obs: str) -> str:
    for line in obs.splitlines():
        if "task is to" in line.lower():
            return line.split(":", 1)[-1].strip().rstrip(".").lower()
    return ""


def _format_initial_user(obs: str, admissible: list[str], max_turns: int) -> str:
    return (
        f"{obs}\n"
        f"Valid actions ({len(admissible)}): {', '.join(admissible)}\n"
        f"[turn 0/{max_turns}] Now respond with `Thought:` followed by `Action:`."
    )


def _build_split_rows(cfg, split: str, n_target: int, max_turns: int) -> list[dict]:
    """Reset env n_target times on a given split, capture initial obs+task per game.

    Each row gets game_index = position in the SORTED game_files list, so
    the Interaction's pinning logic (cycle reset N+1 times on the same sorted
    list) lands on the same game during training.
    """
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    env = AlfredTWEnv(cfg, train_eval=split)
    # Determinism across processes: sort before init_env (which freezes order).
    env.game_files.sort()
    n_total = len(env.game_files)
    target = n_total if n_target < 0 else min(n_target, n_total)
    print(f"[{split}] {n_total} games available, building {target} rows")
    print(f"[{split}] first game: {env.game_files[0]}")

    batched = env.init_env(batch_size=1)
    rows: list[dict] = []
    task_type_counts: dict[str, int] = {}
    for i in range(target):
        obs, info = batched.reset()
        admissible = info.get("admissible_commands", [[]])[0]
        task_desc = _extract_task_from_obs(obs[0])

        # Loose task-type tagging for stats (not filtering anymore)
        if "heat" in task_desc or "hot" in task_desc:
            t = "heat"
        elif "cool" in task_desc or "cold" in task_desc:
            t = "cool"
        elif "clean" in task_desc:
            t = "clean"
        elif "examine" in task_desc or "desklamp" in task_desc:
            t = "examine"
        elif "two" in task_desc:
            t = "pick_two"
        elif "put" in task_desc:
            t = "pick_place"
        else:
            t = "other"
        task_type_counts[t] = task_type_counts.get(t, 0) + 1

        initial_user = _format_initial_user(obs[0], admissible, max_turns)
        rows.append({
            "data_source": "alfworld",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_user},
            ],
            "ability": "agentic",
            "reward_model": {"style": "rule", "ground_truth": "task_success"},
            "extra_info": {
                "split_role": "train" if split == "train" else "val",
                "alfworld_split": split,
                "index": i,
                "game_index": i,
                "task_desc": task_desc,
                "interaction_kwargs": {
                    "name": "alfworld",
                    "game_index": i,
                    "task": task_desc,
                    "split": split,
                },
            },
        })

    print(f"[{split}] task type breakdown: {task_type_counts}")
    return rows


@app.function(image=image, volumes={"/output": volume}, timeout=3600)
def build_dataset(
    n_train: int = -1,        # -1 = use all train-split games
    n_val: int = 134,         # 134 = all valid_unseen games
    max_turns: int = 30,
    output_subdir: str = "alfworld_paper_faithful",
) -> dict:
    import os

    import pandas as pd
    import yaml

    with open("/root/alfworld_data/configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("\n=== Building TRAIN parquet from `train` split ===")
    train_rows = _build_split_rows(cfg, split="train", n_target=n_train, max_turns=max_turns)

    print("\n=== Building VAL parquet from `eval_out_of_distribution` (valid_unseen) ===")
    val_rows = _build_split_rows(cfg, split="eval_out_of_distribution", n_target=n_val, max_turns=max_turns)

    out_dir = f"/output/data/{output_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    train_path = f"{out_dir}/train.parquet"
    val_path = f"{out_dir}/val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    volume.commit()

    print(f"\nwrote {len(train_rows)} train rows -> {train_path}")
    print(f"wrote {len(val_rows)} val rows -> {val_path}")
    print("\nfirst 3 train task descriptions:")
    for r in train_rows[:3]:
        print(f"  - game_index={r['extra_info']['game_index']}: {r['extra_info']['task_desc']}")

    return {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_path": train_path,
        "val_path": val_path,
    }


@app.local_entrypoint()
def main():
    print(build_dataset.remote())
