"""Build a parquet of ALFWorld tasks for verl multi-turn GRPO training.

Runs inside a Modal container with ALFWorld installed. For each of N games we:
  - Reset AlfredTWEnv
  - Capture the initial English observation (the task prompt)
  - Emit a parquet row in verl's expected schema

Writes train.parquet and val.parquet to the persistent volume.

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

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)

SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFWorld text-based household "
    "simulator. Each turn you see a room description and a list of valid actions. "
    "Respond with a single action on a line prefixed 'Action:'. "
    "Example: 'Action: go to fridge 1'. Do not output anything else on that line."
)


_HARD_TASK_KEYWORDS = ("heat", "cool", "clean", "examine")


def _is_pick_and_place(task_desc: str) -> bool:
    """Pick & Place (type 1) tasks contain 'put' but NOT heat/cool/clean/examine verbs."""
    t = task_desc.lower()
    if "put" not in t:
        return False
    if any(k in t for k in _HARD_TASK_KEYWORDS):
        return False
    if "two" in t:  # type 6 = Pick Two & Place, harder
        return False
    return True


def _extract_task_from_obs(obs: str) -> str:
    for line in obs.splitlines():
        if "task is to" in line.lower():
            return line.split(":", 1)[-1].strip().rstrip(".").lower()
    return ""


@app.function(image=image, volumes={"/output": volume}, timeout=1200)
def build_dataset(
    n_train: int = 4,
    n_val: int = 30,
    max_turns: int = 30,
    max_reset_attempts: int = 134,
    output_subdir: str = "alfworld_pnp",
) -> dict:
    import os

    import pandas as pd
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    with open("/root/alfworld_data/configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    env = AlfredTWEnv(cfg, train_eval="eval_out_of_distribution")
    # Sort for deterministic index -> game mapping across processes.
    # Must sort BEFORE init_env (which freezes order via textworld.gym.register_games).
    env.game_files.sort()
    print(f"First 3 game files (sorted): {env.game_files[:3]}")
    batched = env.init_env(batch_size=1)

    rows: list[dict] = []
    skipped_by_type: dict[str, int] = {}
    target = n_train + n_val
    for i in range(max_reset_attempts):
        if len(rows) >= target:
            break
        obs, info = batched.reset()
        admissible = info.get("admissible_commands", [[]])[0]
        task_desc = _extract_task_from_obs(obs[0])
        if not _is_pick_and_place(task_desc):
            hard_hit = next((k for k in _HARD_TASK_KEYWORDS if k in task_desc), "other")
            skipped_by_type[hard_hit] = skipped_by_type.get(hard_hit, 0) + 1
            continue

        initial_user = (
            f"{obs[0]}\n"
            f"Valid actions ({len(admissible)}): {', '.join(admissible)}\n"
            f"[turn 0/{max_turns}] Reply with exactly 'Action: <command>' "
            f"where <command> is copied verbatim from the valid actions list."
        )

        row_idx = len(rows)
        row = {
            "data_source": "alfworld",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_user},
            ],
            "ability": "agentic",
            "reward_model": {"style": "rule", "ground_truth": "task_success"},
            "extra_info": {
                "split": "train" if row_idx < n_train else "val",
                "index": row_idx,
                "game_index": i,
                "task_desc": task_desc,
                "interaction_kwargs": {
                    "name": "alfworld",
                    "game_index": i,
                    "task": task_desc,
                },
            },
        }
        rows.append(row)

    df_all = pd.DataFrame(rows)
    df_train = df_all.iloc[:n_train].reset_index(drop=True)
    df_val = df_all.iloc[n_train:].reset_index(drop=True)

    out_dir = f"/output/data/{output_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    train_path = f"{out_dir}/train.parquet"
    val_path = f"{out_dir}/val.parquet"
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    volume.commit()

    print(f"wrote {len(df_train)} train rows -> {train_path}")
    print(f"wrote {len(df_val)} val rows -> {val_path}")
    print(f"\nskipped by type (non-Pick&Place): {skipped_by_type}")
    print(f"\nfirst 5 kept task descriptions:")
    for r in rows[:5]:
        print(f"  - game_index={r['extra_info']['game_index']}: {r['extra_info']['task_desc']}")

    return {
        "n_train": len(df_train),
        "n_val": len(df_val),
        "train_path": train_path,
        "val_path": val_path,
    }


@app.local_entrypoint()
def main():
    print(build_dataset.remote())
