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
    .env({
        "ALFWORLD_DATA": "/root/alfworld_data",
        "PYTHONPATH": "/root/project/src",
    })
    .run_commands(
        "alfworld-download -f",
        "git clone --depth 1 https://github.com/alfworld/alfworld.git /tmp/alfworld_repo",
        "mkdir -p /root/alfworld_data/configs",
        "cp /tmp/alfworld_repo/configs/base_config.yaml /root/alfworld_data/configs/base_config.yaml",
    )
    .add_local_dir(
        "/Users/hana/Desktop/project-cs224r/src",
        remote_path="/root/project/src",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


# Verbatim AgentGym ALFWorld system prompt (Xi et al. 2026 base setup).
# Source: WooooDyy/AgentGym - agentenv/agentenv/envs/alfworld.py line 233.
# Differences vs our previous prompt:
#   - explains "Nothing happened" feedback (key for invalid-action recovery)
#   - allows ACTION-only turns (skip THOUGHT) when the action is obvious
#   - explicit "Reminder" enforcing the must-be-from-available-actions rule
SYSTEM_PROMPT = (
    "Interact with a household to solve a task. Imagine you are an intelligent agent "
    "in a household environment and your target is to perform actions to complete the "
    "task goal. At the beginning of your interactions, you will be given the detailed "
    "description of the current environment and your goal to accomplish. For each of "
    "your turn, you will be given a list of actions which you can choose one to perform "
    "in this turn. You should choose from two actions: \"THOUGHT\" or \"ACTION\". "
    "If you choose \"THOUGHT\", you should first think about the current condition and "
    "plan for your future actions, and then output your action in this turn. Your "
    "output must strictly follow this format:\"Thought:\nyour thoughts.\n\nAction:\n"
    "your next action\"; If you choose \"ACTION\", you should directly output the "
    "action in this turn. Your output must strictly follow this format:\"Action:\n"
    "your next action\". After your each turn, the environment will give you "
    "immediate feedback based on which you plan your next few steps. if the envrionment "
    "output \"Nothing happened\", that means the previous action is invalid and you "
    "should try more options.\n Reminder: \n1. the action must be chosen from the "
    "given available actions. Any actions except provided available actions will be "
    "regarded as illegal. \n2. Think when necessary, try to act directly more in the process."
)


def _extract_task_from_obs(obs: str) -> str:
    for line in obs.splitlines():
        if "task is to" in line.lower():
            return line.split(":", 1)[-1].strip().rstrip(".").lower()
    return ""


def _format_initial_user(obs: str, admissible: list[str], max_turns: int) -> str:
    """First user message — matches AgentGym's observe() format verbatim:
    `{obs}\\nAVAILABLE ACTIONS: {comma-joined-actions}` (no space after comma).
    No turn marker; the system prompt already explains the format."""
    return f"{obs}\nAVAILABLE ACTIONS: {','.join(admissible)}"


def _build_split_rows(cfg, split: str, n_target: int, max_turns: int) -> list[dict]:
    """Reset env n_target times on a given split, capture initial obs+task per game.

    Each row gets game_index = position in the SORTED game_files list, so
    the Interaction's pinning logic (cycle reset N+1 times on the same sorted
    list) lands on the same game during training.
    """
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    env = AlfredTWEnv(cfg, train_eval=split)
    n_total_before = len(env.game_files)
    # IMPORTANT: must match alfworld_interaction.py's ordering so the parquet's
    # game_index points to the same game the Interaction will load. For train,
    # both call stratified_sample_train_games(seed=42); for val, both sort.
    if split == "train":
        from alfworld_rl.sampling import stratified_sample_train_games
        env.game_files = stratified_sample_train_games(env.game_files)
    else:
        env.game_files.sort()

    n_total = len(env.game_files)
    target = n_total if n_target < 0 else min(n_target, n_total)
    if target < n_total:
        env.game_files = env.game_files[:target]

    print(f"[{split}] {n_total_before} total -> {n_total} after sampling -> {target} target rows")
    print(f"[{split}] first game: {env.game_files[0]}", flush=True)

    batched = env.init_env(batch_size=1)
    print(f"[{split}] init_env done; starting reset loop...", flush=True)
    rows: list[dict] = []
    task_type_counts: dict[str, int] = {}
    for i in range(target):
        if i % 50 == 0:
            print(f"[{split}] reset {i}/{target}", flush=True)
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


@app.function(image=image, volumes={"/output": volume}, timeout=14400)
def build_dataset(
    n_train: int = 500,       # 500 train-split games (subsample of ~3,300 for speed)
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
    # alfworld_xi_prompt: stratified train (504 games) + full 134 val,
    # using AgentGym's verbatim system prompt + observation format.
    # Stratified order matches alfworld_interaction.py exactly (sampling.py, seed=42).
    print(build_dataset.remote(n_train=-1, n_val=134, output_subdir="alfworld_xi_prompt"))
