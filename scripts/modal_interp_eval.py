"""Interpretability experiment: evaluate a trained checkpoint WITHOUT the action list.

Tests whether a per_step-trained model collapses when the action list is withheld
at test time. Answers: did the model learn to rely on the list as a crutch, or did
it internalize navigation policy independently?

Compares:
  - per_step checkpoint evaluated WITH list    → score_with_list  (run via modal_eval.py)
  - per_step checkpoint evaluated WITHOUT list → score_nolist     (this script)

Usage:
    modal run scripts/modal_interp_eval.py \
      --checkpoint-dir /output/ckpts/babyai_per_step_seed0_steps300_<ts> \
      --run-label per_step_300
"""

import modal

app = modal.App("cs224r-interp-eval")

interp_image = (
    modal.Image.from_registry(
        "verlai/verl:sgl059.latest",
        add_python=None,
    )
    .apt_install("git", "build-essential", "wget", "ca-certificates")
    .run_commands(
        "git clone --depth 1 --branch v0.7.1 https://github.com/volcengine/verl.git /opt/verl",
        "cd /opt/verl && pip install --no-deps -e .",
    )
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
        "PYTHONPATH": "/root/project/src:/opt/verl",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
        "VERL_LOGGING_LEVEL": "INFO",
        "RAY_DEDUP_LOGS": "0",
    })
    .add_local_dir(
        "/Users/hana/Desktop/project-cs224r",
        remote_path="/root/project",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)

_N_ROLLOUTS = 8
_N_VAL_BABYAI = 50


def _subsample_parquet(src: str, n: int, seed: int, dst: str) -> str:
    import pandas as pd
    df = pd.read_parquet(src)
    if n >= len(df):
        df.to_parquet(dst, index=False)
    else:
        df.sample(n=n, random_state=seed).reset_index(drop=True).to_parquet(dst, index=False)
    print(f"subsampled {src}: {len(df)} → {min(n, len(df))} rows → {dst}")
    return dst


def _run_and_parse(cmd: list[str], log_path: str) -> dict:
    import re, subprocess

    print("verl args:")
    for a in cmd:
        print(f"  {a}")

    success_rate = None
    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd, cwd="/opt/verl",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
            m = re.search(r"val-core/[^/]+/acc/mean@1:([0-9.]+)", line)
            if m:
                success_rate = float(m.group(1))
        rc = proc.wait()

    return {"returncode": rc, "success_rate": success_rate}


_PARQUET_BY_CONDITION = {
    "per_step": "/output/data/babyai_phase1_v1/per_step/val.parquet",
    "examples": "/output/data/babyai_phase1_v1/examples/val.parquet",
}


@app.function(
    image=interp_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=7200,
    memory=131072,
)
def eval_nolist_fn(
    checkpoint_dir: str | None = None,
    run_label: str = "base",
    condition: str = "per_step",
    val_parquet_override: str | None = None,
) -> dict:
    """Run BabyAI eval with show_action_list_per_turn=False.

    Uses babyai_interaction_nolist_eval.yaml so the model receives observations
    without the available-action list appended each turn.

    Args:
        checkpoint_dir:       path to trained checkpoint. None = base model (no checkpoint).
        run_label:            label for output files.
        condition:            "per_step" or "examples" — selects the val parquet so
                              the system prompt matches what the model was trained on.
                              Use "per_step" for the base model baseline.
        val_parquet_override: if set, use this parquet directly, ignoring condition.
    """
    import json, os, time

    assert condition in ("per_step", "examples"), f"unknown condition: {condition}"

    os.chdir("/opt/verl")
    run_id = f"nolist_{run_label}_{int(time.time())}"
    log_path = f"/output/eval/{run_id}/stdout.log"
    os.makedirs(f"/output/eval/{run_id}", exist_ok=True)

    if val_parquet_override:
        val_parquet = val_parquet_override
        print(f"using override parquet: {val_parquet}")
    else:
        src = _PARQUET_BY_CONDITION[condition]
        val_parquet = _subsample_parquet(
            src,
            n=_N_VAL_BABYAI, seed=42,
            dst=f"/output/eval/{run_id}/val_subsample.parquet",
        )
        print(f"condition={condition!r} → parquet: {src}")

    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "--config-path", "/root/project/src/babyai_rl/configs",
        "--config-name", "babyai_grpo_test",
        "algorithm.adv_estimator=grpo",
        "data.train_batch_size=16",
        "data.max_prompt_length=2048",
        "data.max_response_length=8192",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        f"actor_rollout_ref.rollout.n={_N_ROLLOUTS}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.response_length=8192",
        "actor_rollout_ref.rollout.prompt_length=2048",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=cs224r-interface-rl",
        f"trainer.experiment_name=interp_nolist_{run_label}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.val_before_train=True",
        "trainer.total_training_steps=0",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        f"trainer.default_local_dir=/output/eval/{run_id}",
        f"data.train_files={val_parquet}",
        f"data.val_files={val_parquet}",
        # No-list: show_action_list_per_turn=False in this YAML
        "actor_rollout_ref.rollout.multi_turn.interaction_config_path="
        "/root/project/src/babyai_rl/configs/interaction_config/babyai_interaction_nolist_eval.yaml",
        "custom_reward_function.path=/root/project/src/babyai_rl/reward.py",
        "custom_reward_function.name=compute_score",
    ]
    if checkpoint_dir:
        cmd = [
            f"trainer.default_local_dir={checkpoint_dir}" if c.startswith("trainer.default_local_dir=") else c
            for c in cmd
        ]
        cmd += ["trainer.resume_mode=auto"]

    print(f"=== eval no-list [{run_label}] ===")
    print(f"checkpoint: {checkpoint_dir or 'base model (no checkpoint)'}")
    print(f"condition (parquet): {condition}")
    print("interaction: babyai_interaction_nolist_eval.yaml (show_action_list_per_turn=False)")

    metrics = _run_and_parse(cmd, log_path)
    result = {
        "run_id": run_id,
        "split": "held_in_nolist",
        "label": run_label,
        "condition": condition,
        "checkpoint_dir": checkpoint_dir,
        **metrics,
    }
    with open(f"/output/eval/{run_id}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return result


@app.local_entrypoint()
def main(checkpoint_dir: str, run_label: str = "trained", condition: str = "per_step"):
    """Evaluate a checkpoint without the action list (interpretability experiment).

    The condition param must match the training condition of the checkpoint so the
    correct val parquet (and system prompt) is used:
      - "per_step"  → per_step val parquet (list-based system prompt)
      - "examples"  → examples val parquet (format-examples system prompt)

    Usage:
        # per_step checkpoint:
        modal run scripts/modal_interp_eval.py \\
          --checkpoint-dir /output/ckpts/babyai_per_step_seed0_steps200_1779869791 \\
          --run-label per_step_300 --condition per_step

        # examples checkpoint:
        modal run scripts/modal_interp_eval.py \\
          --checkpoint-dir /output/ckpts/babyai_examples_seed0_steps200_1779877838 \\
          --run-label examples_300 --condition examples
    """
    print(f"=== interpretability eval (no list) ===")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  label:      {run_label}")
    print(f"  condition:  {condition}")

    r = eval_nolist_fn.remote(
        checkpoint_dir=checkpoint_dir,
        run_label=run_label,
        condition=condition,
    )

    print(f"\n=== result ===")
    print(f"  success_rate (no list): {r.get('success_rate')}")
    print(f"  full result: {r}")
    print()
    print("To compare, get the held-in (with list) score from modal_eval.py:")
    print(f"  modal run scripts/modal_eval.py --checkpoint-dir {checkpoint_dir} --run-label {run_label}")


@app.local_entrypoint()
def eval_nolist_base():
    """Nolist baseline: evaluate the base model (no training) without the action list.

    This is the correct baseline for comparing per_step and examples nolist scores.
    Uses the per_step parquet (list-based system prompt) as the neutral reference.

    Usage:
        modal run scripts/modal_interp_eval.py::eval_nolist_base
    """
    print("=== nolist baseline (base model, no checkpoint) ===")
    r = eval_nolist_fn.remote(checkpoint_dir=None, run_label="base", condition="per_step")
    print(f"\n  success_rate (nolist, base model): {r.get('success_rate')}")
    print(f"  full result: {r}")
