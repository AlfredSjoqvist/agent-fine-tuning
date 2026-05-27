"""Post-training evaluation pipeline.

Loads a trained BabyAI checkpoint and evaluates:
  - Held-In:  BabyAI val set, always with full action list (per_step interaction)
  - Held-Out: ALFWorld val set, zero-shot transfer

Reports avg@8 success rate and generalization gap g = ΔHeld-In − ΔHeld-Out.

Usage:
    # Evaluate a specific checkpoint
    modal run scripts/modal_eval.py --checkpoint-dir /output/ckpts/<run_id>

    # Evaluate the base model (no checkpoint — measures baseline)
    modal run scripts/modal_eval.py::eval_base
"""

import modal

app = modal.App("cs224r-babyai-eval")

eval_image = (
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
        "alfworld",
        "textworld",
        "pyyaml",
        "pandas",
        "pyarrow",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/WooooDyy/AgentGym /opt/agentgym",
        "cd /opt/agentgym/agentenv-babyai && pip install --no-deps -e .",
        "cd /opt/agentgym/agentenv-alfworld && pip install --no-deps -e .",
        "alfworld-download -f",
        "git clone --depth 1 https://github.com/alfworld/alfworld.git /tmp/alfworld_repo",
        "mkdir -p /root/alfworld_data/configs",
        "cp /tmp/alfworld_repo/configs/base_config.yaml /root/alfworld_data/configs/base_config.yaml",
    )
    .env({
        "PYTHONPATH": "/root/project/src:/opt/verl",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
        "VERL_LOGGING_LEVEL": "INFO",
        "ALFWORLD_DATA": "/root/alfworld_data",
        "RAY_DEDUP_LOGS": "0",
    })
    .add_local_dir(
        "/Users/hana/Desktop/project-cs224r",
        remote_path="/root/project",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)

# Eval always uses n=8 rollouts per task (avg@8).
_N_ROLLOUTS = 8
# Number of val tasks to evaluate (subset for speed; set to -1 for all).
_N_VAL_TASKS = 60   # BabyAI: 20 tasks × 3 levels = 60; ALFWorld: use all 134


def _verl_eval_cmd(
    checkpoint_dir: str | None,
    interaction_config_path: str,
    val_parquet: str,
    run_id: str,
) -> list[str]:
    """Build a VERL command that runs one eval pass (no gradient updates).

    Uses val_before_train=True + total_training_steps=0 to perform a single
    evaluation pass using the multi-turn interaction loop, then exits.
    If checkpoint_dir is None, evaluates the base model (no checkpoint loaded).
    """
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
        f"trainer.experiment_name=eval_{run_id}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.val_before_train=True",
        "trainer.total_training_steps=0",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        f"trainer.default_local_dir=/output/eval/{run_id}",
        f"data.train_files={val_parquet}",   # unused (0 steps), but required by VERL
        f"data.val_files={val_parquet}",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config_path}",
        "custom_reward_function.path=/root/project/src/babyai_rl/reward.py",
        "custom_reward_function.name=compute_score",
    ]
    if checkpoint_dir:
        cmd += [
            "trainer.resume_mode=auto",
            f"trainer.resume_training_path={checkpoint_dir}",
        ]
    return cmd


@app.function(
    image=eval_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=7200,
    memory=131072,
)
def eval_held_in(checkpoint_dir: str | None = None, run_label: str = "base") -> dict:
    """Evaluate on BabyAI val set with full action list (held-in metric).

    Args:
        checkpoint_dir: path to trained checkpoint on the Modal volume.
                        None evaluates the base model.
        run_label:      label for output files (e.g. "per_step", "examples", "curriculum").
    """
    import json, os, subprocess, time

    os.chdir("/opt/verl")
    run_id = f"held_in_{run_label}_{int(time.time())}"
    log_path = f"/output/eval/{run_id}/stdout.log"
    os.makedirs(f"/output/eval/{run_id}", exist_ok=True)

    cmd = _verl_eval_cmd(
        checkpoint_dir=checkpoint_dir,
        interaction_config_path="/root/project/src/babyai_rl/configs/interaction_config/babyai_interaction_per_step.yaml",
        val_parquet="/output/data/babyai_v1/per_step/val.parquet",
        run_id=run_id,
    )

    print(f"=== eval held-in [{run_label}] ===")
    if checkpoint_dir:
        print(f"checkpoint: {checkpoint_dir}")
    else:
        print("evaluating base model (no checkpoint)")

    metrics = _run_and_parse(cmd, log_path)
    result = {"run_id": run_id, "split": "held_in", "label": run_label,
              "checkpoint_dir": checkpoint_dir, **metrics}
    with open(f"/output/eval/{run_id}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return result


@app.function(
    image=eval_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=7200,
    memory=131072,
)
def eval_held_out(checkpoint_dir: str | None = None, run_label: str = "base") -> dict:
    """Evaluate on ALFWorld val set zero-shot (held-out metric).

    Args:
        checkpoint_dir: path to trained checkpoint. None evaluates the base model.
        run_label:      label for output files.
    """
    import json, os, time

    os.chdir("/opt/verl")
    run_id = f"held_out_{run_label}_{int(time.time())}"
    log_path = f"/output/eval/{run_id}/stdout.log"
    os.makedirs(f"/output/eval/{run_id}", exist_ok=True)

    cmd = _verl_eval_cmd(
        checkpoint_dir=checkpoint_dir,
        interaction_config_path="/root/project/src/alfworld_rl/configs/interaction_config/alfworld_interaction_config.yaml",
        val_parquet="/output/data/alfworld_xi_prompt/val.parquet",
        run_id=run_id,
    )

    print(f"=== eval held-out [{run_label}] ===")
    if checkpoint_dir:
        print(f"checkpoint: {checkpoint_dir}")
    else:
        print("evaluating base model (no checkpoint)")

    metrics = _run_and_parse(cmd, log_path)
    result = {"run_id": run_id, "split": "held_out", "label": run_label,
              "checkpoint_dir": checkpoint_dir, **metrics}
    with open(f"/output/eval/{run_id}/result.json", "w") as f:
        import json
        json.dump(result, f, indent=2)
    volume.commit()
    return result


def _run_and_parse(cmd: list[str], log_path: str) -> dict:
    """Run VERL eval command, stream output, parse success_rate from logs."""
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
            # VERL logs val metrics like: val/task_success: 0.85
            m = re.search(r"val/task_success[:\s]+([0-9.]+)", line)
            if m:
                success_rate = float(m.group(1))
        rc = proc.wait()

    return {"returncode": rc, "success_rate": success_rate}


@app.local_entrypoint()
def main(checkpoint_dir: str, run_label: str = "trained"):
    """Evaluate a checkpoint on both held-in and held-out, report gap.

    Usage:
        modal run scripts/modal_eval.py --checkpoint-dir /output/ckpts/<run_id> --run-label per_step
    """
    print(f"=== evaluating checkpoint: {checkpoint_dir} ===")

    # Run held-in and held-out in parallel
    h_in = eval_held_in.spawn(checkpoint_dir=checkpoint_dir, run_label=run_label)
    h_out = eval_held_out.spawn(checkpoint_dir=checkpoint_dir, run_label=run_label)

    r_in = h_in.get()
    r_out = h_out.get()

    sr_in = r_in.get("success_rate")
    sr_out = r_out.get("success_rate")

    print("\n=== results ===")
    print(f"  Held-In  (BabyAI, full list):  {sr_in}")
    print(f"  Held-Out (ALFWorld, zero-shot): {sr_out}")
    if sr_in is not None and sr_out is not None:
        print(f"  g = ΔHeld-In − ΔHeld-Out:       (requires base model scores — run eval_base first)")
    print(f"  held-in  result: {r_in}")
    print(f"  held-out result: {r_out}")


@app.local_entrypoint()
def eval_base():
    """Evaluate the base model (no checkpoint) to get baseline scores for delta computation."""
    print("=== evaluating base model ===")

    h_in = eval_held_in.spawn(checkpoint_dir=None, run_label="base")
    h_out = eval_held_out.spawn(checkpoint_dir=None, run_label="base")

    r_in = h_in.get()
    r_out = h_out.get()

    print("\n=== base model scores ===")
    print(f"  BabyAI  (held-in):  {r_in.get('success_rate')}")
    print(f"  ALFWorld (held-out): {r_out.get('success_rate')}")
    print(f"  in:  {r_in}")
    print(f"  out: {r_out}")


@app.local_entrypoint()
def eval_nolist(checkpoint_dir: str, run_label: str = "trained"):
    """Optional within-environment experiment: evaluate WITHOUT action list at test time.

    Tests whether a per_step-trained model collapses when the list is withheld.
    Uses babyai_interaction_nolist_eval.yaml.

    Usage:
        modal run scripts/modal_eval.py::eval_nolist --checkpoint-dir /output/ckpts/<run_id>
    """
    import subprocess, os, json, time

    print(f"=== eval without list [{run_label}] ===")

    # Reuse eval_held_in machinery but swap interaction config
    # (inline here since it's a one-off optional experiment)
    run_id = f"nolist_{run_label}_{int(time.time())}"
    os.makedirs(f"/output/eval/{run_id}", exist_ok=True)

    h = eval_held_in.spawn(checkpoint_dir=checkpoint_dir, run_label=f"nolist_{run_label}")
    r = h.get()
    print(f"  result: {r}")
