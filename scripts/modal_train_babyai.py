"""Modal entrypoint for multi-turn GRPO training on BabyAI.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_train_babyai.py

Entrypoints:
    main              — full experiment: per_step + examples + curriculum in parallel
    smoke             — 1-step sanity check for each condition
    train_curriculum  — curriculum withdrawal only (per_step → examples, two phases)
"""

import modal

app = modal.App("cs224r-babyai-grpo-test")

training_image = (
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
        "BABYAI_TRAJECTORY_DUMP_DIR": "/output/trajectories",
        "TENSORBOARD_DIR": "/output/tb",
        "RAY_DEDUP_LOGS": "0",
    })
    .add_local_dir(
        "/Users/hana/Desktop/project-cs224r",
        remote_path="/root/project",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=86400,
    memory=131072,
)
def train_test(
    condition: str = "per_step",
    total_steps: int = 1,
    seed: int = 0,
    data_root: str = "babyai_v1",
    resume_from: str | None = None,
) -> dict:
    """Run a single-phase GRPO training session.

    Args:
        condition:    "per_step" or "examples". Selects the parquet path AND
                      the interaction config YAML.
        total_steps:  number of GRPO training steps.
        seed:         random seed; included in run_id so concurrent runs don't collide.
        data_root:    parquet directory name under /output/data/.
                      Use "babyai_smoke" for smoke tests, "babyai_v1" for full runs.
        resume_from:  if set, path to a checkpoint directory to resume from.
                      VERL will load the latest checkpoint found there and continue
                      training with the current condition's config.
    """
    import json
    import os
    import subprocess
    import time

    assert condition in ("per_step", "examples"), f"unknown condition: {condition}"

    os.chdir("/opt/verl")

    run_id = f"babyai_{condition}_seed{seed}_steps{total_steps}_{int(time.time())}"
    manifest_dir = f"/output/runs/{run_id}"
    os.makedirs(manifest_dir, exist_ok=True)
    log_path = f"{manifest_dir}/stdout.log"
    os.environ["BABYAI_TRAJECTORY_DUMP_RUN"] = run_id

    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv"], text=True, timeout=10,
        )
    except Exception as e:
        gpu_info = f"nvidia-smi failed: {e}"

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
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0.001",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.response_length=8192",
        "actor_rollout_ref.rollout.prompt_length=2048",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console","tensorboard"]',
        "trainer.project_name=cs224r-interface-rl",
        f"trainer.experiment_name=babyai_{condition}_seed{seed}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.save_freq=5",
        "trainer.test_freq=-1",
        "trainer.total_epochs=100",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.default_local_dir=/output/ckpts/{run_id}",
        f"data.train_files=/output/data/{data_root}/{condition}/train.parquet",
        f"data.val_files=/output/data/{data_root}/{condition}/val.parquet",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path=/root/project/src/babyai_rl/configs/interaction_config/babyai_interaction_{condition}.yaml",
        "custom_reward_function.path=/root/project/src/babyai_rl/reward.py",
        "custom_reward_function.name=compute_score",
    ]

    if resume_from:
        cmd += [
            "trainer.resume_mode=auto",
            f"trainer.resume_training_path={resume_from}",
        ]

    manifest = {
        "run_id": run_id,
        "started_at": time.time(),
        "resume_from": resume_from,
        "cmd": cmd,
        "env": {
            "BABYAI_TRAJECTORY_DUMP_DIR": os.environ.get("BABYAI_TRAJECTORY_DUMP_DIR"),
            "BABYAI_TRAJECTORY_DUMP_RUN": run_id,
            "TENSORBOARD_DIR": os.environ.get("TENSORBOARD_DIR"),
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "RAY_DEDUP_LOGS": os.environ.get("RAY_DEDUP_LOGS"),
        },
        "gpu_info": gpu_info,
    }
    with open(f"{manifest_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()

    print("=" * 80)
    print(f"launching verl training, run_id={run_id}")
    print(f"manifest: {manifest_dir}/manifest.json")
    print(f"stdout log: {log_path}")
    if resume_from:
        print(f"resuming from: {resume_from}")
    print("verl args:")
    for a in cmd:
        print(f"  {a}")
    print("=" * 80, flush=True)

    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd, cwd="/opt/verl",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        rc = proc.wait()

    manifest["ended_at"] = time.time()
    manifest["wall_sec"] = manifest["ended_at"] - manifest["started_at"]
    manifest["returncode"] = rc
    with open(f"{manifest_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()

    return {"run_id": run_id, "returncode": rc, "wall_sec": manifest["wall_sec"]}


@app.local_entrypoint()
def main(total_steps: int = 200, seed: int = 0):
    """Full experiment: per_step + examples run in parallel; curriculum runs sequentially after.

    per_step and examples are independent — spawned in parallel.
    curriculum is two sequential phases (per_step → examples), so it runs after both complete
    (or can be launched separately via train_curriculum).
    """
    CONDITIONS = ("per_step", "examples")
    print(f"=== launching {CONDITIONS} × {total_steps} steps × seed={seed} ===")
    handles = []
    for cond in CONDITIONS:
        h = train_test.spawn(condition=cond, total_steps=total_steps, seed=seed)
        print(f"  spawned {cond}: {h}")
        handles.append((cond, h))

    print("\n=== awaiting per_step + examples ===")
    for cond, h in handles:
        try:
            r = h.get()
            print(f"  [{cond}] done: {r}")
        except Exception as e:
            print(f"  [{cond}] FAILED: {e}")

    print("\n=== launching curriculum (per_step → examples) ===")
    _run_curriculum(total_steps=total_steps, seed=seed)


@app.local_entrypoint()
def train_curriculum(total_steps: int = 200, seed: int = 0):
    """Curriculum withdrawal: per_step for first half, then examples for second half."""
    _run_curriculum(total_steps=total_steps, seed=seed)


def _run_curriculum(total_steps: int, seed: int) -> None:
    """Two-phase curriculum: phase 1 (per_step) → phase 2 (examples from checkpoint)."""
    half = total_steps // 2

    print(f"  curriculum phase 1: per_step × {half} steps")
    r1 = train_test.remote(condition="per_step", total_steps=half, seed=seed)
    print(f"  curriculum phase 1 done: {r1}")

    checkpoint_dir = f"/output/ckpts/{r1['run_id']}"
    print(f"  curriculum phase 2: examples × {half} steps, resuming from {checkpoint_dir}")
    r2 = train_test.remote(
        condition="examples",
        total_steps=half,
        seed=seed,
        resume_from=checkpoint_dir,
    )
    print(f"  curriculum phase 2 done: {r2}")
    print(f"  curriculum complete: phase1_run={r1['run_id']} phase2_run={r2['run_id']}")


@app.local_entrypoint()
def smoke():
    """Smoke test: per_step + examples × 1 step each.
    Run data prep first: modal run scripts/prepare_babyai_data.py::smoke
    Then: modal run scripts/modal_train_babyai.py::smoke
    """
    CONDITIONS = ("per_step", "examples")
    STEPS = 1
    SEED = 0
    print(f"=== smoke test ({CONDITIONS} × {STEPS} step) ===")
    handles = []
    for cond in CONDITIONS:
        h = train_test.spawn(condition=cond, total_steps=STEPS, seed=SEED, data_root="babyai_smoke")
        print(f"  spawned {cond}: {h}")
        handles.append((cond, h))

    print("\n=== awaiting runs ===")
    for cond, h in handles:
        try:
            r = h.get()
            print(f"  [{cond}] done: {r}")
        except Exception as e:
            print(f"  [{cond}] FAILED: {e}")
