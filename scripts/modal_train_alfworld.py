"""Modal entrypoint for multi-turn GRPO training on ALFWorld.

Test run:
  - Qwen2.5-0.5B-Instruct (cheap pipeline validation; swap to 3B for real runs)
  - 8 tasks × GRPO group=4 → 32 rollouts per step
  - ~5 training steps
  - Single A100-80GB
  - Expected wall-clock ~30-60 min, cost ~$1-3

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_train_alfworld.py
"""

import modal

app = modal.App("cs224r-alfworld-grpo-test")

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
        "alfworld",
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
    .env({
        "PYTHONPATH": "/root/project/src:/opt/verl",
        "ALFWORLD_BASE_CONFIG": "/root/alfworld_data/configs/base_config.yaml",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
        "VERL_LOGGING_LEVEL": "INFO",
        "ALFWORLD_TRAJECTORY_DUMP_DIR": "/output/trajectories",
        # tensorboard goes to /output so we can read it post-mortem if the run crashes
        "TENSORBOARD_DIR": "/output/tb",
        # ray dedupes identical log lines across workers — disable so we see all 8
        "RAY_DEDUP_LOGS": "0",
    })
    .add_local_dir(
        "c:/Users/Alfred/Desktop/project-cs224r",
        remote_path="/root/project",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=86400,  # 24hr — my time estimates have been wrong every time, default to long
    memory=131072,  # 128 GB — bigger for 8 GPUs
)
def train_test() -> dict:
    import json
    import os
    import subprocess
    import time

    os.chdir("/opt/verl")

    # Persist a manifest of this run so we can reconcile log file → config later.
    run_id = f"smoke_8xh100_v7_{int(time.time())}"
    manifest_dir = f"/output/runs/{run_id}"
    os.makedirs(manifest_dir, exist_ok=True)
    log_path = f"{manifest_dir}/stdout.log"
    # Pin the trajectory-dump subdir to this run id so we can correlate.
    os.environ["ALFWORLD_TRAJECTORY_DUMP_RUN"] = run_id

    # Snapshot GPU info at launch for the manifest
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv"], text=True, timeout=10,
        )
    except Exception as e:
        gpu_info = f"nvidia-smi failed: {e}"

    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "--config-path", "/root/project/src/alfworld_rl/configs",
        "--config-name", "alfworld_grpo_test",
        "algorithm.adv_estimator=grpo",
        "data.train_batch_size=16",  # PAPER EXACT (was 4 for 1-GPU)
        "data.max_prompt_length=2048",
        "data.max_response_length=16384",  # bumped from 8192: v3 hit cap on 70% of rollouts
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct",  # was 3B; capability test
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",  # PAPER EXACT
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0.001",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",  # was 2; Qwen-3B uses 14.9GB so tp=1 fits, gives 8 inference slots vs 4
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",  # PAPER EXACT (was 0.4)
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.response_length=16384",  # match data.max_response_length
        "actor_rollout_ref.rollout.prompt_length=2048",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console","tensorboard"]',
        "trainer.project_name=cs224r-interface-rl",
        "trainer.experiment_name=smoke_8xh100_v7",  # SMOKE TEST: 1 step, Qwen-7B + Xi prompt + L_init
        "trainer.n_gpus_per_node=8",  # PAPER EXACT (was 1)
        "trainer.nnodes=1",
        "trainer.val_before_train=False",  # SMOKE TEST: skip baseline val (~85min on this hw)
        "trainer.save_freq=1",      # SMOKE TEST: save after each step (verifies ckpt path)
        "trainer.test_freq=-1",     # SMOKE TEST: skip mid-train val
        "trainer.total_epochs=100",
        "trainer.total_training_steps=1",  # SMOKE TEST: 1 step to test Xi-prompt unblocks reward signal
        "trainer.default_local_dir=/output/ckpts/smoke_8xh100_v7",  # SMOKE
        "data.train_files=/output/data/alfworld_xi_prompt/train.parquet",
        "data.val_files=/output/data/alfworld_xi_prompt/val.parquet",
        "actor_rollout_ref.rollout.multi_turn.interaction_config_path=/root/project/src/alfworld_rl/configs/interaction_config/alfworld_interaction_config.yaml",
        "custom_reward_function.path=/root/project/src/alfworld_rl/reward.py",
        "custom_reward_function.name=compute_score",
    ]

    manifest = {
        "run_id": run_id,
        "started_at": time.time(),
        "cmd": cmd,
        "env": {
            "ALFWORLD_TRAJECTORY_DUMP_DIR": os.environ.get("ALFWORLD_TRAJECTORY_DUMP_DIR"),
            "ALFWORLD_TRAJECTORY_DUMP_RUN": run_id,
            "TENSORBOARD_DIR": os.environ.get("TENSORBOARD_DIR"),
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "RAY_DEDUP_LOGS": os.environ.get("RAY_DEDUP_LOGS"),
        },
        "gpu_info": gpu_info,
    }
    with open(f"{manifest_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()  # commit manifest immediately so it survives early crashes

    print("=" * 80)
    print(f"launching verl training, run_id={run_id}")
    print(f"manifest: {manifest_dir}/manifest.json")
    print(f"stdout log: {log_path}")
    print("verl args:")
    for a in cmd:
        print(f"  {a}")
    print("=" * 80, flush=True)

    # Stream subprocess output to BOTH the modal log stream (visible via
    # `modal app logs`) AND a file on the persistent volume (survives a
    # disconnect / log-buffer truncation). Without this, if the modal CLI's
    # log streaming dies mid-run we lose the diagnostic context.
    with open(log_path, "w", buffering=1) as logf:  # line-buffered
        proc = subprocess.Popen(
            cmd, cwd="/opt/verl",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        rc = proc.wait()

    # Final manifest update with end state, then commit volume so
    # everything (logs, ckpts, trajectories, tensorboard) is persisted.
    manifest["ended_at"] = time.time()
    manifest["wall_sec"] = manifest["ended_at"] - manifest["started_at"]
    manifest["returncode"] = rc
    with open(f"{manifest_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()

    return {"run_id": run_id, "returncode": rc, "wall_sec": manifest["wall_sec"]}


@app.local_entrypoint()
def main():
    result = train_test.remote()
    print("\ntraining result:", result)
