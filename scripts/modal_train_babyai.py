"""Modal entrypoint for multi-turn GRPO training on BabyAI.

This is a structural mirror of modal_train_alfworld.py, with the env layer
swapped to BabyAI (via AgentGym's babyai wrapper, which gives high-level
"go to X / pick up Y / open door Z" actions over the underlying minigrid env).

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_train_babyai.py
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
        # BabyAI deps from AgentGym's agentenv-babyai/pyproject.toml
        "gym",
        "gymnasium<1.0.0",
        "minigrid<2.5.0",
        "matplotlib",
        "fastapi",
        "uvicorn",
        # Our own deps
        "pyyaml",
        "pandas",
        "pyarrow",
    )
    .run_commands(
        # Clone AgentGym to get its agentenv_babyai package (provides the
        # high-level wrapper over minigrid that emits "go to red ball 1"-style
        # actions). We install it editable from the repo path.
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
        "c:/Users/Alfred/Desktop/project-cs224r",
        remote_path="/root/project",
    )
)

volume = modal.Volume.from_name("cs224r-interface-rl", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=86400,  # 24hr — runtime estimates have been wrong every time
    memory=131072,
)
def train_test() -> dict:
    import json
    import os
    import subprocess
    import time

    os.chdir("/opt/verl")

    run_id = f"babyai_smoke_v1_{int(time.time())}"
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
        "trainer.experiment_name=babyai_smoke_v1",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.save_freq=1",
        "trainer.test_freq=-1",
        "trainer.total_epochs=100",
        "trainer.total_training_steps=1",
        "trainer.default_local_dir=/output/ckpts/babyai_smoke_v1",
        "data.train_files=/output/data/babyai_v1/train.parquet",
        "data.val_files=/output/data/babyai_v1/val.parquet",
        "actor_rollout_ref.rollout.multi_turn.interaction_config_path=/root/project/src/babyai_rl/configs/interaction_config/babyai_interaction_config.yaml",
        "custom_reward_function.path=/root/project/src/babyai_rl/reward.py",
        "custom_reward_function.name=compute_score",
    ]

    manifest = {
        "run_id": run_id,
        "started_at": time.time(),
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
def main():
    result = train_test.remote()
    print("\ntraining result:", result)
