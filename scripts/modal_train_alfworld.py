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
    })
    .add_local_dir(
        "/Users/hana/Desktop/224project",
        remote_path="/root/project",
    )
)

volume = modal.Volume.from_name("224project-data", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="A100-80GB:1",
    volumes={"/output": volume},
    timeout=10800,
    memory=65536,
)
def train_test() -> dict:
    import os
    import subprocess

    os.chdir("/opt/verl")

    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "--config-path", "/root/project/src/alfworld_rl/configs",
        "--config-name", "alfworld_grpo_test",
        "algorithm.adv_estimator=grpo",
        "data.train_batch_size=12",
        "data.max_prompt_length=2048",
        "data.max_response_length=8192",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=12",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0.001",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.response_length=8192",
        "actor_rollout_ref.rollout.prompt_length=2048",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=224project-alfworld",
        "trainer.experiment_name=alfworld_learning_v2_qwen3b",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        "trainer.total_epochs=100",
        "trainer.total_training_steps=20",
        "trainer.default_local_dir=/output/ckpts/alfworld_learning_v2",
        "data.train_files=/output/data/alfworld_pnp_v2/train.parquet",
        "data.val_files=/output/data/alfworld_pnp_v2/val.parquet",
        "actor_rollout_ref.rollout.multi_turn.interaction_config_path=/root/project/src/alfworld_rl/configs/interaction_config/alfworld_interaction_config.yaml",
        "custom_reward_function.path=/root/project/src/alfworld_rl/reward.py",
        "custom_reward_function.name=compute_score",
    ]

    print("=" * 80)
    print("launching verl training with:")
    for a in cmd:
        print(f"  {a}")
    print("=" * 80)

    result = subprocess.run(cmd, cwd="/opt/verl")
    volume.commit()

    return {"returncode": result.returncode}


@app.local_entrypoint()
def main():
    result = train_test.remote()
    print("\ntraining result:", result)
