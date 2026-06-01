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
        "git clone --depth 1 https://github.com/alfworld/alfworld.git /tmp/alfworld_repo",
        "mkdir -p /root/alfworld_data/configs",
        "cp /tmp/alfworld_repo/configs/base_config.yaml /root/alfworld_data/configs/base_config.yaml",
        # ALFWORLD_DATA must be set at build time so alfworld-download writes to the
        # same path that the runtime env var points to.
        "ALFWORLD_DATA=/root/alfworld_data alfworld-download -f",
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
_N_VAL_BABYAI = 50    # randomly sampled from 60 available
_N_VAL_ALFWORLD = 80  # randomly sampled from 134 available


def _subsample_parquet(src: str, n: int, seed: int, dst: str) -> str:
    """Load parquet at src, randomly sample n rows, write to dst, return dst."""
    import pandas as pd
    df = pd.read_parquet(src)
    if n >= len(df):
        df.to_parquet(dst, index=False)
    else:
        df.sample(n=n, random_state=seed).reset_index(drop=True).to_parquet(dst, index=False)
    print(f"subsampled {src}: {len(df)} → {min(n, len(df))} rows → {dst}")
    return dst


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
        # Point default_local_dir at the checkpoint so VERL's resume_mode=auto
        # finds the weights there. save_freq=-1 so nothing new is written.
        cmd = [
            f"trainer.default_local_dir={checkpoint_dir}" if c.startswith("trainer.default_local_dir=") else c
            for c in cmd
        ]
        cmd += ["trainer.resume_mode=auto"]
    return cmd


@app.function(
    image=eval_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=7200,
    memory=131072,
)
def eval_held_in(checkpoint_dir: str | None = None, run_label: str = "base", _val_parquet_override: str | None = None) -> dict:
    """Evaluate on BabyAI val set with full action list (held-in metric).

    Args:
        checkpoint_dir:        path to trained checkpoint. None evaluates the base model.
        run_label:             label for output files.
        _val_parquet_override: if set, use this parquet directly instead of subsampling.
    """
    import json, os, subprocess, time

    os.chdir("/opt/verl")
    run_id = f"held_in_{run_label}_{int(time.time())}"
    log_path = f"/output/eval/{run_id}/stdout.log"
    os.makedirs(f"/output/eval/{run_id}", exist_ok=True)

    if _val_parquet_override:
        val_parquet = _val_parquet_override
        print(f"using override parquet: {val_parquet}")
    else:
        val_parquet = _subsample_parquet(
            "/output/data/babyai_phase1_v1/per_step/val.parquet",
            n=_N_VAL_BABYAI, seed=42,
            dst=f"/output/eval/{run_id}/val_subsample.parquet",
        )
    cmd = _verl_eval_cmd(
        checkpoint_dir=checkpoint_dir,
        interaction_config_path="/root/project/src/babyai_rl/configs/interaction_config/babyai_interaction_per_step.yaml",
        val_parquet=val_parquet,
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

    val_parquet = _subsample_parquet(
        "/output/data/alfworld_xi_prompt/val.parquet",
        n=_N_VAL_ALFWORLD, seed=42,
        dst=f"/output/eval/{run_id}/val_subsample.parquet",
    )
    cmd = _verl_eval_cmd(
        checkpoint_dir=checkpoint_dir,
        interaction_config_path="/root/project/src/alfworld_rl/configs/interaction_config/alfworld_interaction_config.yaml",
        val_parquet=val_parquet,
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


@app.function(
    image=eval_image,
    gpu="H100:8",
    volumes={"/output": volume},
    timeout=7200,
    memory=131072,
)
def eval_held_out_condA(checkpoint_dir: str | None = None, run_label: str = "base", subsample_seed: int = 42, full_list: bool = False) -> dict:
    """Condition A: ALFWorld eval with full action list per turn (held-out).

    Same parquet and system prompt as eval_held_out, but show_action_list_per_turn=True
    so the model sees the admissible-actions list at every turn, not just at episode start.
    Tests whether the 0.013 floor is a format-disambiguation failure rather than a
    weight-level transfer failure.
    """
    import json, os, time

    os.chdir("/opt/verl")
    run_id = f"held_out_condA_{run_label}_{int(time.time())}"
    log_path = f"/output/eval/{run_id}/stdout.log"
    os.makedirs(f"/output/eval/{run_id}", exist_ok=True)

    val_parquet = _subsample_parquet(
        "/output/data/alfworld_xi_prompt/val.parquet",
        n=_N_VAL_ALFWORLD, seed=subsample_seed,
        dst=f"/output/eval/{run_id}/val_subsample.parquet",
    )
    interaction_yaml = (
        "alfworld_interaction_perstep_fulllist.yaml" if full_list
        else "alfworld_interaction_perstep.yaml"
    )
    cmd = _verl_eval_cmd(
        checkpoint_dir=checkpoint_dir,
        interaction_config_path=f"/root/project/src/alfworld_rl/configs/interaction_config/{interaction_yaml}",
        val_parquet=val_parquet,
        run_id=run_id,
    )

    print(f"=== eval held-out condA [{run_label}] (full_list={full_list}) ===")
    if checkpoint_dir:
        print(f"checkpoint: {checkpoint_dir}")
    else:
        print("evaluating base model (no checkpoint)")

    metrics = _run_and_parse(cmd, log_path)
    result = {"run_id": run_id, "split": "held_out_condA", "label": run_label,
              "checkpoint_dir": checkpoint_dir, **metrics}
    with open(f"/output/eval/{run_id}/result.json", "w") as f:
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
            # VERL logs initial val metrics like:
            # step:0 - val-core/babyai/acc/mean@1:0.255 - ...
            # step:0 - val-core/alfworld/acc/mean@1:0.42 - ...
            m = re.search(r"val-core/[^/]+/acc/mean@1:([0-9.]+)", line)
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


@app.function(image=eval_image, timeout=300)
def probe_alfworld_data() -> dict:
    """CPU-only sanity check: load AlfredTWEnv and report game counts per split.
    No GPU, no VERL — runs in ~60s to verify the ALFWorld download path is correct.
    """
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    with open("/root/alfworld_data/configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    results = {}
    for split in ("eval_out_of_distribution",):
        env = AlfredTWEnv(cfg, train_eval=split)
        n = len(env.game_files)
        first = env.game_files[0] if env.game_files else "NONE"
        print(f"split={split!r}: {n} games, first={first}")
        results[split] = {"n_games": n, "first": first}
    return results


@app.local_entrypoint()
def probe_alfworld():
    """Quick CPU-only check that ALFWorld data loaded correctly.

    Usage:
        modal run scripts/modal_eval.py::probe_alfworld
    """
    r = probe_alfworld_data.remote()
    print(r)


@app.local_entrypoint()
def eval_per_level(checkpoint_dir: str | None = None, run_label: str = "base"):
    """Evaluate a checkpoint on GoToRedBall only (17 tasks × 8 rollouts).

    Uses the pre-filtered parquet at data/val_gotoredball.parquet.

    Usage:
        modal run scripts/modal_eval.py::eval_per_level --checkpoint-dir /output/ckpts/<run_id> --run-label per_step
        modal run scripts/modal_eval.py::eval_per_level  # base model
    """
    h = eval_held_in.spawn(
        checkpoint_dir=checkpoint_dir,
        run_label=f"gotoredball_{run_label}",
        _val_parquet_override="/output/data/val_gotoredball.parquet",
    )
    r = h.get()
    print(f"\n=== GoToRedBall only [{run_label}] ===")
    print(f"  success_rate: {r.get('success_rate')}")
    print(f"  result: {r}")


@app.local_entrypoint()
def eval_nolist(checkpoint_dir: str, run_label: str = "trained"):
    """Alias: use modal_interp_eval.py for the no-list interpretability experiment.

    This function is a stub. Run the interpretability eval with:
        modal run scripts/modal_interp_eval.py --checkpoint-dir <path> --run-label <label>
    """
    print("eval_nolist has moved to scripts/modal_interp_eval.py")
    print("Run: modal run scripts/modal_interp_eval.py \\")
    print(f"       --checkpoint-dir {checkpoint_dir} --run-label {run_label}")


@app.local_entrypoint()
def eval_condA(run_label: str = "base", checkpoint_dir: str = "", subsample_seed: int = 42, full_list: bool = False):
    """Run Condition A eval for a single checkpoint. Launch in separate terminal tabs for full visibility.

    Usage:
        # Base model
        modal run scripts/modal_eval.py::eval_condA --run-label base

        # per_step@300
        modal run scripts/modal_eval.py::eval_condA \\
            --run-label per_step_300 \\
            --checkpoint-dir /output/ckpts/babyai_per_step_seed0_steps200_1779869791

        # examples@300
        modal run scripts/modal_eval.py::eval_condA \\
            --run-label examples_300 \\
            --checkpoint-dir /output/ckpts/babyai_examples_seed0_steps200_1779877838
    """
    ckpt = checkpoint_dir if checkpoint_dir else None
    r = eval_held_out_condA.remote(checkpoint_dir=ckpt, run_label=run_label, subsample_seed=subsample_seed, full_list=full_list)
    print(f"\n=== Condition A [{run_label}] (seed={subsample_seed}, full_list={full_list}) ===")
    print(f"  success_rate: {r.get('success_rate')}")
    print(f"  returncode:   {r.get('returncode')}")
    print(f"  run_id:       {r.get('run_id')}")


@app.local_entrypoint()
def eval_condA_base():
    """Condition A baseline: evaluate base model (no checkpoint) with per-turn action list.

    Required data point for interpreting eval_condA_all results — without it you cannot
    separate the effect of training from the effect of the format change itself.

    Usage:
        modal run scripts/modal_eval.py::eval_condA_base
    """
    r = eval_held_out_condA.remote(checkpoint_dir=None, run_label="base")
    print("\n=== Condition A base model smoke result ===")
    print(f"  success_rate: {r.get('success_rate')}")
    print(f"  returncode:   {r.get('returncode')}")
    print(f"  run_id:       {r.get('run_id')}")


@app.local_entrypoint()
def eval_condA_all(per_step_ckpt: str, examples_ckpt: str):
    """Run Condition A eval (show_action_list_per_turn=True) for all three checkpoints in parallel.

    Includes base model as a required baseline — needed to separate the effect of training
    from the effect of the format change (per-turn action list) itself.

    Usage:
        modal run scripts/modal_eval.py::eval_condA_all \\
            --per-step-ckpt ckpts/babyai_per_step_seed0_steps200_1779869791 \\
            --examples-ckpt ckpts/babyai_examples_seed0_steps200_1779877838
    """
    h_base     = eval_held_out_condA.spawn(checkpoint_dir=None,          run_label="base")
    h_perstep  = eval_held_out_condA.spawn(checkpoint_dir=per_step_ckpt, run_label="per_step_300")
    h_examples = eval_held_out_condA.spawn(checkpoint_dir=examples_ckpt, run_label="examples_300")

    r_base = h_base.get()
    r_ps   = h_perstep.get()
    r_ex   = h_examples.get()

    print("\n=== Condition A results (show_action_list_per_turn=True) ===")
    print(f"  base:          {r_base.get('success_rate')}")
    print(f"  per_step@300:  {r_ps.get('success_rate')}")
    print(f"  examples@300:  {r_ex.get('success_rate')}")
    print("\n  (standard held-out for comparison: base=0.013, per_step=0.013, examples=0.013)")


@app.local_entrypoint()
def eval_condA_fulllist(run_label: str, seed: int, checkpoint_dir: str = ""):
    """Condition A with FULL admissible-actions list (no 15-action cap). Use for rounds 2 and 3.

    Usage:
        modal run --detach scripts/modal_eval.py::eval_condA_fulllist --run-label base_rep2 --seed 43
        modal run --detach scripts/modal_eval.py::eval_condA_fulllist --run-label per_step_rep2 --seed 43 --checkpoint-dir /output/ckpts/babyai_per_step_seed0_steps200_1779869791
        modal run --detach scripts/modal_eval.py::eval_condA_fulllist --run-label examples_rep2 --seed 43 --checkpoint-dir /output/ckpts/babyai_examples_seed0_steps200_1779877838
    """
    ckpt = checkpoint_dir if checkpoint_dir else None
    r = eval_held_out_condA.remote(checkpoint_dir=ckpt, run_label=run_label, subsample_seed=seed, full_list=True)
    print(f"\n=== Condition A fulllist [{run_label}] (seed={seed}) ===")
    print(f"  success_rate: {r.get('success_rate')}")
    print(f"  returncode:   {r.get('returncode')}")
    print(f"  run_id:       {r.get('run_id')}")
