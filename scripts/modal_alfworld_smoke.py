"""Smoke test: ALFWorld inside a Modal container.

Confirms:
  - ALFWorld pip install succeeds
  - Game data downloads to $ALFWORLD_DATA (baked into image)
  - A task loads and returns an English observation
  - A step() transitions the state and returns admissible_commands

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_alfworld_smoke.py

Expected wall-clock: ~3-5 min first run (image build + data download),
~15 s subsequent (cached image).
"""

import modal

app = modal.App("cs224r-alfworld-smoke")

alfworld_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget", "ca-certificates")
    .pip_install(
        "alfworld[full]",
        "textworld",
        "pyyaml",
    )
    .env({"ALFWORLD_DATA": "/root/alfworld_data"})
    .run_commands("alfworld-download -f")
)


@app.function(image=alfworld_image, timeout=600)
def test_alfworld() -> dict:
    import os

    import alfworld.agents.environment as environment
    import yaml

    config_path = f"{os.environ['ALFWORLD_DATA']}/configs/base_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_type = config["env"]["type"]
    print(f"env type from config: {env_type}")

    env = getattr(environment, env_type)(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    obs, info = env.reset()
    print("=" * 60)
    print("INITIAL OBSERVATION:")
    print(obs[0])
    print("=" * 60)

    admissible = info["admissible_commands"][0]
    print(f"\n{len(admissible)} admissible commands. First 10:")
    for cmd in admissible[:10]:
        print(f"  - {cmd}")

    first_action = admissible[0] if admissible else "look"
    obs2, scores, dones, info2 = env.step([first_action])
    print("\n" + "=" * 60)
    print(f"AFTER ACTION '{first_action}':")
    print(obs2[0])
    print("=" * 60)
    print(f"score={scores[0]}, done={dones[0]}")

    return {
        "env_type": env_type,
        "n_admissible_commands": len(admissible),
        "first_action": first_action,
        "score_after_step": scores[0],
        "done_after_step": dones[0],
    }


@app.local_entrypoint()
def main():
    result = test_alfworld.remote()
    print("\nalfworld smoke test result:", result)
