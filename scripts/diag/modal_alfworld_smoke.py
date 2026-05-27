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
    .run_commands(
        "alfworld-download -f",
        "git clone --depth 1 https://github.com/alfworld/alfworld.git /tmp/alfworld_repo",
        "mkdir -p /root/alfworld_data/configs",
        "cp /tmp/alfworld_repo/configs/base_config.yaml /root/alfworld_data/configs/base_config.yaml",
    )
)


def _find_base_config() -> str:
    import os

    import alfworld

    pkg_dir = os.path.dirname(alfworld.__file__)
    search_roots = [pkg_dir, os.path.dirname(pkg_dir), os.environ.get("ALFWORLD_DATA", "")]
    for root in search_roots:
        if not root:
            continue
        for dirpath, _, files in os.walk(root):
            if "base_config.yaml" in files:
                return os.path.join(dirpath, "base_config.yaml")
    raise FileNotFoundError(f"base_config.yaml not found under {search_roots}")


@app.function(image=alfworld_image, timeout=600)
def test_alfworld() -> dict:
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config_path = _find_base_config()
    print(f"found config at: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
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
        "env_type": "AlfredTWEnv",
        "n_admissible_commands": len(admissible),
        "first_action": first_action,
        "score_after_step": scores[0],
        "done_after_step": dones[0],
    }


@app.local_entrypoint()
def main():
    result = test_alfworld.remote()
    print("\nalfworld smoke test result:", result)
