"""Diagnostic: exercise AlfworldInteraction directly, 10 turns, canned actions.

Bypasses verl's agent loop entirely — answers the question 'is the Interaction itself
capable of running 10 turns, or does it terminate early?'. If this runs 10 turns clean,
the num_turns=2 limit in the training run comes from verl's loop or SGLang's generation
termination, not our code.

Run:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run scripts/modal_diag_interaction.py
"""

import modal

app = modal.App("cs224r-diag-interaction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget", "ca-certificates")
    .pip_install(
        "alfworld",
        "textworld",
        "pyyaml",
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


@app.function(image=image, timeout=300)
def diag() -> dict:
    """Run AlfworldInteraction with mock model output; log every turn."""
    import asyncio
    import sys

    # Skip verl import since we're bypassing it — patch the BaseInteraction import
    # in alfworld_interaction module by providing a stub.
    class _StubBaseInteraction:
        def __init__(self, config):
            self.config = config
            self.name = config.get("name", "diag")

    stub_module = type(sys)("verl.interactions.base")
    stub_module.BaseInteraction = _StubBaseInteraction
    parent = type(sys)("verl.interactions")
    parent.base = stub_module
    root = type(sys)("verl")
    root.interactions = parent
    sys.modules["verl"] = root
    sys.modules["verl.interactions"] = parent
    sys.modules["verl.interactions.base"] = stub_module

    from alfworld_rl.alfworld_interaction import AlfworldInteraction

    interaction = AlfworldInteraction(
        {"split": "eval_out_of_distribution", "max_turns": 30, "name": "alfworld"}
    )
    print(f"[config check] interaction.max_turns = {interaction.max_turns}")

    async def run_rollout():
        instance_id = await interaction.start_interaction(game_index=0)
        state = interaction._instance_dict[instance_id]
        print(f"[start] instance={instance_id}")
        print(f"[start] initial obs (first 400 chars):\n{state['obs'][:400]}")
        print(f"[start] {len(state['admissible'])} admissible commands")
        print(f"[start] first 5 admissible: {state['admissible'][:5]}")

        log = []
        for i in range(35):
            # Mock the assistant's last message: try a real admissible command each turn
            if state["admissible"]:
                mock_action = state["admissible"][i % len(state["admissible"])]
            else:
                mock_action = "look"
            messages = [{"role": "assistant", "content": f"Action: {mock_action}"}]

            should_terminate, next_obs, reward, extra = await interaction.generate_response(
                instance_id, messages
            )
            log.append({
                "step": i,
                "action_sent": extra["action_sent"],
                "reward": reward,
                "env_done": extra["env_done"],
                "should_terminate": should_terminate,
                "obs_snippet": next_obs[:120],
            })
            print(
                f"[turn {i}] action={extra['action_sent']!r} reward={reward:.2f} "
                f"env_done={extra['env_done']} terminate={should_terminate}"
            )
            if should_terminate:
                print(f"[turn {i}] TERMINATED at step {i}, reason=env_done={extra['env_done']}, "
                      f"turn_counter={interaction._instance_dict[instance_id]['turn']}")
                break

        await interaction.finalize_interaction(instance_id)
        return {
            "steps_executed": len(log),
            "last_entry": log[-1] if log else None,
            "terminated_early": log and log[-1]["should_terminate"],
        }

    return asyncio.run(run_rollout())


@app.local_entrypoint()
def main():
    print(diag.remote())
