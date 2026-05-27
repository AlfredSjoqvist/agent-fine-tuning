"""Diagnostic Modal function — just imports BabyAI deps and instantiates one
env. Used to verify the image is functional, separate from verl training.
"""

import modal

app = modal.App("cs224r-babyai-diag")

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
        "gym", "gymnasium<1.0.0", "minigrid<2.5.0", "matplotlib",
        "fastapi", "uvicorn", "pyyaml", "pandas", "pyarrow",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/WooooDyy/AgentGym /opt/agentgym",
        "cd /opt/agentgym/agentenv-babyai && pip install --no-deps -e .",
    )
)


@app.function(image=training_image, timeout=600)
def diag() -> dict:
    """Run inside Modal: does the image actually work?"""
    import sys, traceback
    out = {"steps": []}

    def step(name: str, fn):
        try:
            r = fn()
            out["steps"].append({"name": name, "ok": True, "result": str(r)[:200]})
            print(f"[OK] {name}: {str(r)[:200]}", flush=True)
        except Exception as e:
            out["steps"].append({
                "name": name, "ok": False, "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[FAIL] {name}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    step("import agentenv_babyai", lambda: __import__("agentenv_babyai"))
    step("from agentenv_babyai.environment import BabyAI",
         lambda: __import__("agentenv_babyai.environment", fromlist=["BabyAI"]).BabyAI)
    step("import gymnasium", lambda: __import__("gymnasium"))
    step("import minigrid", lambda: __import__("minigrid"))

    def make_env():
        from agentenv_babyai.environment import BabyAI
        env = BabyAI(max_episode_steps=50, game_name="BabyAI-GoToRedBall-v0", seed=42)
        return f"env={type(env).__name__} obs={env._get_obs()[:120]} actions={env._get_action_space()[:5]}"
    step("instantiate BabyAI env", make_env)

    return out


@app.local_entrypoint()
def main():
    result = diag.remote()
    print("\n=== diagnostic result ===")
    for s in result.get("steps", []):
        marker = "OK" if s.get("ok") else "FAIL"
        print(f"  [{marker}] {s['name']}")
        if not s.get("ok"):
            print(f"    error: {s.get('error')}")
