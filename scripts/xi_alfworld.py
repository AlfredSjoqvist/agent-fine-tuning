"""AlfWorld harness — Xi et al. (2026) reproduction, faithful to AgentGym setup.

================================================================================
SETUP (one-time, requires ~3.5GB game files)
================================================================================
  pip install alfworld textworld pyyaml
  alfworld-download -f      # downloads game files (~3.5GB)
  # AgentGym wrapper is optional for direct play (we use alfworld directly):
  git clone https://github.com/WooooDyy/AgentGym
  cd AgentGym/agentenv-alfworld && pip install --no-deps -e . && cd ../..

================================================================================
RUN
================================================================================
  # Show the system prompt and a synthetic turn 0/1 — no install needed
  python scripts/xi_alfworld.py --show-prompt

  # Interactive play (requires alfworld install + game files)
  python scripts/xi_alfworld.py --play
  python scripts/xi_alfworld.py --play --idx 5      # different game

================================================================================
PAPER NOTES (Xi et al. Section 3.2, Table 1)
================================================================================
  max_turns: 30 (training and rollout). Xi uses K=20 for evaluation.
  reward:    SPARSE / binary — 1.0 on task completion, 0 on timeout
  list_per_step: ✗ (Xi Table 1 ACT). AgentGym's observe() appends the
    admissible-actions list at reset only; step() returns just the
    observation. The system prompt nonetheless tells the LLM "you will
    be given a list of actions ... for each of your turn" — a contract
    violation worth noting when evaluating Xi's methodology.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ============================================================================
# System prompt — verbatim from AgentGym/agentenv/agentenv/envs/alfworld.py
# ============================================================================

SYSTEM_PROMPT = (
    "Interact with a household to solve a task. Imagine you are an intelligent agent "
    "in a household environment and your target is to perform actions to complete the "
    "task goal. At the beginning of your interactions, you will be given the detailed "
    "description of the current environment and your goal to accomplish. For each of "
    "your turn, you will be given a list of actions which you can choose one to "
    "perform in this turn. You should choose from two actions: \"THOUGHT\" or "
    "\"ACTION\". If you choose \"THOUGHT\", you should first think about the current "
    "condition and plan for your future actions, and then output your action in this "
    "turn. Your output must strictly follow this format:\"Thought:\nyour thoughts.\n\n"
    "Action:\nyour next action\"; If you choose \"ACTION\", you should directly output "
    "the action in this turn. Your output must strictly follow this format:\"Action:\n"
    "your next action\". After your each turn, the environment will give you immediate "
    "feedback based on which you plan your next few steps. if the envrionment output "
    "\"Nothing happened\", that means the previous action is invalid and you should try "
    "more options.\n Reminder: \n"
    "1. the action must be chosen from the given available actions. Any actions except "
    "provided available actions will be regarded as illegal. \n"
    "2. Think when necessary, try to act directly more in the process."
)

MAX_TURNS = 30        # Xi Section 3.2
REWARD_TYPE = "sparse / binary (1.0 = task complete, 0 = otherwise)"
LIST_PER_STEP = False  # Xi Table 1 ACT = ✗


# ============================================================================
# Pretty-print helpers (duplicated across xi_*.py for standalone-ness)
# ============================================================================

class Color:
    enabled = True
    @classmethod
    def wrap(cls, t, c):
        if not cls.enabled: return t
        codes = {"red":"31","green":"32","yellow":"33","cyan":"36","gray":"90","bold":"1"}
        return f"\033[{codes[c]}m{t}\033[0m"

def hr(ch="─", n=78): return ch * n

def print_message(role, content):
    print()
    print(Color.wrap(f"┌─ {role.upper()} {hr('─', max(1, 76 - len(role)))}", "cyan"))
    for line in content.split("\n"):
        print(Color.wrap("│ ", "cyan") + line)
    print(Color.wrap(f"└{hr('─', 78)}", "cyan"))

def print_meta(text, color="gray"):
    print(Color.wrap(f"[{text}]", color))


# ============================================================================
# Show-prompt mode (no env install needed)
# ============================================================================

def show_prompt():
    print(Color.wrap(hr("═"), "bold"))
    print(Color.wrap("  ENV: AlfWorld", "bold"))
    print(Color.wrap(hr("═"), "bold"))
    print_meta(f"max_turns={MAX_TURNS}  reward={REWARD_TYPE}  list_per_step={LIST_PER_STEP}", "yellow")
    print_meta("source: Shridhar et al. 2020 (ALFWorld); AgentGym wrapper")
    print_meta(
        "action grammar: free-form TextWorld commands "
        "(go to <recep>, open <recep>, take <obj> from <recep>, put <obj> in <recep>, etc.) "
        "validated against per-game admissible_commands"
    )

    print_message("system (sent once)", SYSTEM_PROMPT)

    syn_turn0 = (
        "You are in the middle of a room. Looking quickly around you, you see a "
        "cabinet 1, a cabinet 2, a coffeemachine 1, a countertop 1, a drawer 1, "
        "a fridge 1, a microwave 1, and a sinkbasin 1.\n"
        "Your task is to: put a hot potato in fridge.\n"
        "AVAILABLE ACTIONS: go to cabinet 1,go to cabinet 2,go to coffeemachine 1,"
        "go to countertop 1,go to drawer 1,go to fridge 1,go to microwave 1,"
        "go to sinkbasin 1,inventory,look"
    )
    print_message("user — turn 0 (synthetic — list IS shown here)", syn_turn0)

    syn_turn1 = (
        "You arrive at countertop 1. On the countertop 1, you see a potato 1, "
        "a knife 1, and a fork 1."
    )
    print_message("user — turn 1 (synthetic — list NOT re-emitted)", syn_turn1)
    print_meta(
        "Note: AgentGym's ALFWorld step() returns just the observation. "
        "The list-of-actions is shown ONLY at reset (turn 0). This is "
        "functionally L_init in our taxonomy."
    )


# ============================================================================
# Interactive play (requires alfworld + game files)
# ============================================================================

def play(idx: int, max_turns: int):
    try:
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    except ImportError as e:
        print(f"ERROR: alfworld not installed.\n  {e}\n\nSee SETUP at top of file.",
              file=sys.stderr); sys.exit(1)

    cfg_candidates = [
        os.path.expanduser("~/alfworld_data/configs/base_config.yaml"),
        os.path.expanduser("~/.cache/alfworld/configs/base_config.yaml"),
        "AgentGym/agentenv-alfworld/agentenv_alfworld/base_config.yaml",
    ]
    cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), None)
    if cfg_path is None:
        print(
            "ERROR: AlfWorld base_config.yaml not found at any of:\n  "
            + "\n  ".join(cfg_candidates) + "\n\n"
            "After running `alfworld-download -f`, you may also need to copy "
            "AgentGym's base_config.yaml into your alfworld data dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    print_meta(f"env=AlfWorld idx={idx} max_turns={max_turns} cfg={cfg_path}", "yellow")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env = AlfredTWEnv(cfg, train_eval="eval_out_of_distribution")
    env.game_files.sort()
    target = env.game_files[idx % len(env.game_files)]
    env.game_files = [target]
    env.num_games = 1
    batched = env.init_env(batch_size=1)
    obs, info = batched.reset()
    admissible = info.get("admissible_commands", [[]])[0]

    print_message("system (paper-faithful)", SYSTEM_PROMPT)
    initial_user = f"{obs[0]}\nAVAILABLE ACTIONS: {','.join(admissible)}"
    print_message("user — turn 0", initial_user)

    cum = 0.0
    for turn in range(max_turns):
        try:
            action = input(Color.wrap(f"\nYour action [turn {turn}/{max_turns}]> ", "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if action.lower() in ("quit", "q", "exit"): break
        if not action: continue
        obs, scores, dones, info = batched.step([action])
        reward = float(scores[0]); done = bool(dones[0])
        delta = reward - cum
        cum = max(cum, reward)
        print_meta(f"reward_delta={delta:+.3f}  cumulative={cum:.3f}  done={done}",
                   "green" if reward > 0 else "red")
        # Paper-faithful: step() returns just the observation, no list re-emit
        print_message(f"user — turn {turn+1}", obs[0])
        if done:
            print_meta("env done", "green" if cum >= 0.99 else "red"); break
    print_meta(f"final cumulative_reward={cum:.3f}", "bold")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("=" * 80)[0])
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--play", action="store_true")
    p.add_argument("--idx", type=int, default=0, help="Game index in sorted eval_out_of_distribution split")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS,
                   help=f"Override paper default ({MAX_TURNS})")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()
    if args.no_color: Color.enabled = False
    if args.show_prompt: show_prompt(); return
    if args.play: play(args.idx, args.max_turns); return
    p.print_help()


if __name__ == "__main__":
    main()
