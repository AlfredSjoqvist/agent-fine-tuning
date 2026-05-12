"""BabyAI harness — Xi et al. (2026) reproduction, faithful to AgentGym setup.

System prompt + observation format + max_turns are taken verbatim from
AgentGym (commit d014732d9f...) and Xi et al. Section 3.2. This file is
self-contained — run it directly.

================================================================================
SETUP (one-time, ~3 min)
================================================================================
  pip install gym "gymnasium<1.0.0" "minigrid<2.5.0" matplotlib fastapi uvicorn
  git clone https://github.com/WooooDyy/AgentGym
  cd AgentGym/agentenv-babyai
  pip install --no-deps -e .
  cd ../..

================================================================================
RUN
================================================================================
  # Show the system prompt and a synthetic turn 0/1 — no install needed
  python scripts/xi_babyai.py --show-prompt

  # Interactive play (env install required)
  python scripts/xi_babyai.py --play
  python scripts/xi_babyai.py --play --level BabyAI-Pickup-v0 --seed 7
  python scripts/xi_babyai.py --play --level BabyAI-GoToLocal-v0 --seed 42

================================================================================
PAPER NOTES (Xi et al. Section 3.2, Table 1)
================================================================================
  max_turns: 10 (training and rollout). Xi uses K=20 for evaluation.
  reward:    DENSE — 1 - 0.9 * (step_count / max_steps) on success, 0 on timeout
  list_per_step: ✓ — BabyAI is the ONLY env in Xi's lineup where the
                     env emits a per-turn admissible-action list. This is
                     the feature Xi attributes BabyAI's negative cross-env
                     transfer to (Section 5, p. 7).
"""

import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ============================================================================
# System prompt — verbatim from AgentGym/agentenv/agentenv/envs/babyai.py
# ============================================================================

SYSTEM_PROMPT = (
    "You are an exploration master that wants to finish every goal you are given. "
    "Every round I will give you an observation, and you have to respond an action "
    "and your thought based on the observation to finish the given task. You are "
    "placed in a room and you need to accomplish the given goal with actions.\n\n"
    "You can use the following actions: \n\n"
    "- turn right \n\n"
    "- turn left \n\n"
    "- move forward \n\n"
    "- go to <obj> <id> \n\n"
    "- pick up <obj> <id> \n\n"
    "- go through <door> <id>: <door> must be an open door. \n\n"
    "- toggle and go through <door> <id>: <door> can be a closed door or a locked door. "
    "If you want to open a locked door, you need to carry a key that is of the same color "
    "as the locked door. \n\n"
    "- toggle: there is a closed or locked door right in front of you and you can toggle it.\n"
    "Your response should use the following format:\n"
    "Thought:\n"
    "<Your Thought>\n"
    "\n"
    "Action:\n"
    "<Your Action>"
)

MAX_TURNS = 10        # Xi Section 3.2
REWARD_TYPE = "dense (1 - 0.9 * step_count / max_steps on success; 0 on timeout)"
LIST_PER_STEP = True  # Xi Table 1 ACT = ✓


# ============================================================================
# Action surface form normalization (AgentGym uses 'pickup', LLMs say 'pick up')
# ============================================================================

def rewrite_list_for_llm(admissible):
    return [re.sub(r"^pickup\s+", "pick up ", a) for a in admissible]

def normalize_action_for_env(action):
    a = action.strip()
    a = re.sub(r"^pick\s+up\s+", "pickup ", a, flags=re.IGNORECASE)
    a = re.sub(r"^pick-up\s+", "pickup ", a, flags=re.IGNORECASE)
    return a


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
    print(Color.wrap("  ENV: BabyAI", "bold"))
    print(Color.wrap(hr("═"), "bold"))
    print_meta(f"max_turns={MAX_TURNS}  reward={REWARD_TYPE}  list_per_step={LIST_PER_STEP}", "yellow")
    print_meta("source: Chevalier-Boisvert et al. 2018 (BabyAI); minigrid 2.3.1; AgentGym wrapper")
    print_meta(
        "action grammar: turn left | turn right | move forward | "
        "go to <obj> <id> | pick up <obj> <id> | go through <door> <id> | "
        "toggle and go through <door> <id> | toggle | check available actions"
    )

    print_message("system (sent once)", SYSTEM_PROMPT)

    syn_turn0 = (
        "Goal: pick up the yellow key\n\n"
        "In front of you in this room, you can see several objects: There is a "
        "yellow key 1 1 steps in front of you and 1 steps to your left.  There "
        "is a red box 1 1 steps in front of you and 1 steps to your right.  The "
        "room has walls around you. You are facing a wall 3 steps away. You are "
        "not carrying anything.\n"
        "Available actions: ['turn left', 'turn right', 'move forward', "
        "'pick up yellow key 1', 'pick up red box 1', 'go to yellow key 1', "
        "'go to red box 1', 'check available actions']"
    )
    print_message("user — turn 0 (synthetic example)", syn_turn0)

    syn_turn1 = (
        "In front of you in this room, you can see several objects: There is a "
        "yellow key 1 right in front of you 1 steps away.  There is a red box "
        "1 1 steps to your right.\n"
        "Available actions: ['turn left', 'turn right', 'move forward', "
        "'pick up yellow key 1', 'pick up red box 1', 'go to yellow key 1', "
        "'go to red box 1', 'check available actions']"
    )
    print_message("user — turn 1 (synthetic — list IS re-emitted in BabyAI)", syn_turn1)
    print_meta("Note: this is the ONLY env in Xi's lineup that re-emits the list per turn.")


# ============================================================================
# Interactive play (env install required)
# ============================================================================

def play(level: str, seed: int, max_turns: int):
    try:
        from agentenv_babyai.environment import BabyAI
    except ImportError as e:
        print(f"ERROR: agentenv_babyai not installed.\n  {e}\n\nSee SETUP at top of file.",
              file=sys.stderr); sys.exit(1)

    print_meta(f"env=BabyAI level={level} seed={seed} max_turns={max_turns}", "yellow")
    env = BabyAI(max_episode_steps=50, game_name=level, seed=seed)
    obs = env._get_obs()
    admissible = env._get_action_space()
    goal = env._get_goal()

    print_message("system (paper-faithful)", SYSTEM_PROMPT)
    initial_user = (
        f"Goal: {goal}\n\n{obs}\n"
        f"Available actions: [{', '.join(repr(a) for a in rewrite_list_for_llm(admissible))}]"
    )
    print_message("user — turn 0", initial_user)

    cum = 0.0
    for turn in range(max_turns):
        try:
            action = input(Color.wrap(f"\nYour action [turn {turn}/{max_turns}]> ", "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if action.lower() in ("quit", "q", "exit"): break
        if not action: continue
        action_for_env = normalize_action_for_env(action)
        if action_for_env != action:
            print_meta(f"normalized: {action!r} -> {action_for_env!r}")
        obs, reward, done, infos = env.step(action_for_env)
        cum_new = float(reward)
        delta = cum_new - cum
        cum = cum_new
        valid = bool(infos.get("action_is_valid", False))
        print_meta(f"valid={valid}  reward_delta={delta:+.3f}  cumulative={cum:.3f}  done={done}",
                   "green" if valid else "red")
        per_turn = (
            f"{obs}\nAvailable actions: ["
            f"{', '.join(repr(a) for a in rewrite_list_for_llm(env._get_action_space()))}]"
        )
        print_message(f"user — turn {turn+1}", per_turn)
        if done:
            print_meta("env done", "green" if cum >= 0.99 else "red"); break
    print_meta(f"final cumulative_reward={cum:.3f}", "bold")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("=" * 80)[0])
    p.add_argument("--show-prompt", action="store_true",
                   help="Dump the system prompt + synthetic turns (no env install needed)")
    p.add_argument("--play", action="store_true",
                   help="Interactive play (requires agentenv_babyai install)")
    p.add_argument("--level", default="BabyAI-GoToLocal-v0",
                   help="BabyAI level (default: BabyAI-GoToLocal-v0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS,
                   help=f"Override paper default ({MAX_TURNS})")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()
    if args.no_color: Color.enabled = False
    if args.show_prompt: show_prompt(); return
    if args.play: play(args.level, args.seed, args.max_turns); return
    p.print_help()


if __name__ == "__main__":
    main()
