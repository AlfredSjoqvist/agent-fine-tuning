"""TextCraft harness — Xi et al. (2026) reproduction, faithful to AgentGym setup.

================================================================================
SETUP (one-time, ~5 min)
================================================================================
  git clone https://github.com/WooooDyy/AgentGym
  cd AgentGym/agentenv-textcraft
  pip install --no-deps -e .
  cd ../..

================================================================================
RUN
================================================================================
  # Show the system prompt and a synthetic turn 0/1 — no install needed
  python scripts/xi_textcraft.py --show-prompt

  # Interactive play (requires agentenv_textcraft install)
  python scripts/xi_textcraft.py --play
  python scripts/xi_textcraft.py --play --idx 3 --seed 7

================================================================================
PAPER NOTES (Xi et al. Section 3.2, Table 1)
================================================================================
  max_turns: 15 (training and rollout). Xi uses K=20 for evaluation.
  reward:    SPARSE / binary — 1.0 on target item crafted, 0 otherwise
  list_per_step: ✗ (Xi Table 1 ACT). Recipes shown ONLY at reset.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ============================================================================
# System prompt — verbatim from AgentGym/agentenv/agentenv/envs/textcraft.py
# ============================================================================

SYSTEM_PROMPT = (
    "You are given few useful crafting recipes to craft items in Minecraft. Crafting "
    "commands are of the format \"craft [target object] using [input ingredients]\".\n"
    "Every round I will give you an observation, you have to respond an action based "
    "on the state and instruction. You can \"get\" an object (ingredients) from the "
    "inventory or the environment, look-up the game inventory by \"inventory\", or "
    "\"craft\" (target) using any of the crafting commands.\n"
    "Your output must strictly follow this format:\"Thought:\nyour thoughts.\n\n"
    "Action:\nyour next action\"\n\n"
    "Reminder: \n"
    "1. Always specify the quantity when using \"get\" and \"craft\" commands. "
    "- Example of get: get 1 lapis lazuli "
    "- Example1 of craft: craft 1 blue dye using 1 lapis lazuli "
    "- Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot\n"
    "2. When using \"get\" command, do not specify whether the item comes from the "
    "inventory or the environment.\n"
    "3. You can use ONLY crafting commands provided, do not use your own crafting "
    "commands. However, if the crafting command uses a generic ingredient like "
    "\"planks\", you can use special types of the same ingredient e.g. \"dark oak "
    "planks\" in the command instead.\n"
)

MAX_TURNS = 15        # Xi Section 3.2
REWARD_TYPE = "sparse / binary (1.0 = target crafted, 0 = otherwise)"
LIST_PER_STEP = False # Xi Table 1 ACT = ✗


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
    print(Color.wrap("  ENV: TextCraft", "bold"))
    print(Color.wrap(hr("═"), "bold"))
    print_meta(f"max_turns={MAX_TURNS}  reward={REWARD_TYPE}  list_per_step={LIST_PER_STEP}", "yellow")
    print_meta("source: Sanghi et al. 2022 (TextCraft); AgentGym wrapper")
    print_meta(
        "action grammar: get N <item> | craft N <target> using N <ingredient>, ... | inventory"
    )

    print_message("system (sent once)", SYSTEM_PROMPT)

    syn_turn0 = (
        "Crafting commands:\n"
        "craft 1 stick using 2 planks\n"
        "craft 4 planks using 1 oak log\n"
        "craft 1 wooden pickaxe using 3 planks, 2 stick\n"
        "\n"
        "Goal: craft 1 wooden pickaxe."
    )
    print_message("user — turn 0 (synthetic — recipes shown here, never again)", syn_turn0)

    syn_turn1 = "Got 1 oak log"
    print_message("user — turn 1 (synthetic — recipes NOT re-emitted)", syn_turn1)
    print_meta(
        "Note: TextCraft only emits the recipe list at reset. Subsequent turns "
        "show just the textual outcome of the previous action."
    )


# ============================================================================
# Interactive play (requires agentenv_textcraft install)
# ============================================================================

def play(idx: int, seed: int, max_turns: int):
    try:
        from agentenv_textcraft.environment import TextCraft
    except ImportError as e:
        print(f"ERROR: agentenv_textcraft not installed.\n  {e}\n\nSee SETUP at top of file.",
              file=sys.stderr); sys.exit(1)

    print_meta(f"env=TextCraft idx={idx} seed={seed} max_turns={max_turns}", "yellow")
    env = TextCraft()
    initial_obs, _info = env.reset(seed=seed, data_idx=idx)

    print_message("system (paper-faithful)", SYSTEM_PROMPT)
    print_message("user — turn 0", initial_obs)

    cum = 0.0
    for turn in range(max_turns):
        try:
            action = input(Color.wrap(f"\nYour action [turn {turn}/{max_turns}]> ", "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if action.lower() in ("quit", "q", "exit"): break
        if not action: continue
        obs, reward, terminated, truncated, info = env.step(action)
        cum = max(cum, float(reward))
        print_meta(
            f"reward={float(reward):.3f}  cumulative={cum:.3f}  "
            f"terminated={terminated}  truncated={truncated}",
            "green" if reward > 0 else "red",
        )
        # Paper-faithful: textcraft observe() is just the obs string, no recipe re-emit
        print_message(f"user — turn {turn+1}", obs)
        if terminated or truncated:
            print_meta("env done", "green" if cum >= 0.99 else "red"); break
    print_meta(f"final cumulative_reward={cum:.3f}", "bold")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("=" * 80)[0])
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--play", action="store_true")
    p.add_argument("--idx", type=int, default=0, help="Task index (selects target item)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS,
                   help=f"Override paper default ({MAX_TURNS})")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()
    if args.no_color: Color.enabled = False
    if args.show_prompt: show_prompt(); return
    if args.play: play(args.idx, args.seed, args.max_turns); return
    p.print_help()


if __name__ == "__main__":
    main()
