"""SearchQA harness — Xi et al. (2026) reproduction, faithful to AgentGym setup.

================================================================================
SETUP
================================================================================
SearchQA requires a retrieval index (Wikipedia / similar) for the env's
<search> queries to return real <information>. Setting this up locally is
involved. The recommended path is --show-prompt only.

If you really want to run it:
  - See https://github.com/WooooDyy/AgentGym/tree/main/agentenv-searchqa
    for the AgentGym wrapper

================================================================================
RUN
================================================================================
  # Show the system prompt and a synthetic turn 0/1 — no install needed
  python scripts/xi_searchqa.py --show-prompt

  # Play mode is not supported locally — prints the SETUP guidance and exits.
  python scripts/xi_searchqa.py --play

================================================================================
PAPER NOTES (Xi et al. Section 3.2, Table 1)
================================================================================
  max_turns: 5 (training and rollout). Xi uses K=20 for evaluation.
  reward:    SPARSE / binary — 1.0 if the final <answer> exact-matches
             the gold answer (with format-correctness check)
  list_per_step: ✗ (Xi Table 1 ACT). No action list — free-form
             <search>/<answer> tags.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ============================================================================
# System prompt — verbatim from AgentGym/agentenv/agentenv/envs/searchqa.py
# ============================================================================

SYSTEM_PROMPT = (
    "You must always reason inside <think>...</think> first; if you lack knowledge, "
    "issue a <search>...</search> and then stop; do not generate <information> or "
    "<answer> yet; wait for external input between <information>...</information> "
    "before continuing; resume only when new <information> is given; do not skip "
    "steps or anticipate answers early."
)

MAX_TURNS = 5         # Xi Section 3.2
REWARD_TYPE = "sparse / binary (1.0 if final <answer> exact-matches; format check)"
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
# Show-prompt mode
# ============================================================================

def show_prompt():
    print(Color.wrap(hr("═"), "bold"))
    print(Color.wrap("  ENV: SearchQA", "bold"))
    print(Color.wrap(hr("═"), "bold"))
    print_meta(f"max_turns={MAX_TURNS}  reward={REWARD_TYPE}  list_per_step={LIST_PER_STEP}", "yellow")
    print_meta("source: Dunn et al. 2017 (SearchQA); AgentGym wrapper")
    print_meta(
        "action grammar: <search>query</search>  |  <answer>final_answer</answer>"
    )

    print_message("system (sent once)", SYSTEM_PROMPT)

    syn_turn0 = "Question: Who was the first person to walk on the moon?"
    print_message("user — turn 0 (synthetic — just the question)", syn_turn0)

    syn_turn1 = (
        "<information>Neil Armstrong (1930-2012) was an American astronaut and the "
        "first person to walk on the Moon. He commanded the Apollo 11 mission on "
        "July 20, 1969.</information>"
    )
    print_message("user — turn 1 (synthetic — after assistant issues <search>)", syn_turn1)
    print_meta(
        "Note: no action list at any turn. The LLM uses <think>/<search>/<answer> "
        "tags to drive the env. Most constrained / least scaffolded of Xi's 5 envs."
    )


# ============================================================================
# Play mode (not supported locally without retrieval index)
# ============================================================================

def play():
    print(
        "\nSearchQA interactive play is not supported by this script.\n"
        "It requires:\n"
        "  - A retrieval index (Wikipedia / similar) for <search> to return results\n"
        "  - A running FastAPI server (from AgentGym/agentenv-searchqa)\n"
        "\n"
        "Consult: https://github.com/WooooDyy/AgentGym/tree/main/agentenv-searchqa\n"
        "\n"
        "Use --show-prompt to inspect the harness without installing.\n"
    )
    sys.exit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("=" * 80)[0])
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--play", action="store_true",
                   help="(Not supported locally — prints setup guidance and exits.)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()
    if args.no_color: Color.enabled = False
    if args.show_prompt: show_prompt(); return
    if args.play: play(); return
    p.print_help()


if __name__ == "__main__":
    main()
