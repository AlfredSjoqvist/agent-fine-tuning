"""WebShop harness — Xi et al. (2026) reproduction, faithful to AgentGym setup.

================================================================================
SETUP
================================================================================
WebShop requires the Princeton WebShop simulator + a ~3GB product database +
a running HTTP server. Setting this up locally is impractical for just
inspecting the harness. The recommended path is --show-prompt only.

If you really want to run it:
  - See https://github.com/princeton-nlp/WebShop for the env install
  - See https://github.com/WooooDyy/AgentGym/tree/main/agentenv-webshop for
    the AgentGym wrapper that starts a FastAPI server on top of WebShop

================================================================================
RUN
================================================================================
  # Show the system prompt and a synthetic turn 0/1 — no install needed
  python scripts/xi_webshop.py --show-prompt

  # Play mode is not supported locally — prints the SETUP guidance and exits.
  python scripts/xi_webshop.py --play

================================================================================
PAPER NOTES (Xi et al. Section 3.2, Table 1)
================================================================================
  max_turns: 10 (training and rollout). Xi uses K=20 for evaluation.
  reward:    DENSE — exact-match purchase ≈ 1.0; partial credit for
             partially-matching attributes (color, size, price band, etc.)
  list_per_step: ✗ (Xi Table 1 ACT). Clickables appear in the HTML obs
             each turn, but not as an explicit named "list of actions."
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ============================================================================
# System prompt — verbatim from AgentGym/agentenv/agentenv/envs/webshop.py
# (the "text" variant; AgentGym also has a function-call variant for
#  code-as-action prompting that we omit here for clarity)
# ============================================================================

SYSTEM_PROMPT = (
    "You are web shopping.\n"
    "I will give you instructions about what to do.\n"
    "You have to follow the instructions.\n"
    "Every round I will give you an observation and a list of available actions, "
    "you have to respond an action based on the state and instruction.\n"
    "You can use search action if search is available.\n"
    "You can click one of the buttons in clickables.\n"
    "An action should be of the following structure:\n"
    "search[keywords]\n"
    "click[value]\n"
    "If the action is not valid, perform nothing.\n"
    "Keywords in search are up to you, but the value in click must be a value in "
    "the list of available actions.\n"
    "Remember that your keywords in search should be carefully designed.\n"
    "Your response should use the following format:\n\n"
    "Thought:\n"
    "I think ... \n\n"
    "Action: \n"
    "click[something]"
)

MAX_TURNS = 10        # Xi Section 3.2
REWARD_TYPE = "dense (partial credit for matching attributes; ≈1.0 for exact match)"
LIST_PER_STEP = False # Xi Table 1 ACT = ✗ (clickables in HTML, no explicit "list" line)


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
    print(Color.wrap("  ENV: WebShop", "bold"))
    print(Color.wrap(hr("═"), "bold"))
    print_meta(f"max_turns={MAX_TURNS}  reward={REWARD_TYPE}  list_per_step={LIST_PER_STEP}", "yellow")
    print_meta("source: Yao et al. 2022a (WebShop); AgentGym wrapper")
    print_meta("action grammar: search[keywords]  |  click[clickable_value]")

    print_message("system (sent once)", SYSTEM_PROMPT)

    syn_turn0 = (
        "WebShop\n"
        "Instruction: i am looking for a pair of running shoes, mens, size 10, "
        "with extra cushioning, and price lower than 80 dollars\n"
        "[Search]"
    )
    print_message("user — turn 0 (synthetic — only the Search button is clickable)", syn_turn0)

    syn_turn1 = (
        "Instruction: i am looking for a pair of running shoes, mens, size 10, "
        "with extra cushioning, and price lower than 80 dollars\n"
        "[Back to Search]\n"
        "Page 1 (Total results: 50)\n"
        "[Next >]\n"
        "[B07X6DJ5KR]\n"
        "ASICS Men's Gel-Venture 8 Running Shoes\n"
        "$70.00 to $99.95\n"
        "[B08DLBHX7L]\n"
        "Nike Men's Pegasus 38 Running Shoe\n"
        "$120.00"
    )
    print_message("user — turn 1 (synthetic — after search; clickables in HTML)", syn_turn1)
    print_meta(
        "Note: WebShop's 'available actions' are clickable elements rendered "
        "inline in the HTML obs (e.g. [B07X6DJ5KR], [Next >], [Back to Search]). "
        "There is no explicit named 'list of valid actions' line."
    )


# ============================================================================
# Play mode (not supported locally without WebShop server infrastructure)
# ============================================================================

def play():
    print(
        "\nWebShop interactive play is not supported by this script.\n"
        "It requires:\n"
        "  - The Princeton WebShop simulator (~3GB product database)\n"
        "  - A running FastAPI server (from AgentGym/agentenv-webshop)\n"
        "\n"
        "Setup is involved; consult:\n"
        "  https://github.com/princeton-nlp/WebShop\n"
        "  https://github.com/WooooDyy/AgentGym/tree/main/agentenv-webshop\n"
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
