"""Interactive BabyAI player for testing the four harness designs.

Plays a BabyAI episode in your terminal showing you THE EXACT MESSAGES the LLM
would see under each of the four conditions (L_per_step, L_init, L_examples,
L_none). You type the action each turn. The env steps, the next user message
is shown verbatim as the LLM would receive it.

------------------------------------------------------------------------------
SETUP (one-time)
------------------------------------------------------------------------------
    pip install gym "gymnasium<1.0.0" "minigrid<2.5.0" matplotlib fastapi uvicorn
    git clone https://github.com/WooooDyy/AgentGym
    cd AgentGym/agentenv-babyai
    pip install --no-deps -e .
    cd ../..

(fastapi+uvicorn are needed because agentenv_babyai/__init__.py imports them
even though we don't run the server. Trailing dot in `pip install -e .` is
required — it's the path argument.)

------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------
    # Play under L_per_step (Xi's BabyAI default — list every turn)
    python scripts/play_babyai.py --condition per_step

    # Play under L_init (list at turn 0, observation only after)
    python scripts/play_babyai.py --condition init

    # Play under L_examples (system prompt has format examples; no per-instance list)
    python scripts/play_babyai.py --condition examples

    # Play under L_none (no list, no examples — schema only)
    python scripts/play_babyai.py --condition none

    # Other useful flags
    python scripts/play_babyai.py --condition examples --level BabyAI-Pickup-v0 --seed 7
    python scripts/play_babyai.py --condition examples --no-color

    # Compare the prompts of all four conditions WITHOUT playing
    python scripts/play_babyai.py --show-prompts

------------------------------------------------------------------------------
DURING PLAY
------------------------------------------------------------------------------
At each turn, the script prints the user message exactly as the LLM would see
it, then asks you to type an action. Special inputs:

    help    show the underlying admissible-action list (cheats — useful for
            seeing what the env considers valid right now)
    quit    exit early
    <enter> repeat prompt
"""

import argparse
import os
import re
import sys
import warnings

# Quiet noisy imports
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def _import_env():
    """Lazy import — --show-prompts mode doesn't need the env installed."""
    try:
        from agentenv_babyai.environment import BabyAI
        return BabyAI
    except ImportError as e:
        print(
            "ERROR: agentenv_babyai is not importable.\n"
            "Setup steps:\n"
            '  pip install gym "gymnasium<1.0.0" "minigrid<2.5.0" matplotlib fastapi uvicorn\n'
            "  git clone https://github.com/WooooDyy/AgentGym\n"
            "  cd AgentGym/agentenv-babyai\n"
            "  pip install --no-deps -e .\n"
            "  cd ../..\n\n"
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# System prompt building blocks
# ---------------------------------------------------------------------------

INTRO = (
    "You are an exploration master that wants to finish every goal you are "
    "given. Every round I will give you an observation, and you have to "
    "respond an action and your thought based on the observation to finish "
    "the given task. You are placed in a room and you need to accomplish the "
    "given goal with actions.\n\n"
)

SCHEMA_BLOCK = (
    "You can use the following actions (the schema):\n"
    "- turn left\n"
    "- turn right\n"
    "- move forward\n"
    "- go to <obj> <id>\n"
    "- pick up <obj> <id>\n"
    "- drop\n"
    "- go through <door> <id>: <door> must be an open door.\n"
    "- toggle and go through <door> <id>: <door> can be a closed door or a locked door. "
    "If the door is locked, you need to carry a key of the same color.\n"
    "- toggle (e.g., to open a box that is right in front of you)\n"
    "- check available actions\n\n"
)

# L_init omits "check available actions" from the schema so the LLM never
# learns the backdoor exists (the interaction also blocks it at runtime).
SCHEMA_BLOCK_NO_CHECK = SCHEMA_BLOCK.replace("- check available actions\n", "")

EXAMPLES_BLOCK = (
    "Examples of valid action surface forms (the placeholders <obj> <id> get "
    "filled in based on what is in your room — your specific room will have "
    "DIFFERENT objects, colors, and IDs than these examples):\n"
    "- 'turn left'\n"
    "- 'turn right'\n"
    "- 'move forward'\n"
    "- 'go to red ball 1'\n"
    "- 'go to grey closed door 1'\n"
    "- 'go to green box 2'\n"
    "- 'pick up yellow key 1'\n"
    "- 'pick up purple ball 1'\n"
    "- 'drop'\n"
    "- 'toggle and go through red closed door 1'\n"
    "- 'go through red open door 1'\n"
    "- 'toggle'\n"
    "- 'check available actions'\n\n"
    "Object IDs use the form '<color> <object> <integer>', e.g., 'yellow key 1'. "
    "Doors include their state in the name, e.g., 'grey closed door 1' vs "
    "'grey open door 1'.\n\n"
    "Each turn, you will receive a description of what you can see. You must "
    "construct a valid action by:\n"
    "  1. Recognizing what action types exist from the schema above.\n"
    "  2. Reading the observation to find what objects are in your room (with "
    "their colors and IDs, e.g., \"a green ball 1 1 steps in front of you\").\n"
    "  3. Filling in the schema with the specific objects from your observation.\n\n"
    "If the environment responds with \"Nothing happens.\" or \"The action is "
    "not recognized.\", that means your action was invalid (object doesn't "
    "exist, action doesn't apply, or path is blocked). Try a different action.\n\n"
)

CONTRACTS = {
    "per_step": (
        "For each of your turns, you will be given a list of actions which "
        "you can choose one to perform in this turn. The list is updated each "
        "turn to reflect what is currently valid based on what you can see "
        "and where you are.\n\n"
    ),
    "init": (
        "At the start of the task, you will be given a list of "
        "currently-available actions. For all subsequent turns, you will see "
        "only the observation — you must derive valid actions from the schema "
        "above and the objects you observe in the room.\n\n"
    ),
    "examples": (
        "Each turn, you will see only an observation describing the room. "
        "Use the schema and the format examples above plus what you observe "
        "to choose a valid action.\n\n"
    ),
    "none": (
        "Each turn, you will see only an observation describing the room. "
        "Use the schema above and what you observe to choose a valid action. "
        "If the environment responds with 'Nothing happens.', your action "
        "was invalid — try a different one.\n\n"
    ),
}

FORMAT_BLOCK = (
    "Your response must follow this format:\n"
    "Thought:\n"
    "your thoughts.\n"
    "\n"
    "Action:\n"
    "your next action"
)


def build_system_prompt(condition: str) -> str:
    schema = SCHEMA_BLOCK_NO_CHECK if condition == "init" else SCHEMA_BLOCK
    parts = [INTRO, schema]
    if condition == "examples":
        parts.append(EXAMPLES_BLOCK)
    parts.append(CONTRACTS[condition])
    parts.append(FORMAT_BLOCK)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Action surface-form normalization (boundary translation)
# ---------------------------------------------------------------------------
#
# AgentGym's BabyAI env uses 'pickup <obj> <id>' (one word). The natural
# English form an LLM produces is 'pick up <obj> <id>' (two words). We let
# the LLM live in natural English and translate at the env boundary:
#   - Outgoing (env list -> LLM):  'pickup X' -> 'pick up X'
#   - Incoming (LLM action -> env): 'pick up X' -> 'pickup X'
#
# This is symmetric and applied to BOTH directions, so both per_step and
# none conditions get the same normalization (no bias).

def _rewrite_list_for_llm(admissible: list[str]) -> list[str]:
    """Make AgentGym's quirky surface forms look like natural English to the LLM."""
    out = []
    for a in admissible:
        a = re.sub(r"^pickup\s+", "pick up ", a)
        out.append(a)
    return out


def _normalize_action_for_env(action: str) -> str:
    """Translate the LLM's natural English action into AgentGym's expected form."""
    a = action.strip()
    a = re.sub(r"^pick\s+up\s+", "pickup ", a, flags=re.IGNORECASE)
    a = re.sub(r"^pick-up\s+", "pickup ", a, flags=re.IGNORECASE)
    return a


# ---------------------------------------------------------------------------
# Per-turn user message formatters
# ---------------------------------------------------------------------------

def _format_obs_with_list(obs: str, admissible: list[str]) -> str:
    rewritten = _rewrite_list_for_llm(admissible)
    return f"{obs}\nAvailable actions: [{', '.join(repr(a) for a in rewritten)}]"


def make_initial_user(
    obs: str, admissible: list[str], condition: str, goal: str = ""
) -> str:
    """Turn-0 user message. The goal is prepended so the LLM knows the task.
    L_per_step and L_init include the action list; L_examples and L_none don't.

    NOTE: our existing data-prep code does NOT yet include the goal in the
    parquet's initial user message. That's a real bug to fix before the full
    experiment — the LLM would otherwise be guessing what task to solve.
    """
    goal_line = f"Goal: {goal}\n\n" if goal else ""
    if condition == "per_step":
        return goal_line + _format_obs_with_list(obs, admissible)
    if condition == "init":
        filtered = [a for a in admissible if a.lower() != "check available actions"]
        return goal_line + _format_obs_with_list(obs, filtered)
    return goal_line + obs


def make_per_turn_user(obs: str, admissible: list[str], condition: str) -> str:
    """Turn 1+ user message. Only L_per_step re-emits the list. Goal is not
    repeated each turn (it's in conversation history from turn 0)."""
    if condition == "per_step":
        return _format_obs_with_list(obs, admissible)
    return obs


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

class Color:
    enabled = True

    @classmethod
    def wrap(cls, text: str, code: str) -> str:
        if not cls.enabled:
            return text
        codes = {
            "red": "31", "green": "32", "yellow": "33", "blue": "34",
            "magenta": "35", "cyan": "36", "gray": "90", "bold": "1",
        }
        return f"\033[{codes[code]}m{text}\033[0m"


def hr(char: str = "─", n: int = 78) -> str:
    return char * n


def print_message(role: str, content: str):
    label = role.upper()
    print()
    print(Color.wrap(f"┌─ {label} {hr('─', 76 - len(label))}", "cyan"))
    for line in content.split("\n"):
        print(Color.wrap("│ ", "cyan") + line)
    print(Color.wrap(f"└{hr('─', 78)}", "cyan"))


def print_meta(text: str, color: str = "gray"):
    print(Color.wrap(f"[{text}]", color))


# ---------------------------------------------------------------------------
# Show-prompts mode (no env, just dump the four conditions)
# ---------------------------------------------------------------------------

def show_prompts():
    """Dump system prompts and a synthetic turn-0 / turn-1 message for all
    four conditions, side-by-side, so you can compare without playing."""
    fake_obs = (
        "In front of you in this room, you can see several objects: There is "
        "a green ball 1 1 steps in front of you and 2 steps to your left. "
        "There is a grey closed door 1 3 steps in front of you. The room has "
        "walls around you. You are facing a wall 4 steps away."
    )
    fake_admissible = [
        "turn left", "turn right", "move forward",
        "toggle and go through grey closed door 1",
        "go to green ball 1", "go to grey closed door 1",
        "check available actions",
    ]
    fake_admissible_init = [a for a in fake_admissible if a != "check available actions"]
    fake_goal = "go to the green ball"
    for cond in ("per_step", "init", "examples", "none"):
        print()
        print(Color.wrap(hr("═"), "bold"))
        print(Color.wrap(f"  CONDITION: L_{cond}", "bold"))
        print(Color.wrap(hr("═"), "bold"))
        print_message(f"system (sent once)", build_system_prompt(cond))
        admissible_for_cond = fake_admissible_init if cond == "init" else fake_admissible
        print_message(
            f"user — turn 0 (initial)",
            make_initial_user(fake_obs, admissible_for_cond, cond, goal=fake_goal),
        )
        print_message(
            f"user — turn 1+ (after a step)",
            make_per_turn_user(fake_obs, fake_admissible, cond),
        )


# ---------------------------------------------------------------------------
# Interactive game loop
# ---------------------------------------------------------------------------

def play(condition: str, level: str, seed: int, max_turns: int):
    print_meta(f"condition=L_{condition}  level={level}  seed={seed}", "yellow")
    print_meta("Loading BabyAI env...")
    BabyAI = _import_env()
    env = BabyAI(max_episode_steps=50, game_name=level, seed=seed)
    initial_obs = env._get_obs()
    initial_actions = env._get_action_space()
    goal = env._get_goal()

    print_meta(f"GOAL (from underlying env): {goal!r}")
    print_meta(
        f"Underlying admissible at turn 0: {len(initial_actions)} actions",
    )

    # System prompt — sent once at the top of the conversation.
    print_message("system (this is what the LLM sees once)", build_system_prompt(condition))

    # Turn 0 user message — includes the goal so the LLM knows the task.
    initial_user = make_initial_user(initial_obs, initial_actions, condition, goal=goal)
    print_message("user — turn 0", initial_user)

    cumulative_reward = 0.0
    turn = 0
    done = False

    while not done and turn < max_turns:
        # Solicit action from the human player.
        prompt = (
            f"\nYour action [turn {turn}/{max_turns}]  "
            f"(type 'help' to peek at the underlying admissible list, 'quit' to exit)\n> "
        )
        try:
            action = input(Color.wrap(prompt, "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_meta("interrupted")
            break

        if action.lower() in ("quit", "exit", "q"):
            print_meta("quit")
            break
        if action.lower() == "help":
            adm = env._get_action_space()
            print_meta(
                f"Underlying admissible actions ({len(adm)}, env-native spelling; "
                f"the LLM sees these rewritten via _rewrite_list_for_llm):"
            )
            for a in adm:
                print(f"  - {a!r}")
            continue
        if not action:
            continue

        # Translate the human/LLM's natural English action into AgentGym's
        # expected surface form before passing to env.step.
        action_for_env = _normalize_action_for_env(action)
        if action_for_env != action:
            print_meta(f"normalized: {action!r} -> {action_for_env!r}")

        # Step the env.
        obs, reward, env_done, infos = env.step(action_for_env)
        reward_delta = float(reward) - cumulative_reward
        cumulative_reward = float(reward)
        was_valid = bool(infos.get("action_is_valid", False))
        turn += 1
        done = bool(env_done)

        # Block the list-recovery backdoor for init condition by intercepting
        # the env's response. The env accepts many fuzzy variants of "check
        # available actions" so we match on the response pattern instead of
        # the input string.
        if condition == "init" and obs.startswith("You can take the following actions"):
            print_meta("blocked: action list recovery is not available in L_init", "red")
            continue

        new_admissible = env._get_action_space()
        per_turn_user = make_per_turn_user(obs, new_admissible, condition)

        meta = (
            f"action_was_valid={was_valid}  "
            f"reward_delta={reward_delta:+.3f}  "
            f"cumulative_reward={cumulative_reward:.3f}  "
            f"env_done={done}"
        )
        print_meta(meta, "green" if was_valid else "red")

        # Per-turn user message — exactly what the LLM sees next.
        print_message(f"user — turn {turn}", per_turn_user)

        if done:
            print_meta("ENVIRONMENT REPORTS DONE")
            if cumulative_reward >= 0.99:
                print(Color.wrap("✓ Task complete!", "green"))
            break

    if turn >= max_turns and not done:
        print_meta(f"Hit max_turns={max_turns} without completing", "red")

    print()
    print(Color.wrap(hr("═"), "bold"))
    print(Color.wrap(f"  FINAL  cumulative_reward={cumulative_reward:.3f}  turns={turn}  done={done}", "bold"))
    print(Color.wrap(hr("═"), "bold"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("------")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--condition", choices=["per_step", "init", "examples", "none"],
        default="per_step",
        help="Which harness to play under (default: per_step)",
    )
    p.add_argument(
        "--level", default="BabyAI-GoToLocal-v0",
        help="BabyAI level to play (default: GoToLocal — multi-object room "
             "with rich turn 0). Other examples: BabyAI-GoToRedBall-v0, "
             "BabyAI-Pickup-v0, BabyAI-Open-v0",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (determines room layout)",
    )
    p.add_argument(
        "--max-turns", type=int, default=30,
        help="Episode turn cap",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color codes (for terminals that don't render them)",
    )
    p.add_argument(
        "--show-prompts", action="store_true",
        help="Dump system prompts and synthetic user messages for all four "
             "conditions, then exit. Doesn't load any env.",
    )
    args = p.parse_args()

    if args.no_color:
        Color.enabled = False

    if args.show_prompts:
        show_prompts()
        return

    play(args.condition, args.level, args.seed, args.max_turns)


if __name__ == "__main__":
    main()
