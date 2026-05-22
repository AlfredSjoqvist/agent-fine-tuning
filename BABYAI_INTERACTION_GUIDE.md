# Guidebook: Handwriting `babyai_interaction.py`

This guide explains every design decision in `src/babyai_rl/babyai_interaction.py` so you can reconstruct it from scratch, understand why each piece exists, and speak to it confidently.

---

## 1. What the file does

`babyai_interaction.py` is a **multi-turn rollout adapter** that connects the BabyAI text game environment (from AgentGym) to verl's GRPO training loop.

verl's `AgentLoopWorker` runs a fixed protocol:

```
start_interaction(kwargs)
  → generate_response(instance_id, messages)   # called once per LLM turn
  → generate_response(...)
  → ...
  → finalize_interaction(instance_id)
calculate_score(instance_id)                    # called after finalize
```

Your job is to implement `BaseInteraction` with those four async methods.

---

## 2. The `BaseInteraction` contract

```python
class BaseInteraction:
    def __init__(self, config: dict): ...

    async def start_interaction(self, instance_id, **kwargs) -> str:
        """Spin up env, store state, return instance_id."""

    async def generate_response(self, instance_id, messages, **kwargs
        ) -> tuple[bool, str, float, dict]:
        """Parse last assistant turn, step env, return
        (should_terminate, next_user_prompt, reward_delta, extra_info)."""

    async def calculate_score(self, instance_id, **kwargs) -> float:
        """Return final scalar reward for this rollout."""

    async def finalize_interaction(self, instance_id, **kwargs) -> None:
        """Clean up env handle, remove from instance dict."""
```

All methods are **async** because verl may run rollouts in parallel with asyncio. Keep blocking I/O (env.step) as-is — it's CPU-bound and fine in the same thread for our use case.

---

## 3. State management: `_instance_dict`

verl runs many rollouts in parallel. Each rollout has its own env instance, identified by `instance_id` (a UUID string). You need a class-level dict to hold per-rollout state:

```python
self._instance_dict: dict[str, dict[str, Any]] = {}
```

The state dict for each instance holds:
| Key | What it tracks |
|-----|---------------|
| `env` | The live `BabyAI` gym object |
| `obs` | Current text observation |
| `admissible` | Current list of valid actions (changes every step) |
| `goal` | The task goal string (fixed for lifetime of episode) |
| `turn` | How many assistant turns have elapsed |
| `done` | Whether the env signaled episode end |
| `task_success` | True only if `done=True` AND `float(reward) > 0` |
| `task_reward` | The shaped reward emitted on success (0.0 until then) |
| `task` / `game_name` / `seed` | Provenance for logging |

---

## 4. `start_interaction`: spinning up the env

```python
async def start_interaction(self, instance_id=None, game_name=None, seed=0, task="", **kwargs):
```

Key steps:
1. Generate `instance_id = str(uuid4())` if one wasn't provided.
2. Instantiate `BabyAI(max_episode_steps=..., game_name=game, seed=seed)` — BabyAI calls `.reset()` internally so the env is ready immediately.
3. Read initial state: `env._get_action_space()`, `env._get_obs()`, `env._get_goal()`.
4. Store everything in `self._instance_dict[instance_id]`.
5. Return `instance_id`.

`game_name` and `seed` come from `interaction_kwargs` in the parquet row, so each training example maps deterministically to one room layout.

---

## 5. `generate_response`: the core turn loop

```python
async def generate_response(self, instance_id, messages, **kwargs):
```

### 5a. Extract the last assistant turn

verl passes the full message history. Walk it **in reverse** to find the last `role == "assistant"` entry:

```python
assistant_content = ""
for item in reversed(messages):
    if item.get("role") == "assistant":
        assistant_content = item.get("content", "") or ""
        break
```

### 5b. Parse the action

Call `_parse_action(assistant_content, state["admissible"])`. Returns `(action, parse_path)`. `parse_path` is a string tag for logging so you can see post-hoc how often the model used ReAct format vs. fell back to heuristics.

### 5c. Normalize for the env

AgentGym's BabyAI uses `"pickup X"` but the model (and the admissible list shown to it) uses `"pick up X"`. Call `_normalize_action_for_env(action)` to rewrite before `env.step()`.

### 5d. Step the env

```python
obs, reward, done, infos = state["env"].step(action_for_env)
```

### 5e. Handle `block_check_available`

If the config flag `block_check_available=True` and the env's response starts with `"You can take the following actions"`, the model tried to use the `"check available actions"` backdoor. Return a penalty prompt without advancing the game state meaningfully.

### 5f. Update state and decide termination

```python
should_terminate = done_this_turn or state["turn"] >= self.max_turns
```

Detect success with `float(reward) > 0` — BabyAI returns 0.0 every step and 1.0 only on task completion, so no cumulative tracking is needed. Compute the shaped reward only on success:

```python
success_this_turn = done_this_turn and float(reward) > 0
if success_this_turn:
    step_count = state["turn"] + 1  # not yet incremented at this point
    reward_this_turn = 1.0 - 0.9 * step_count / self.max_episode_steps
else:
    reward_this_turn = 0.0
```

Update `turn`, `done`, and on success set `task_success = True` and `task_reward = reward_this_turn`.

### 5g. Build `next_prompt`

Call `_format_observation(obs, admissible, turn, max_turns, show_action_list, shuffle_list)`.

### 5h. Return

```python
return should_terminate, next_prompt, reward_delta, extra_info_dict
```

`extra_info_dict` must include at minimum `"turn"`, `"cumulative_reward"`, `"env_done"`, `"task_success"`.

---

## 6. `calculate_score`

```python
async def calculate_score(self, instance_id, **kwargs) -> float:
    state = self._instance_dict.get(instance_id)
    if not state:
        return 0.0
    return float(state.get("task_reward", 0.0))
```

This method is required by `BaseInteraction` but is **not the active reward path** in this project. `modal_train_babyai.py` registers `reward.py` as the custom reward function, so verl calls `compute_score` there instead. Implement `calculate_score` as a stub that reads `task_reward` from state — it produces the same number as `compute_score` and satisfies the interface.

---

## 7. `finalize_interaction`

```python
async def finalize_interaction(self, instance_id, **kwargs) -> None:
    state = self._instance_dict.get(instance_id, {})
    # Log final state...
    env = state.get("env")
    if env is not None:
        try:
            env.env.close()   # BabyAI wraps minigrid; close the inner env
        except Exception:
            pass
    self._instance_dict.pop(instance_id, None)
```

Always guard with `.get()` — verl may call `finalize` even if `start` raised an exception.

---

## 8. `_parse_action`: action extraction

The model is prompted to respond in ReAct format:

```
Thought:
I see a red ball to my left.

Action:
go to red ball 1
```

The parser handles three cases in priority order:

1. **Same-line prefix**: `"Action: go to red ball 1"` — match with regex `r"action\s*:\s*(.+)$"` scanning lines in reverse.
2. **Multi-line prefix**: `"Action:\n"` followed by the action on the next line.
3. **Admissible-list fallback**: if no `Action:` prefix, check if any line matches an admissible action (case-insensitive). This rescues the model when it omits the prefix.
4. **Last-line fallback**: take the last non-empty line as a last resort.

**Why reverse?** Multi-step reasoning often has the action at the end. Scanning in reverse finds it without needing to know exactly where it appears.

---

## 9. Surface-form rewrites: the `pickup`/`pick up` mismatch

AgentGym's BabyAI internally uses `"pickup X"` (no space). Natural English and the model's prior say `"pick up X"`. Two symmetric functions handle this:

- `_rewrite_list_for_llm(admissible)`: applied before showing the admissible list to the model. `"pickup X"` → `"pick up X"`.
- `_normalize_action_for_env(action)`: applied before `env.step()`. `"pick up X"` → `"pickup X"`.

This lets the model output natural English and still have its action accepted by the env.

---

## 10. `_format_observation`: per-turn user messages

```python
def _format_observation(obs, admissible, turn, max_turns, show_action_list, shuffle_list):
```

- `show_action_list=True` (L_per_step, L_shuffled, L_init): appends `\nAvailable actions: [...]` after the observation text. Admissible list is rewritten for the LLM.
- `show_action_list=False` (L_none, L_examples): returns `obs` unchanged.
- `shuffle_list=True` (L_shuffled condition): permutes the admissible list order randomly each turn. Preserves information content but eliminates positional/string-rote learning.

---

## 11. Trajectory dumping (`_dump_event`)

```python
_DUMP_DIR = os.environ.get("BABYAI_TRAJECTORY_DUMP_DIR", "")
```

If this env var is set, every `start`, `turn`, and `finalize` event is appended as a JSON line to `{DUMP_DIR}/{run_id}/{instance_id}.jsonl`. This is **purely diagnostic** — training works without it. Always wrap in `try/except` so a disk error never kills a rollout.

---

## 12. Config flags (set in YAML, passed to `__init__`)

| Config key | Type | Effect |
|---|---|---|
| `max_turns` | int | Max assistant turns before forced termination |
| `max_episode_steps` | int | Passed to BabyAI env (minigrid step budget) |
| `default_game_name` | str | Fallback level if parquet row has no `game_name` |
| `show_action_list_per_turn` | bool | Whether to append admissible list each turn |
| `shuffle_list_per_turn` | bool | Whether to shuffle that list randomly per turn |
| `block_check_available` | bool | Whether to intercept `"check available actions"` |

---

## 13. Conditions and what changes where

| Condition | System prompt | Initial user msg | Per-turn msg | Interaction flag |
|---|---|---|---|---|
| L_per_step | base + format | obs + list | obs + list | `show_action_list=True` |
| L_shuffled | base + format | obs + list | obs + shuffled list | `show_action_list=True, shuffle=True` |
| L_init | base + format | obs + list (no "check") | obs only | `show_action_list=False, block=True` |
| L_none / L_examples | base/examples + format | obs only | obs only | `show_action_list=False` |

The **data** (parquet) controls the system prompt and turn-0 user message. The **interaction** (this file + YAML config) controls what happens on turns 1+.

---

## 14. Common mistakes to avoid

- **Using running-max delta**: BabyAI returns 0.0 every step and 1.0 on completion. `float(reward) > 0` is all you need to detect success — no cumulative tracking required.
- **Wrong step count**: `state["turn"]` is incremented *after* the reward block. Use `state["turn"] + 1` as the step count when computing the shaped reward.
- **Not closing inner env**: `BabyAI` wraps minigrid. Call `env.env.close()` not `env.close()` in `finalize_interaction`.
- **Mutable default admissible set**: Always recompute `env._get_action_space()` after each step — the admissible list changes as the room state changes.
- **Overwriting `_instance_dict` entries**: Multiple concurrent rollouts share the same `BabyAIInteraction` object. Never use class-level mutable state for per-rollout data — always key by `instance_id`.
- **Not guarding `finalize`**: `self._instance_dict.get(instance_id, {})` — verl may call finalize on a failed start.

---

## 15. Reading checklist before writing

Before you write any method, re-read:

1. `BaseInteraction` — what signatures verl expects, what types each argument is.
2. `agentenv_babyai.environment.BabyAI` — what `_get_action_space()`, `_get_obs()`, `_get_goal()`, and `step()` return.
3. `prepare_babyai_data.py` — what the parquet rows look like, specifically `interaction_kwargs` (which becomes `**kwargs` in `start_interaction`).
4. `reward.py` — what `compute_score` consumes from `extra_info["turn_scores"]` (verl fills this from your per-turn `reward_delta`).