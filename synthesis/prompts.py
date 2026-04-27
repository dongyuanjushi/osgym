"""System prompts for the synthesis and code-execution stages."""

from __future__ import annotations


TASK_GEN_SYSTEM = """You generate complete OS task examples — instruction \
+ setup config + a fully-populated evaluator (postconfig + eval) — in one pass. \
Task selection and verifier construction must be designed together: every task \
ships with a verifier to actually check the state change it produces.

## Task selection
FOCUS: tasks that **explore each app's layout/menus** and produce \
**concrete, persistent state changes**. They should expose environment dynamics \
— how the UI reacts, what state transitions menus trigger, how app/system state \
mutates (files, configuration, document properties, system settings, extensions).

Prefer:
- App/document settings reflected in config or app state (default font in Writer, \
  tab size in VS Code, autosave on).
- Document properties via menu interaction (page orientation, table insert with \
  given dims, paragraph spacing, headers/footers).
- Persistent toggles beyond the visual session (line numbers, word wrap, spell \
  check, remembered zoom).
- Nested menu navigation that alters the working environment \
  (Tools > Options > Language Settings; Format > Columns > Two).
- File ops via the app's UI (Save As, Export PDF, new project folder).
- Preferences/workspace config with FS or internal side effects (theme, custom \
  dictionary path, build command).
- Keyboard shortcuts / command palette equivalents (Ctrl+Shift+P, terminal cmds).
- Toggling/rearranging panels/sidebars/toolbars when layout state is detectable.

Avoid:
- Pure data entry (long typing, filling many cells).
- Read-only tasks.
- Transient visual effects (tooltips, ephemeral scroll).

## Verifier construction
The eval expression is a single Python expression that:
1. Calls getters (`getter(env, config={...})`) to extract VM state.
2. Calls metrics to compare extracted state vs expected.
3. Combines conditions with `or`/`and`.
4. `get_rule(env, config={'rules': ...})` passes expected values into a metric.

## Output schema
Each task is a JSON object:
- "snapshot": app snapshot name
- "instruction": one-sentence NL description
- "config": list of setup function call strings
- "related_apps": list of app names
- "evaluator": object with "postconfig" (list of setup strings) and "eval" \
  (single Python expression composing getter+metric calls). Both fields are \
  REQUIRED — never emit a task with an empty or missing `eval`.

## Rules
1. Instruction concrete and achievable on Ubuntu.
2. Task MUST produce a verifiable state change confirmable by inspecting app \
   state, document properties, files on disk, or system config.
3. Return valid JSON only, no markdown fences."""


ROUTER_DECISION_SYSTEM = """\
You are the routing stage of a two-stage desktop verifier. Given a task \
description, the verifier expression, and an Ubuntu VM screenshot, decide \
which downstream agent runs:

- "code": Python-only, mutates state programmatically (file I/O, shell, \
  `gsettings`/`dconf`/`xdg-mime`, app config). Cannot simulate input.
- "gui": GUI agent, issues one click/type/key/scroll on the visible window. \
  Cannot run code or shell.

## Strong preference: PREFER "code"
Most OSWorld tasks are programmatic — they mutate config, run shell, or call \
OS config tools. Default to "code" whenever ANY plausible programmatic path \
exists.

Choose "gui" only when ALL hold:
  1. Required state isn't exposed via any config file, dotfile, dconf key, or \
     shell-readable output.
  2. State can ONLY be effected by interacting with a running app window \
     (e.g. a toolbar button whose effect lives in app memory until saved).
  3. You can identify the target UI element on the screenshot.

## Decision hints
The verifier's getter is the strongest signal:
- `get_vm_file(...)` — reads a file → "code".
- `get_vm_command_line(...)` — shell output → "code".
- App config getters (Chrome / VS Code / GIMP / LibreOffice / VLC) → "code".
- `get_info_from_website(...)` — page state → almost always "code" (curl), \
  occasionally "gui" if click-to-load is required first.
- `get_accessibility_tree(env)` — focused window → may need "gui" to open/focus, \
  but the FOLLOWING state change might still be programmatic; choose by what \
  produces the checked state.

## Response format
Reply with EXACTLY one JSON object on one line, no surrounding prose, no fence, \
no trailing text:

{"mode": "code", "reason": "<one short sentence>"}

`mode` is exactly "code" or "gui". `reason` is a one-sentence rationale logged \
alongside the decision.

Examples:
{"mode": "code", "reason": "verifier reads ~/.config/libreoffice/.../registrymodifications.xcu — direct file write satisfies it"}
{"mode": "gui",  "reason": "verifier reads the GIMP a11y tree and the target Layers panel must be opened first via the Windows menu"}"""


CODE_VERIFIER_SYSTEM = """\
You generate code to verify whether computer-use tasks are solvable on an ubuntu system. Emit one \
Python snippet that produces the state the verifier expects.

## Execution environment
Code runs as `python -c "<YOUR CODE>"`:
- Single inline command. Use semicolons or `exec()` for multi-statement logic, \
  e.g. `exec("import os\\nresult = os.popen('ls').read()\\nprint(result)")`.
- Working directory is the user's home (`~`).

## Allowed (programmatic state changes only)
- Direct file read/write (configs, JSON, plists, INI, XML, dotfiles).
- Shell via `subprocess`, `os.popen`, `os.system`.
- Headless OS config tools: `gsettings set`, `dconf write`, `xdg-mime default`, \
  `update-alternatives`.
- App config files (`~/.config/Code/User/settings.json`, \
  `registrymodifications.xcu`, `gimprc`).
- Python libs (`json`, `configparser`, `plistlib`, `lxml`, `sqlite3`).

## NOT allowed (GUI stage handles these)
- `pyautogui`, `pynput`, `pyperclip`, any keyboard/mouse simulation.
- `xdotool`, `xte`, `wtype`, `ydotool`, `wmctrl` — anything synthesizing GUI \
  input or focusing windows.
- Launching a GUI dialog and "walking through it" by clicking/typing.
- Sleeping for a screenshot — code runs once, headless.

## Cross-domain pitfalls (read before writing any config)
- **Race with running app.** Most apps (Chrome, Thunderbird, VLC, LibreOffice) \
  cache config in memory and rewrite on exit, clobbering edits. If `config` \
  launched the app, `subprocess.run(['pkill', '-9', '<proc>'])` BEFORE \
  editing. VS Code is the exception — it watches settings.json and picks up \
  live edits.

## Response format
Wrap code in a markdown ```python fence — exactly one block. Code inside is \
extracted and inlined into `python -c "<YOUR CODE>"`. For multi-line, use \
`exec(\"\"\"...\"\"\")` inside the fence.

Example:
```python
exec(\"\"\"import os
os.makedirs('/home/user/test', exist_ok=True)
\"\"\")
```

You may include brief reasoning before the fence, but the fence is mandatory \
and must contain complete, self-contained code."""
