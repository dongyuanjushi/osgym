"""
End-to-end Synthesis Pipeline for OSWorld Task Examples and Verifier Functions.

Modes:
  synthesize  - Generate task examples + verifiers via LLM (no VM needed)
  verify      - Execute existing synthetic examples on VMs and validate
  full        - Synthesize then verify

Each example is executed on two parallel environments:
  1. GUI automation  (agent-driven via Qwen3VLWithWM)
  2. Code execution  (LLM-generated Python snippets)

Parallel VM management follows the pattern in run_multienv_docker.py:
  multiprocessing workers pull tasks from a shared queue; each worker
  creates/destroys its own DesktopEnv per task.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import glob
import inspect
import json
import logging
import os
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from multiprocessing import Manager, Process, current_process
from typing import Any, Dict, List, Optional, Tuple

from desktop_env.desktop_env import DesktopEnv
from mm_agents.qwen3_vl import Qwen3VLAgent
from mm_agents.utils.call_llm import call_llm_with_single_response
from mm_agents.utils.utils import encode_screenshot, parse_json_response

logger = logging.getLogger("desktopenv.synthesis")

# Global state for signal handling
_processes: List[Process] = []
_is_terminating = False

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "refactored_evaluation_examples",
    "examples",
)


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis Memory – cross-round experience store
# ═══════════════════════════════════════════════════════════════════════════


class SynthesisMemory:
    """JSON-backed memory of past synthesis rounds.

    Each entry records one task's synthesis + verification outcome so that
    future rounds can:
      - avoid regenerating similar tasks,
      - avoid patterns that consistently fail,
      - steer toward unexplored UI areas.

    File layout (``synthesis_memory.json``):
    ```json
    {
      "entries": [ { ... }, ... ],
      "stats": { "total": N, "verified": N, "rejected": N }
    }
    ```
    """

    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "synthesis_memory.json")
        self.entries: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {"total": 0, "verified": 0, "rejected": 0}

    # -- persistence --------------------------------------------------------

    def load(self) -> "SynthesisMemory":
        if os.path.isfile(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            self.stats = data.get("stats", self.stats)
            logger.info("Loaded synthesis memory: %d entries from %s", len(self.entries), self.path)
        else:
            logger.info("No existing synthesis memory at %s – starting fresh", self.path)
        return self

    def save(self) -> None:
        self.stats = {
            "total": len(self.entries),
            "verified": sum(1 for e in self.entries if e.get("verified")),
            "rejected": sum(1 for e in self.entries if not e.get("verified")),
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"entries": self.entries, "stats": self.stats}, f, indent=2)
        logger.info("Saved synthesis memory (%d entries) to %s", len(self.entries), self.path)

    # -- recording ----------------------------------------------------------

    def record(
        self,
        example: Dict[str, Any],
        domain: str,
        gui_result: Optional[Dict[str, Any]],
        code_result: Optional[Dict[str, Any]],
        verified: bool,
    ) -> None:
        gui_score = gui_result.get("score", 0.0) if gui_result else None
        code_score = code_result.get("score", 0.0) if code_result else None
        gui_error = gui_result.get("error") if gui_result else None
        code_error = code_result.get("error") if code_result else None

        # Build concise failure reasons
        failure_reasons = []
        if not verified:
            if gui_score is None or gui_score <= 0:
                reason = f"gui_score={gui_score}"
                if gui_error:
                    reason += f" error={gui_error[:120]}"
                failure_reasons.append(reason)
            if code_score is None or code_score <= 0:
                reason = f"code_score={code_score}"
                if code_error:
                    reason += f" error={code_error[:120]}"
                failure_reasons.append(reason)

        entry = {
            "id": example.get("id", ""),
            "domain": domain,
            "instruction": example.get("instruction", ""),
            "evaluator_eval": example.get("evaluator", {}).get("eval", ""),
            "round_timestamp": datetime.datetime.now().isoformat(),
            "gui_score": gui_score,
            "code_score": code_score,
            "gui_steps": gui_result.get("steps") if gui_result else None,
            "code_steps": code_result.get("steps") if code_result else None,
            "verified": verified,
            "failure_reasons": failure_reasons,
        }
        self.entries.append(entry)

    # -- querying -----------------------------------------------------------

    def get_domain_entries(self, domain: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("domain") == domain]

    # -- prompt formatting --------------------------------------------------

    def format_for_prompt(self, domain: str, max_entries: int = 30) -> str:
        """Return a text block summarising past experience for *domain*,
        suitable for injection into the LLM user prompt."""
        domain_entries = self.get_domain_entries(domain)
        if not domain_entries:
            return ""

        verified = [e for e in domain_entries if e.get("verified")]
        rejected = [e for e in domain_entries if not e.get("verified")]

        lines: List[str] = []
        lines.append(
            f"## Past synthesis experience for \"{domain}\" "
            f"({len(verified)} verified, {len(rejected)} rejected)\n"
        )

        # --- Verified tasks: show instruction so LLM avoids duplicates ---
        if verified:
            lines.append("### Previously VERIFIED tasks (do NOT generate similar ones):")
            for e in verified[-max_entries:]:
                lines.append(f"  - \"{e['instruction']}\"  [gui={e.get('gui_score')}, code={e.get('code_score')}]")
            lines.append("")

        # --- Rejected tasks with failure analysis ---
        if rejected:
            lines.append("### Previously REJECTED tasks (learn from these failures):")
            for e in rejected[-max_entries:]:
                reasons = "; ".join(e.get("failure_reasons", ["unknown"]))
                lines.append(
                    f"  - \"{e['instruction']}\"\n"
                    f"    evaluator: {e.get('evaluator_eval', 'n/a')[:100]}\n"
                    f"    failure: {reasons}"
                )
            lines.append("")

            # --- Aggregate failure patterns ---
            failure_keywords: Dict[str, int] = {}
            for e in rejected:
                for r in e.get("failure_reasons", []):
                    if "error=" in r:
                        # Extract short error signature
                        err = r.split("error=")[-1][:60]
                        failure_keywords[err] = failure_keywords.get(err, 0) + 1
            if failure_keywords:
                lines.append("### Common failure patterns (avoid these):")
                for kw, cnt in sorted(failure_keywords.items(), key=lambda x: -x[1])[:10]:
                    lines.append(f"  - ({cnt}x) {kw}")
                lines.append("")

        # --- Coverage summary: which UI areas are already explored ---
        all_instructions = [e["instruction"] for e in domain_entries]
        if all_instructions:
            lines.append(
                f"### Coverage: {len(all_instructions)} tasks generated so far. "
                f"Explore UI areas and menu paths NOT yet covered above.\n"
            )

        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# Module 1: Domain Discovery & Information Gathering
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DomainInfo:
    name: str
    example_files: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FunctionCatalog:
    setup_functions: List[Dict[str, str]] = field(default_factory=list)
    getter_functions: List[Dict[str, str]] = field(default_factory=list)
    metric_functions: List[Dict[str, str]] = field(default_factory=list)


def discover_domains() -> List[str]:
    if not os.path.isdir(EXAMPLES_DIR):
        raise FileNotFoundError(f"Examples directory not found: {EXAMPLES_DIR}")
    domains = sorted(
        d for d in os.listdir(EXAMPLES_DIR)
        if os.path.isdir(os.path.join(EXAMPLES_DIR, d))
    )
    logger.info("Discovered %d domains: %s", len(domains), domains)
    return domains


def load_domain_examples(domain: str, max_examples: int = 0) -> DomainInfo:
    domain_dir = os.path.join(EXAMPLES_DIR, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
    info = DomainInfo(name=domain)
    json_files = sorted(glob.glob(os.path.join(domain_dir, "*.json")))
    info.example_files = json_files
    to_load = json_files if max_examples <= 0 else json_files[:max_examples]
    for fp in to_load:
        with open(fp, "r") as f:
            info.examples.append(json.load(f))
    logger.info("Loaded %d/%d examples for domain '%s'", len(info.examples), len(json_files), domain)
    return info


def _func_info(obj, name: str) -> Dict[str, str]:
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = "(...)"
    doc = (inspect.getdoc(obj) or "").split("\n")[0]
    return {"name": name, "signature": f"{name}{sig}", "doc": doc}


def catalog_functions() -> FunctionCatalog:
    cat = FunctionCatalog()

    from desktop_env.controllers.setup import SetupController
    for name in sorted(dir(SetupController)):
        if name.startswith("_") and name.endswith("_setup"):
            obj = getattr(SetupController, name)
            if callable(obj):
                cat.setup_functions.append(_func_info(obj, name))

    from desktop_env.evaluators import getters as gmod
    for name in sorted(dir(gmod)):
        if name.startswith("get_"):
            obj = getattr(gmod, name)
            if callable(obj):
                cat.getter_functions.append(_func_info(obj, name))

    from desktop_env.evaluators import metrics as mmod
    for name in sorted(dir(mmod)):
        obj = getattr(mmod, name, None)
        if callable(obj) and not name.startswith("_"):
            cat.metric_functions.append(_func_info(obj, name))

    logger.info(
        "Cataloged %d setup, %d getter, %d metric functions",
        len(cat.setup_functions), len(cat.getter_functions), len(cat.metric_functions),
    )
    return cat


# ═══════════════════════════════════════════════════════════════════════════
# Module 2: LLM-Based Task Example Generation
# ═══════════════════════════════════════════════════════════════════════════

TASK_GEN_SYSTEM = """You are an expert at generating desktop automation task examples for the OSWorld benchmark.

FOCUS: Generate tasks that **explore the software layout and menus** of each application. \
The goal is to produce tasks whose execution reveals rich environment dynamics — how the UI \
reacts to interactions, what state transitions menus trigger, and how panels/toolbars/dialogs \
rearrange when features are activated. Good tasks exercise the spatial and interactive \
structure of the application rather than just its data-processing capabilities.

Prefer tasks that:
- Open, close, toggle, or rearrange UI panels, sidebars, toolbars, and status bars \
  (e.g., View > Show Sidebar, View > Toolbars > Drawing)
- Navigate through nested menu hierarchies to discover and activate features \
  (e.g., Tools > Options > Language Settings, Format > Columns > Two)
- Trigger dialogs, pop-ups, or context menus and interact with their controls \
  (e.g., Insert > Special Character, right-click > Paragraph Properties)
- Switch between application modes or views that change the visible layout \
  (e.g., Normal / Outline / Master view in Impress, Reader / Source in VS Code)
- Adjust display settings such as zoom, page layout, ruler visibility, or grid snapping
- Explore tab/workspace management (new tab, split view, tab groups, window tiling)
- Activate or configure inspectors, property panels, layer dialogs, or developer tools

Avoid tasks that are purely data-entry (typing long text, filling spreadsheet cells) \
or that only read information without changing any UI state.

Each task example is a JSON object with these fields:
- "id": a UUID4 string
- "snapshot": the application snapshot name (usually the domain or app name)
- "instruction": a one-sentence natural language description of the task
- "source": "synthetic"
- "config": a list of setup function call strings that prepare the environment
- "related_apps": a list of application names involved
- "evaluator": an object with optional "postconfig" (list of setup strings) and "eval" \
(a Python expression composing getter + metric functions)

Rules:
1. The instruction must be concrete and achievable on an Ubuntu desktop.
2. The task MUST explore the application's layout, menus, or UI structure.
3. The config must use ONLY the setup functions provided.
4. The evaluator "eval" must compose getter and metric functions from the library provided.
5. Do NOT invent new getter/metric functions unless absolutely necessary.
6. Return valid JSON only, no markdown fences."""


def _fmt_funcs(funcs: List[Dict[str, str]]) -> str:
    return "\n".join(f"- {f['signature']}: {f['doc']}" for f in funcs)


def generate_task_examples(
    domain_info: DomainInfo,
    catalog: FunctionCatalog,
    llm_config: Dict[str, Any],
    num_to_generate: int = 1,
    max_steps: int = 15,
    memory: Optional[SynthesisMemory] = None,
) -> List[Dict[str, Any]]:
    ref = json.dumps(domain_info.examples[:5], indent=2)

    # Build memory context (empty string if no prior experience)
    memory_block = ""
    if memory is not None:
        memory_block = memory.format_for_prompt(domain_info.name)
        if memory_block:
            memory_block = f"\n{memory_block}\n"

    user = (
        f"Domain: {domain_info.name}\n"
        f"Existing examples: {len(domain_info.example_files)}\n\n"
        f"## Complexity budget\n"
        f"Each task must be completable by a GUI agent in at most **{max_steps} steps** "
        f"(each step = one mouse click, keystroke, or typed string). "
        f"Keep tasks focused on exploring one aspect of the software layout or menu "
        f"structure per task.\n\n"
        f"{memory_block}"
        f"## Reference examples\n{ref}\n\n"
        f"## Setup functions\n{_fmt_funcs(catalog.setup_functions)}\n\n"
        f"## Getter functions\n{_fmt_funcs(catalog.getter_functions)}\n\n"
        f"## Metric functions\n{_fmt_funcs(catalog.metric_functions)}\n\n"
        f"Generate {num_to_generate} new, diverse task example(s) for \"{domain_info.name}\".\n"
        f"Each task should explore a different part of the software's layout, menus, or UI "
        f"panels to maximize coverage of the application's interactive structure.\n"
        f"Return a JSON array of task objects."
    )
    messages = [
        {"role": "system", "content": TASK_GEN_SYSTEM},
        {"role": "user", "content": user},
    ]
    logger.info("Generating %d task example(s) for '%s' ...", num_to_generate, domain_info.name)
    raw = call_llm_with_single_response(messages=messages, llm_config=llm_config, max_tokens=4096, temperature=0.7)
    logger.info("LLM response: %d chars", len(raw))

    parsed = parse_json_response(raw)
    if parsed is None:
        import re
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON array")
                return []
        else:
            logger.error("No JSON array found in LLM response")
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    for ex in parsed:
        if not ex.get("id"):
            ex["id"] = str(uuid.uuid4())
    logger.info("Generated %d example(s)", len(parsed))
    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# Module 3: Verifier Function Generation
# ═══════════════════════════════════════════════════════════════════════════

VERIFIER_GEN_SYSTEM = """You are an expert at composing verifier (evaluator) expressions for OSWorld desktop tasks.

A verifier expression is a single Python expression that:
1. Uses getter functions (signature: getter(env, config={...})) to extract state from the VM.
2. Uses metric functions to compare extracted state against expected values.
3. Conditions combine with Python `or` / `and`.
4. get_rule(env, config={'rules': ...}) passes expected values.

You may also return a "postconfig" list of setup function calls to run before evaluation.

Rules:
- ONLY use the getter and metric functions provided.
- Return a JSON object: {"postconfig": [...], "eval": "..."}"""


def generate_verifier(
    task_instruction: str,
    task_config: List[str],
    domain: str,
    catalog: FunctionCatalog,
    reference_evaluators: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    ref = json.dumps(reference_evaluators[:5], indent=2)
    user = (
        f"Domain: {domain}\nInstruction: {task_instruction}\n"
        f"Config: {json.dumps(task_config)}\n\n"
        f"## Reference evaluators\n{ref}\n\n"
        f"## Setup functions (for postconfig)\n{_fmt_funcs(catalog.setup_functions)}\n\n"
        f"## Getter functions\n{_fmt_funcs(catalog.getter_functions)}\n\n"
        f"## Metric functions\n{_fmt_funcs(catalog.metric_functions)}\n\n"
        f"Generate a verifier. Return JSON: {{\"postconfig\": [...], \"eval\": \"...\"}}"
    )
    messages = [
        {"role": "system", "content": VERIFIER_GEN_SYSTEM},
        {"role": "user", "content": user},
    ]
    logger.info("Generating verifier for: '%s' ...", task_instruction[:80])
    raw = call_llm_with_single_response(messages=messages, llm_config=llm_config, max_tokens=2048, temperature=0.3)
    parsed = parse_json_response(raw)
    if parsed is None:
        logger.error("Failed to parse verifier response")
        return {"postconfig": [], "eval": ""}
    ev = {"postconfig": parsed.get("postconfig", []), "eval": parsed.get("eval", "")}
    logger.info("Verifier eval: %s", ev["eval"][:120])
    return ev


# ═══════════════════════════════════════════════════════════════════════════
# Module 3b: Static Validation of Setup / Evaluator Scripts
# ═══════════════════════════════════════════════════════════════════════════


def _collect_valid_names(catalog: FunctionCatalog) -> Tuple[set, set, set]:
    """Return (setup_names, getter_names, metric_names) from the catalog."""
    setup_names = {f["name"] for f in catalog.setup_functions}
    getter_names = {f["name"] for f in catalog.getter_functions}
    metric_names = {f["name"] for f in catalog.metric_functions}
    return setup_names, getter_names, metric_names


def _validate_setup_string(s: str, valid_setup_names: set) -> Optional[str]:
    """Check one setup config/postconfig string.  Returns an error message or None."""
    # Extract function name: everything before the first '('
    paren = s.find("(")
    if paren == -1:
        return f"not a function call: {s!r}"
    func_name = s[:paren].strip()
    if func_name not in valid_setup_names:
        return f"unknown setup function: {func_name!r}"
    # Try parsing as Python to catch syntax errors
    try:
        ast.parse(s, mode="eval")
    except SyntaxError as e:
        return f"syntax error in setup string: {e}"
    return None


def _extract_call_names(node: ast.AST) -> List[str]:
    """Walk an AST and collect every function-call name (direct calls only)."""
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            names.append(child.func.id)
    return names


def _validate_eval_expr(eval_str: str, valid_getter_names: set,
                        valid_metric_names: set) -> Optional[str]:
    """Check one evaluator eval expression.  Returns an error message or None."""
    if not eval_str or not eval_str.strip():
        return "empty eval expression"
    try:
        tree = ast.parse(eval_str, mode="eval")
    except SyntaxError as e:
        return f"syntax error in eval expression: {e}"

    called = _extract_call_names(tree)
    if not called:
        return "eval expression contains no function calls"

    # Every called name must be a known getter, metric, or Python builtin
    # that the eval namespace allows (env, config are names, not calls)
    allowed = valid_getter_names | valid_metric_names | {"float", "int", "str", "len", "bool", "abs", "max", "min"}
    unknown = [n for n in called if n not in allowed]
    if unknown:
        return f"unknown function(s) in eval: {unknown}"
    return None


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_example_scripts(example: Dict[str, Any], catalog: FunctionCatalog) -> ValidationResult:
    """Statically validate the setup config and evaluator scripts of a synthesized example.

    Checks:
      1. Every config entry references a known setup function and is syntactically valid Python.
      2. Every postconfig entry (same check).
      3. The evaluator eval expression parses as Python and only calls known getter/metric functions.

    Returns a ValidationResult with .valid and .errors.
    """
    setup_names, getter_names, metric_names = _collect_valid_names(catalog)
    errors: List[str] = []

    # --- config ---
    for i, entry in enumerate(example.get("config", [])):
        err = _validate_setup_string(entry, setup_names)
        if err:
            errors.append(f"config[{i}]: {err}")

    # --- evaluator.postconfig ---
    evaluator = example.get("evaluator", {})
    for i, entry in enumerate(evaluator.get("postconfig", [])):
        err = _validate_setup_string(entry, setup_names)
        if err:
            errors.append(f"postconfig[{i}]: {err}")

    # --- evaluator.eval ---
    eval_str = evaluator.get("eval", "")
    err = _validate_eval_expr(eval_str, getter_names, metric_names)
    if err:
        errors.append(f"eval: {err}")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ═══════════════════════════════════════════════════════════════════════════
# Synthesize stage: generate examples + verifiers (no VM)
# ═══════════════════════════════════════════════════════════════════════════


def run_synthesize(args: argparse.Namespace, memory: Optional[SynthesisMemory] = None) -> List[Dict[str, Any]]:
    """Generate synthetic examples + verifiers via LLM. Returns flat list of examples."""
    all_domains = discover_domains()
    catalog = catalog_functions()
    targets = args.domains if args.domains else all_domains
    targets = [d for d in targets if d in all_domains]

    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    os.makedirs(args.output_dir, exist_ok=True)
    all_examples: List[Dict[str, Any]] = []

    for domain in targets:
        logger.info("=" * 60)
        logger.info("Synthesizing for domain: %s", domain)
        domain_info = load_domain_examples(domain, max_examples=args.max_ref_examples)
        ref_evaluators = [ex.get("evaluator", {}) for ex in domain_info.examples]

        examples = generate_task_examples(
            domain_info, catalog, llm_config, args.num_examples, args.max_steps, memory,
        )

        for ex in examples:
            if not ex.get("evaluator", {}).get("eval"):
                ex["evaluator"] = generate_verifier(
                    ex.get("instruction", ""), ex.get("config", []),
                    domain, catalog, ref_evaluators, llm_config,
                )
            ex.setdefault("_domain", domain)

        # ---- Static validation: reject examples with bad scripts early ----
        valid_examples = []
        for ex in examples:
            vr = validate_example_scripts(ex, catalog)
            if vr.valid:
                valid_examples.append(ex)
            else:
                logger.warning(
                    "SCRIPT VALIDATION FAILED for %s: %s – skipping",
                    ex.get("id", "?"), "; ".join(vr.errors),
                )
                # Record into memory as a rejected synthesis (no scores)
                if memory is not None:
                    memory.record(
                        example=ex, domain=domain,
                        gui_result={"score": 0.0, "error": f"script_validation: {'; '.join(vr.errors)}"},
                        code_result={"score": 0.0, "error": f"script_validation: {'; '.join(vr.errors)}"},
                        verified=False,
                    )
        logger.info(
            "Domain '%s': %d/%d examples passed script validation",
            domain, len(valid_examples), len(examples),
        )

        # Persist only validated examples
        domain_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        for ex in valid_examples:
            path = os.path.join(domain_dir, f"{ex['id']}.json")
            with open(path, "w") as f:
                json.dump(ex, f, indent=2)
            logger.info("Saved %s", path)

        all_examples.extend(valid_examples)

    # Save memory with any script-validation failures recorded above
    if memory is not None:
        memory.save()

    # Write a manifest listing every generated (and validated) example
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = {}
    for ex in all_examples:
        d = ex.get("_domain", "unknown")
        manifest.setdefault(d, []).append(ex["id"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest with %d examples saved to %s", len(all_examples), manifest_path)
    return all_examples


# ═══════════════════════════════════════════════════════════════════════════
# Module 4: Parallel Verification Workers  (follows run_multienv_docker.py)
# ═══════════════════════════════════════════════════════════════════════════


def create_environment(args: argparse.Namespace, docker_host: str = None, host_id: str = None):
    remote_host = None if docker_host == "localhost" else docker_host
    return DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=False,
        remote_host=remote_host,
        host_id=host_id,
    )


def cleanup_environment(env, tag: str) -> None:
    if env is not None:
        try:
            env.close()
            logger.info("[%s] Environment closed", tag)
        except Exception as e:
            logger.error("[%s] Error closing env: %s", tag, e)


def run_gui_task(agent, env, example: Dict[str, Any], args, result_dir: str) -> Dict[str, Any]:
    """Execute one example via GUI agent. Returns result dict."""
    proc = current_process().name
    instruction = example["instruction"]
    agent.reset(result_dir)

    env.reset(task_config=example)
    time.sleep(5)
    obs = env._get_obs()

    trajectory = []
    done = False
    step_idx = 0

    while not done and step_idx < args.max_steps:
        _, thought, action_code = agent.predict(instruction, obs)
        ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        logger.info("[%s][gui] step %d: %s", proc, step_idx, str(action_code)[:100])

        trajectory.append({"step": step_idx, "timestamp": ts, "thought": thought, "action": action_code})

        if action_code in ("DONE", "FAIL"):
            done = True
            break

        obs, _, done, info = env.step(action_code, args.sleep_after_execution)

        # Save screenshot
        if obs.get("screenshot"):
            with open(os.path.join(result_dir, f"step_{step_idx}_{ts}.png"), "wb") as fp:
                fp.write(obs["screenshot"])
        step_idx += 1

    time.sleep(5)
    score = env.evaluate()
    logger.info("[%s][gui] score=%.2f for %s", proc, score, example["id"])

    # Persist
    with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
        json.dump(trajectory, fp, indent=2)
    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write(f"{score}\n")
    return {"id": example["id"], "mode": "gui", "score": score, "steps": len(trajectory)}


def run_code_task(env, example: Dict[str, Any], args, llm_config: Dict[str, Any], result_dir: str) -> Dict[str, Any]:
    """Execute one example via LLM-generated code. Returns result dict."""
    proc = current_process().name
    instruction = example["instruction"]

    env.reset(task_config=example)
    time.sleep(5)
    obs = env._get_obs()

    code_system = (
        "You are a desktop automation agent. Generate a single Python snippet to execute "
        "on the VM. Available: pyautogui, subprocess, os, time. "
        "Return ONLY the code. Use 'DONE' when finished, 'FAIL' if infeasible."
    )

    trajectory = []
    done = False
    step_idx = 0

    while not done and step_idx < args.max_steps:
        history_str = "\n".join(f"Step {t['step']}: {t['code']}" for t in trajectory[-3:])
        messages = [
            {"role": "system", "content": code_system},
            {"role": "user", "content": f"Task: {instruction}\n\nPrevious:\n{history_str}\n\nNext code:"},
        ]
        if obs.get("screenshot"):
            b64 = encode_screenshot(obs["screenshot"])
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": b64}},
                {"type": "text", "text": "Current screenshot."},
            ]})

        raw = call_llm_with_single_response(messages=messages, llm_config=llm_config, max_tokens=1024, temperature=0.2)
        code = raw.strip().strip("`").strip()
        if code.startswith("python"):
            code = code[6:].strip()

        ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        logger.info("[%s][code] step %d: %s", proc, step_idx, code[:100])
        trajectory.append({"step": step_idx, "timestamp": ts, "code": code})

        if code in ("DONE", "FAIL"):
            done = True
            break

        obs, _, done, info = env.step(code, args.sleep_after_execution)

        if obs.get("screenshot"):
            with open(os.path.join(result_dir, f"step_{step_idx}_{ts}.png"), "wb") as fp:
                fp.write(obs["screenshot"])
        step_idx += 1

    time.sleep(5)
    score = env.evaluate()
    logger.info("[%s][code] score=%.2f for %s", proc, score, example["id"])

    with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
        json.dump(trajectory, fp, indent=2)
    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write(f"{score}\n")
    return {"id": example["id"], "mode": "code", "score": score, "steps": len(trajectory)}


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker(task_queue, args: argparse.Namespace, shared_results: list,
            docker_host: str = None, host_id: str = None):
    """Worker pulled from the task queue. Each task is (example_dict, exec_mode)."""
    proc = current_process().name
    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    screen_size = (args.screen_width, args.screen_height)

    # Build agent once per worker (reused across tasks in this process)
    agent = Qwen3VLAgent(
        screen_size=screen_size, approach="greedy",
        policy_model=args.policy_model, policy_model_provider=args.policy_model_provider,
        policy_model_endpoint=args.policy_model_endpoint,
        logger=logger
    )

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break  # queue empty / timeout

        example, exec_mode = item
        env = None
        try:
            logger.info("[%s] Creating env for %s/%s ...", proc, exec_mode, example["id"])
            env = create_environment(args, docker_host, host_id)

            result_dir = os.path.join(
                args.output_dir, example.get("_domain", "unknown"),
                "results", example["id"], exec_mode,
            )
            os.makedirs(result_dir, exist_ok=True)

            if exec_mode == "gui":
                res = run_gui_task(agent, env, example, args, result_dir)
            else:
                res = run_code_task(env, example, args, llm_config, result_dir)

            shared_results.append(res)

        except KeyboardInterrupt:
            logger.warning("[%s] KeyboardInterrupt", proc)
            break
        except Exception as e:
            logger.error("[%s] Error on %s/%s: %s", proc, exec_mode, example["id"], e)
            logger.error(traceback.format_exc())
            shared_results.append({"id": example["id"], "mode": exec_mode, "score": 0.0, "error": str(e)})
        finally:
            cleanup_environment(env, proc)

    logger.info("[%s] Worker finished", proc)


# ═══════════════════════════════════════════════════════════════════════════
# Verify stage: parallel execution & evaluation
# ═══════════════════════════════════════════════════════════════════════════


def _load_synthetic_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load examples from the output dir (generated by synthesize stage)."""
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No manifest at {manifest_path}. Run --mode synthesize first.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    examples = []
    for domain, ids in manifest.items():
        for eid in ids:
            path = os.path.join(args.output_dir, domain, f"{eid}.json")
            if os.path.isfile(path):
                with open(path) as f:
                    ex = json.load(f)
                ex.setdefault("_domain", domain)
                examples.append(ex)
            else:
                logger.warning("Example file not found: %s", path)
    logger.info("Loaded %d synthetic examples from %s", len(examples), manifest_path)
    return examples


def run_verify(args: argparse.Namespace, examples: Optional[List[Dict[str, Any]]] = None,
               memory: Optional[SynthesisMemory] = None) -> List[Dict[str, Any]]:
    """Execute synthetic examples on parallel Docker VMs and evaluate."""
    global _processes

    if examples is None:
        examples = _load_synthetic_examples(args)
    if not examples:
        logger.error("No examples to verify")
        return []

    # Parse host config  e.g. '{"localhost": 4}'
    host_config = json.loads(args.host_env_config)
    process_assignments = []
    idx = 0
    for host, n in host_config.items():
        for _ in range(n):
            process_assignments.append((host, idx))
            idx += 1
    total_workers = len(process_assignments)
    logger.info("Verification: %d examples x 2 modes = %d tasks across %d workers",
                len(examples), len(examples) * 2, total_workers)

    manager = Manager()
    shared_results = manager.list()
    task_queue = manager.Queue()

    # Enqueue (example, mode) pairs -- gui first, then code
    for ex in examples:
        task_queue.put((ex, "gui"))
        task_queue.put((ex, "code"))

    _processes = []
    for host, pidx in process_assignments:
        p = Process(
            target=worker,
            args=(task_queue, args, shared_results, host, f"docker-{pidx}"),
            name=f"SynthWorker-{pidx}",
            daemon=True,
        )
        p.start()
        _processes.append(p)
        logger.info("Started %s (PID %d, host=%s)", p.name, p.pid, host)

    # Monitor
    try:
        while True:
            if task_queue.empty():
                break
            if not any(p.is_alive() for p in _processes):
                logger.error("All workers died")
                break
            time.sleep(5)
        for p in _processes:
            p.join(timeout=60)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt - terminating workers")
    finally:
        _cleanup_processes(_processes)
        try:
            manager.shutdown()
        except Exception:
            pass

    results = list(shared_results)

    # Persist all raw verification results
    agg_path = os.path.join(args.output_dir, "verification_results.json")
    with open(agg_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    gui_scores = [r["score"] for r in results if r.get("mode") == "gui" and "error" not in r]
    code_scores = [r["score"] for r in results if r.get("mode") == "code" and "error" not in r]
    logger.info("GUI  avg=%.3f (%d tasks)", sum(gui_scores) / max(len(gui_scores), 1), len(gui_scores))
    logger.info("Code avg=%.3f (%d tasks)", sum(code_scores) / max(len(code_scores), 1), len(code_scores))

    # ---- Filter: keep only examples where BOTH gui AND code scored > 0 ----
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in results:
        eid = r["id"]
        by_id.setdefault(eid, {})[r.get("mode", "unknown")] = r

    verified_dir = os.path.join(args.output_dir, "verified")
    os.makedirs(verified_dir, exist_ok=True)

    verified_ids = []
    rejected_ids = []
    for eid, modes in by_id.items():
        gui_r = modes.get("gui")
        code_r = modes.get("code")
        gui_ok = gui_r is not None and "error" not in gui_r and gui_r.get("score", 0) > 0
        code_ok = code_r is not None and "error" not in code_r and code_r.get("score", 0) > 0

        if gui_ok and code_ok:
            verified_ids.append(eid)
        else:
            rejected_ids.append(eid)
            reasons = []
            if not gui_ok:
                reasons.append(f"gui={'error' if gui_r and 'error' in gui_r else gui_r.get('score', 'missing') if gui_r else 'not_run'}")
            if not code_ok:
                reasons.append(f"code={'error' if code_r and 'error' in code_r else code_r.get('score', 'missing') if code_r else 'not_run'}")
            logger.info("REJECTED %s: %s", eid, ", ".join(reasons))

    # Record outcomes into synthesis memory
    if memory is not None:
        ex_by_id = {ex["id"]: ex for ex in examples}
        for eid, modes in by_id.items():
            ex = ex_by_id.get(eid)
            if ex is None:
                continue
            memory.record(
                example=ex,
                domain=ex.get("_domain", "unknown"),
                gui_result=modes.get("gui"),
                code_result=modes.get("code"),
                verified=eid in verified_ids,
            )
        memory.save()

    # Copy verified examples to verified/ directory
    verified_manifest = {}
    for ex in examples:
        if ex["id"] not in verified_ids:
            continue
        domain = ex.get("_domain", "unknown")
        domain_dir = os.path.join(verified_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        path = os.path.join(domain_dir, f"{ex['id']}.json")
        with open(path, "w") as f:
            json.dump(ex, f, indent=2)
        verified_manifest.setdefault(domain, []).append(ex["id"])

    # Write verified manifest
    verified_manifest_path = os.path.join(verified_dir, "manifest.json")
    with open(verified_manifest_path, "w") as f:
        json.dump(verified_manifest, f, indent=2)

    logger.info(
        "Verification complete: %d verified, %d rejected out of %d total",
        len(verified_ids), len(rejected_ids), len(by_id),
    )
    logger.info("Verified examples saved to %s", verified_dir)
    logger.info("All raw results saved to %s", agg_path)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Process cleanup (mirrors run_multienv_docker.py)
# ═══════════════════════════════════════════════════════════════════════════


def _cleanup_processes(procs: List[Process], timeout: int = 10):
    alive = []
    for p in procs:
        if p is not None and p.is_alive():
            try:
                p.terminate()
                alive.append(p)
            except Exception:
                pass
    t0 = time.time()
    while alive and time.time() - t0 < timeout:
        alive = [p for p in alive if p.is_alive()]
        if not alive:
            break
        time.sleep(0.5)
    for p in alive:
        try:
            os.kill(p.pid, signal.SIGKILL)
        except Exception:
            pass


def _signal_handler(signum, frame):
    global _is_terminating
    if _is_terminating:
        return
    _is_terminating = True
    logger.info("Signal %d received - shutting down", signum)
    _cleanup_processes(_processes)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OSWorld synthesis pipeline")

    # Mode
    p.add_argument("--mode", choices=["synthesize", "verify", "full"], default="full",
                    help="Pipeline mode")
    p.add_argument("--list-domains", action="store_true", help="List domains and exit")

    # Synthesis args
    p.add_argument("--domains", nargs="*", default=[], help="Target domains (empty=all)")
    p.add_argument("--num-examples", type=int, default=3, help="Examples to generate per domain")
    p.add_argument("--max-ref-examples", type=int, default=10, help="Reference examples for LLM context")

    # LLM config (shared by synthesis and code-execution)
    p.add_argument("--model", type=str, required=False, default="anthropic.claude-sonnet-4-20250514")
    p.add_argument("--provider", type=str, default="bedrock")
    p.add_argument("--endpoint", type=str, default="")

    # Agent / policy model (for GUI trajectories)
    p.add_argument("--policy-model", type=str, default="")
    p.add_argument("--policy-model-provider", type=str, default="")
    p.add_argument("--policy-model-endpoint", type=str, default="")


    # Environment
    p.add_argument("--provider-name", type=str, default="docker")
    p.add_argument("--path-to-vm", type=str, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--action-space", type=str, default="pyautogui")
    p.add_argument("--screen-width", type=int, default=1920)
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--sleep-after-execution", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=15,
                    help="Max steps for GUI/code execution; also guides task complexity during synthesis")

    # Parallelism
    p.add_argument("--host-env-config", type=str, default='{"localhost": 2}',
                    help='JSON: {"host": num_workers, ...}')

    # Output
    p.add_argument("--output-dir", type=str, default="synthetic_output")
    p.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = build_parser()
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="\x1b[1;33m[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s]\x1b[0m %(message)s",
    )

    if args.list_domains:
        for d in discover_domains():
            n = len(glob.glob(os.path.join(EXAMPLES_DIR, d, "*.json")))
            print(f"  {d}: {n} examples")
        return

    # ---- Load synthesis memory (persists across rounds) ----
    memory = SynthesisMemory(args.output_dir).load()

    # ---- Dispatch by mode ----
    if args.mode in ("synthesize", "full"):
        examples = run_synthesize(args, memory)
    else:
        examples = None  # verify mode loads from disk

    if args.mode in ("verify", "full"):
        run_verify(args, examples, memory)


if __name__ == "__main__":
    main()
