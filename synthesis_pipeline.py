"""
End-to-end Synthesis Pipeline for OSWorld Task Examples and Verifier Functions.

Modes:
  synthesize  - Generate task examples + verifiers via LLM (no VM needed)
  verify      - Execute existing synthetic examples on VMs and validate
  full        - Synthesize then verify

Verification is done via code execution: the LLM generates a Python snippet
that is run on the VM, then the evaluator checks the resulting state.

Verification workers are HTTP clients that call the OSGym API server
(main.py) for VM allocation, stepping, evaluation, and shutdown.
Multiprocessing workers pull tasks from a shared queue; the API server
manages Docker VM lifecycle.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime
import glob
import inspect
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from multiprocessing import Manager, Process, current_process
from typing import Any, Dict, List, Optional, Tuple

import requests as http_requests

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

    Entry lifecycle:
      executable=False              → script failed static validation
      executable=True, solvable=False → valid scripts but agent execution failed
      executable=True, solvable=True  → fully verified via agent execution

    File layout (``synthesis_memory.json``):
    ```json
    {
      "entries": [ { ... }, ... ],
      "stats": { "total": N, "executable": N, "solvable": N }
    }
    ```
    """

    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "synthesis_memory.json")
        self.entries: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {"total": 0, "executable": 0, "solvable": 0}

    # -- persistence --------------------------------------------------------

    def load(self) -> "SynthesisMemory":
        if os.path.isfile(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            self.stats = data.get("stats", self.stats)
            # Migrate old entries that used "verified" instead of
            # "executable"/"solvable"
            for e in self.entries:
                if "executable" not in e and "verified" in e:
                    e["executable"] = e.pop("verified")
                if "solvable" not in e:
                    e["solvable"] = False
            logger.info(f"Loaded synthesis memory: {len(self.entries)} entries from {self.path}")
        else:
            logger.info(f"No existing synthesis memory at {self.path} – starting fresh")
        return self

    def save(self) -> None:
        self.stats = {
            "total": len(self.entries),
            "executable": sum(1 for e in self.entries if e.get("executable")),
            "solvable": sum(1 for e in self.entries if e.get("solvable")),
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"entries": self.entries, "stats": self.stats}, f, indent=2)
        logger.info(f"Saved synthesis memory ({len(self.entries)} entries) to {self.path}")

    # -- recording ----------------------------------------------------------

    def record(
        self,
        example: Dict[str, Any],
        domain: str,
        code_result: Optional[Dict[str, Any]],
        executable: bool,
        solvable: bool = None,
    ) -> None:
        code_score = code_result.get("score", 0.0) if code_result else None

        # Build concise failure reasons
        failure_reasons = []
        if not executable:
            failure_reasons.append("The scripts for either setup or evaluator are not executable")
        if solvable is not None and not solvable:
            failure_reasons.append("The generated task can not be solved by code execution verification")

        entry = {
            "id": example.get("id", ""),
            "domain": domain,
            "instruction": example.get("instruction", ""),
            "evaluator_eval": example.get("evaluator", {}).get("eval", ""),
            "round_timestamp": datetime.datetime.now().isoformat(),
            "code_score": code_score,
            "code_steps": code_result.get("steps") if code_result else None,
            "executable": executable,
            "solvable": solvable,
            "failure_reasons": failure_reasons,
        }

        # Upsert: update existing entry for the same id, or append new one.
        eid = entry["id"]
        for i, existing in enumerate(self.entries):
            if existing.get("id") == eid:
                self.entries[i] = entry
                return
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

        solvable = [e for e in domain_entries if e.get("solvable")]
        executable_only = [
            e for e in domain_entries
            if e.get("executable") and not e.get("solvable")
        ]
        not_executable = [e for e in domain_entries if not e.get("executable")]

        lines: List[str] = []
        lines.append(
            f"## Past synthesis experience for \"{domain}\" "
            f"({len(solvable)} solvable, {len(executable_only)} executable-only, "
            f"{len(not_executable)} not-executable)\n"
        )

        # --- Solvable tasks: show instruction so LLM avoids duplicates ---
        if solvable:
            lines.append("### Previously SOLVABLE tasks (do NOT generate similar ones):")
            for e in solvable[-max_entries:]:
                lines.append(
                    f"  - \"{e['instruction']}\"  "
                    f"[code_score={e.get('code_score')}]"
                )
            lines.append("")

        # --- Executable but not solvable: valid scripts, agent failed ---
        if executable_only:
            lines.append(
                "### Previously EXECUTABLE but NOT SOLVABLE tasks "
                "(valid scripts, agent execution failed):"
            )
            for e in executable_only[-max_entries:]:
                reasons = "; ".join(e.get("failure_reasons", ["unknown"]))
                lines.append(
                    f"  - \"{e['instruction']}\"\n"
                    f"    evaluator: {e.get('evaluator_eval', 'n/a')[:100]}\n"
                    f"    failure: {reasons}"
                )
            lines.append("")

        # --- Not executable: invalid scripts ---
        if not_executable:
            lines.append(
                "### Previously NOT EXECUTABLE tasks "
                "(invalid scripts — learn from these):"
            )
            for e in not_executable[-max_entries:]:
                reasons = "; ".join(e.get("failure_reasons", ["unknown"]))
                lines.append(
                    f"  - \"{e['instruction']}\"\n"
                    f"    evaluator: {e.get('evaluator_eval', 'n/a')[:100]}\n"
                    f"    failure: {reasons}"
                )
            lines.append("")

        # --- Aggregate failure patterns from all non-solvable entries ---
        failed_entries = executable_only + not_executable
        if failed_entries:
            failure_keywords: Dict[str, int] = {}
            for e in failed_entries:
                for r in e.get("failure_reasons", []):
                    if "error=" in r:
                        err = r.split("error=")[-1][:60]
                        failure_keywords[err] = failure_keywords.get(err, 0) + 1
            if failure_keywords:
                lines.append("### Common failure patterns (avoid these):")
                for kw, cnt in sorted(failure_keywords.items(), key=lambda x: -x[1])[:10]:
                    lines.append(f"  - ({cnt}x) {kw}")
                lines.append("")

        # --- Coverage summary ---
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
    logger.info(f"Discovered {len(domains)} domains: {domains}")
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
    logger.info(f"Loaded {len(info.examples)}/{len(json_files)} examples for domain '{domain}'")
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

    logger.info(f"Cataloged {len(cat.setup_functions)} setup, {len(cat.getter_functions)} getter, {len(cat.metric_functions)} metric functions")
    return cat


# ═══════════════════════════════════════════════════════════════════════════
# Module 2: LLM-Based Task Example Generation
# ═══════════════════════════════════════════════════════════════════════════

TASK_GEN_SYSTEM = """You are an expert at generating desktop automation task examples for the OSWorld benchmark.

FOCUS: Generate tasks that **explore the software layout and menus** of each application \
while producing **concrete, persistent state changes** in the environment. The goal is to \
produce tasks whose execution reveals rich environment dynamics — how the UI reacts to \
interactions, what state transitions menus trigger, and how the underlying application or \
system state is modified. Good tasks exercise both the interactive structure of the \
application and its ability to alter observable state (files on disk, application \
configuration, document properties, system settings, installed extensions, etc.).

Prefer tasks that:
- Change application or document settings that are reflected in configuration files or \
  application state (e.g., changing the default font in LibreOffice Writer, setting the \
  tab size in VS Code, enabling autosave)
- Modify document properties or structure through menu interactions \
  (e.g., setting page orientation to landscape, inserting a table with specific dimensions, \
  changing paragraph spacing, adding headers/footers)
- Toggle or configure features whose effect persists beyond the visual session \
  (e.g., enabling line numbers, turning on word wrap, activating spell check, setting \
  a specific zoom level that the application remembers)
- Navigate through nested menu hierarchies to activate features that alter the working \
  environment (e.g., Tools > Options > Language Settings to change locale, \
  Format > Columns > Two to restructure layout)
- Create, rename, move, or organize files and folders through the application's own UI \
  (e.g., Save As to a new location, Export as PDF, create a new project folder)
- Adjust application preferences or workspace configurations that produce side effects \
  in the file system or internal state (e.g., changing theme, setting a custom dictionary \
  path, configuring a build command)
- Use keyboard shortcuts or command palettes to perform actions that have equivalent \
  menu-driven paths (e.g., Ctrl+Shift+P in VS Code, terminal commands in file managers)
- Open, close, toggle, or rearrange UI panels, sidebars, toolbars, and status bars \
  when the resulting layout state is detectable (e.g., sidebar visibility stored in config)

Avoid tasks that:
- Are purely data-entry (typing long text, filling spreadsheet cells)
- Only read information without changing any state
- Produce purely transient visual effects that leave no trace once the window is \
  closed (e.g., hovering over a tooltip, scrolling without changing scroll position settings)

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
2. The task MUST produce a verifiable state change — something that can be confirmed by \
   inspecting application state, document properties, files on disk, or system configuration.
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
        f"Keep tasks focused — each should explore one aspect of the software and "
        f"produce a clear, verifiable outcome.\n\n"
        f"{memory_block}"
        f"## Reference examples\n{ref}\n\n"
        f"## Setup functions\n{_fmt_funcs(catalog.setup_functions)}\n\n"
        f"## Getter functions\n{_fmt_funcs(catalog.getter_functions)}\n\n"
        f"## Metric functions\n{_fmt_funcs(catalog.metric_functions)}\n\n"
        f"Generate {num_to_generate} new, diverse task example(s) for \"{domain_info.name}\".\n"
        f"Each task should target a different feature or setting of the application, "
        f"producing a distinct, observable state change that can be verified.\n"
        f"Return a JSON array of task objects."
    )
    messages = [
        {"role": "system", "content": TASK_GEN_SYSTEM},
        {"role": "user", "content": user},
    ]
    
    # breakpoint()
    
    logger.info(f"Generating {num_to_generate} task example(s) for '{domain_info.name}' ...")

    raw = call_llm_with_single_response(messages=messages, llm_config=llm_config, max_tokens=8000, temperature=0.7)

    logger.info(f"LLM response: {len(raw)} chars")

    parsed = parse_json_response(raw)
    if parsed is None:
        logger.error("Failed to parse LLM response as JSON")
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    for ex in parsed:
        if not ex.get("id"):
            ex["id"] = str(uuid.uuid4())
    logger.info(f"Generated {len(parsed)} example(s)")
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
    logger.info(f"Generating verifier for: '{task_instruction[:80]}' ...")
    raw = call_llm_with_single_response(messages=messages, llm_config=llm_config, max_tokens=8000, temperature=0.7)
    parsed = parse_json_response(raw)
    if parsed is None:
        logger.error("Failed to parse verifier response")
        return {"postconfig": [], "eval": ""}
    ev = {"postconfig": parsed.get("postconfig", []), "eval": parsed.get("eval", "")}
    logger.info(f"Verifier eval: {ev['eval'][:120]}")
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
        logger.info(f"Synthesizing for domain: {domain}")
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
                memory.record(
                    example=ex, domain=domain,
                    code_result={"score": 0.0, "error": f"script_validation: {'; '.join(vr.errors)}"},
                    executable=True,
                )
            else:
                logger.warning(f"SCRIPT VALIDATION FAILED for {ex.get('id', '?')}: {'; '.join(vr.errors)} – skipping")
                # Record into memory as not executable (no scores)
                if memory is not None:
                    memory.record(
                        example=ex, domain=domain,
                        code_result={"score": 0.0, "error": f"script_validation: {'; '.join(vr.errors)}"},
                        executable=False,
                    )
        logger.info(f"Domain '{domain}': {len(valid_examples)}/{len(examples)} examples passed script validation")

        # Persist only validated examples
        domain_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        for ex in valid_examples:
            path = os.path.join(domain_dir, f"{ex['id']}.json")
            with open(path, "w") as f:
                json.dump(ex, f, indent=2)
            logger.info(f"Saved {path}")

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
    logger.info(f"Manifest with {len(all_examples)} examples saved to {manifest_path}")
    return all_examples


# ═══════════════════════════════════════════════════════════════════════════
# Module 4: API Client & Parallel Verification Workers
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# HTTP helpers – thin wrappers around the OSGym API server (main.py)
# ---------------------------------------------------------------------------

def _api_reset(server_url: str, task_config: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    """POST /reset — allocate a VM and reset with the given task config."""
    resp = http_requests.post(
        f"{server_url}/reset",
        json={"task_config": task_config, "timeout": timeout},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "problem": str, "vm_id": int}


def _api_step(server_url: str, action: str, vm_id: int) -> Dict[str, Any]:
    """POST /step — send an action to the VM."""
    resp = http_requests.post(
        f"{server_url}/step",
        json={"action": action, "vm_id": vm_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "is_finish": bool, "reward": float}


def _api_evaluate(server_url: str, vm_id: int) -> Dict[str, Any]:
    """POST /evaluate — evaluate the VM."""
    resp = http_requests.post(
        f"{server_url}/evaluate",
        json={"vm_id": vm_id},  
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()  # {"reward": float}


def _api_screenshot(server_url: str, vm_id: int) -> Dict[str, Any]:
    """GET /screenshot — get current screenshot from the VM."""
    resp = http_requests.get(
        f"{server_url}/screenshot",
        params={"vmId": vm_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()  # {"screenshot": b64, "vm_id": int}


def _api_shutdown(server_url: str, vm_id: int) -> None:
    """POST /shutdown — release the VM."""
    try:
        http_requests.post(
            f"{server_url}/shutdown",
            json={"vm_id": vm_id},
            timeout=30,
        )
    except Exception as e:
        logger.warning(f"Failed to shutdown VM {vm_id}: {e}")


# ---------------------------------------------------------------------------
# Task execution via the API
# ---------------------------------------------------------------------------

CODE_EXEC_SYSTEM = """\
You are a desktop automation agent that solves tasks by generating Python code \
executed directly on an Ubuntu VM.

## Execution environment
Your code is run as:
  python -c "import pyautogui; import time; pyautogui.FAILSAFE = False; <YOUR CODE>"
This means:
- `pyautogui` and `time` are already imported — do NOT re-import them.
- Your snippet is a single inline command string. Use semicolons or exec() for \
  multi-statement logic. For complex logic you may write: \
  exec("import os\\nresult = os.popen('ls').read()\\nprint(result)")
- For modules beyond pyautogui/time (subprocess, os, json, shutil, etc.) you \
  MUST import them yourself within the snippet.
- The working directory is the VM user's home (~).

## Capabilities
You can accomplish tasks through multiple approaches:
- **GUI automation**: pyautogui for clicks, keyboard shortcuts, typing, scrolling.
- **Shell commands**: subprocess.run() or os.popen() for CLI operations — \
  editing config files, installing packages, moving files, using xdotool, \
  gsettings, dconf, sed, etc.
- **Direct file manipulation**: Read/write config files, JSON preferences, \
  XML documents, databases with Python's standard library.
- **Hybrid**: Combine GUI actions (to open an app or navigate menus) with \
  direct file/command operations (to set the actual state).

Choose the most reliable approach for each task. Direct file or command-line \
manipulation is often more reliable than GUI clicking for tasks that change \
application settings or file contents.

## Evaluation
After you finish, the system evaluates whether the task was completed by \
running a verifier. The verifier uses getter functions to inspect the VM state:
- `get_vm_file(env, config={'path': ...})` — reads a file from the VM
- `get_vm_command_line(env, config={'command': ...})` — runs a shell command \
  and checks its output
- `get_accessibility_tree(env)` — inspects the UI accessibility tree
- `get_info_from_website(env, config={...})` — extracts data from a web page
- Various app-specific getters for Chrome preferences, VLC config, VS Code \
  settings, GIMP config, LibreOffice documents, etc.

The verifier then passes the getter output to a metric function (e.g. \
`check_json`, `exact_match`, `check_accessibility_tree`) to compare against \
expected values. Understanding what the verifier checks helps you know exactly \
what state to produce.

## Response format
You MUST wrap your code in a markdown ```python code fence. Your response \
should contain exactly one ```python ... ``` block. The code inside the fence \
will be extracted and directly inserted to replace <YOUR CODE> in the execution \
template above. For multi-line logic, use exec(\"\"\"...\"\"\") inside the fence.

Example response:
```python
exec(\"\"\"import os
os.makedirs('/home/user/test', exist_ok=True)
\"\"\")
```

You may include brief reasoning before the code fence, but the code fence is \
mandatory and must contain the complete, self-contained code to execute."""


def _build_code_task_context(example: Dict[str, Any]) -> str:
    """Build the task-specific context block for the code execution agent."""
    parts = [f"Task: {example['instruction']}"]

    evaluator = example.get("evaluator", {})
    eval_expr = evaluator.get("eval", "")
    if eval_expr:
        parts.append(
            f"\nVerifier expression (this is how your result will be checked):\n"
            f"  {eval_expr}\n"
            f"Make sure your actions produce the exact state this verifier expects."
        )

    return "\n".join(parts)


def _extract_code_from_response(raw: str) -> str:
    """Extract Python code from the ```python ... ``` fence in the LLM response."""
    match = re.search(r'```python\s*\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try generic ``` fence
    match = re.search(r'```\s*\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fence found — return stripped raw as last resort
    logger.warning("No ```python fence found in LLM response, using raw output")
    return raw.strip()


def run_code_task(
    server_url: str,
    example: Dict[str, Any],
    max_steps: int,
    sleep_after_execution: float,
    llm_config: Dict[str, Any],
    result_dir: str,
) -> Dict[str, Any]:
    """Execute one example via a single LLM-generated code snippet through the API."""
    proc = current_process().name
    task_context = _build_code_task_context(example)

    reset_data = _api_reset(server_url, example)
    vm_id = reset_data["vm_id"]
    reward = 0

    try:
        screenshot = base64.b64decode(reset_data["screenshot"])

        # Build prompt with initial screenshot
        messages = [
            {"role": "system", "content": CODE_EXEC_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_screenshot(screenshot)}},
                {"type": "text", "text": task_context},
            ]},
        ]

        # Single LLM call to generate the code
        raw = call_llm_with_single_response(
            messages=messages, llm_config=llm_config,
            max_tokens=8000, temperature=0.7,
        )
        # breakpoint()
        code = _extract_code_from_response(raw)

        ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        logger.info(f"[{proc}][code] generated: {code}")

        # Execute the code on the VM
        step_data = _api_step(server_url, code, vm_id)
        time.sleep(sleep_after_execution)

        screenshot_bytes = base64.b64decode(step_data["screenshot"])
        with open(os.path.join(result_dir, f"step_0_{ts}.png"), "wb") as fp:
            fp.write(screenshot_bytes)

        # Evaluate
        time.sleep(5)
        eval_data = _api_evaluate(server_url, vm_id)
        reward = eval_data["reward"]
        logger.info(f"[{proc}][code] score={reward:.2f} for {example['id']}")

        # Save trajectory
        trajectory = [{"step": 0, "timestamp": ts, "code": code}]
        with open(os.path.join(result_dir, "trajectory.json"), "w") as fp:
            json.dump(trajectory, fp, indent=2)
        with open(os.path.join(result_dir, "result.txt"), "w") as fp:
            fp.write(f"{reward}\n")
        return {"id": example["id"], "mode": "code", "score": reward, "steps": 1}

    finally:
        _api_shutdown(server_url, vm_id)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker(task_queue, args: argparse.Namespace, shared_results: list):
    """Worker that pulls examples from the queue and runs code execution."""
    proc = current_process().name
    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}

    while True:
        try:
            example = task_queue.get(timeout=5)
        except Exception:
            break  # queue empty / timeout

        try:
            result_dir = os.path.join(
                args.output_dir, example.get("_domain", "unknown"),
                "trajectories", example["id"], "code",
            )
            os.makedirs(result_dir, exist_ok=True)

            res = run_code_task(
                args.server_url, example,
                args.max_steps, args.sleep_after_execution,
                llm_config, result_dir,
            )
            shared_results.append(res)

        except KeyboardInterrupt:
            logger.warning(f"[{proc}] KeyboardInterrupt")
            break
        except Exception as e:
            logger.error(f"[{proc}] Error on code/{example['id']}: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"[{proc}] Worker finished")


# ═══════════════════════════════════════════════════════════════════════════
# Verify stage: execution & evaluation
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
                logger.warning(f"Example file not found: {path}")
    logger.info(f"Loaded {len(examples)} synthetic examples from {manifest_path}")
    return examples


# ---------------------------------------------------------------------------
# Result processing (shared by both verify modes)
# ---------------------------------------------------------------------------

def _process_verify_results(
    results: List[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    args: argparse.Namespace,
    memory: Optional[SynthesisMemory],
) -> List[Dict[str, Any]]:
    """Aggregate results, update memory, write solvable examples. Returns results."""

    # Persist all raw verification results
    agg_path = os.path.join(args.output_dir, "verification_results.json")
    with open(agg_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    code_scores = [r["score"] for r in results if "error" not in r]
    avg = sum(code_scores) / max(len(code_scores), 1)
    logger.info(f"Code avg={avg:.3f} ({len(code_scores)} tasks)")

    # ---- Classify results into solvable / unsolvable / errored ----
    by_id: Dict[str, Any] = {}
    for r in results:
        by_id[r["id"]] = r

    solvable_dir = os.path.join(args.output_dir, "solvable")
    os.makedirs(solvable_dir, exist_ok=True)

    solvable_ids = []
    unsolvable_ids = []
    errored_ids = []   # execution crashed — leave solvable=None so they retry
    for eid, code_r in by_id.items():
        if "error" in code_r:
            # Execution itself failed (network, VM crash, etc.) — not a
            # definitive verdict on solvability, should be retried.
            errored_ids.append(eid)
            logger.info(f"ERRORED {eid}: {code_r['error']}")
        elif code_r.get("score", 0) > 0:
            solvable_ids.append(eid)
        else:
            unsolvable_ids.append(eid)
            logger.info(f"UNSOLVABLE {eid}: code_score={code_r.get('score', 'missing')}")

    # Record outcomes into synthesis memory
    if memory is not None:
        ex_by_id = {ex["id"]: ex for ex in examples}
        for eid, code_r in by_id.items():
            ex = ex_by_id.get(eid)
            if ex is None:
                continue
            if eid in errored_ids:
                # Keep solvable=None so the filter will pick them up next run
                solvable = None
            else:
                solvable = eid in solvable_ids
            memory.record(
                example=ex,
                domain=ex.get("_domain", "unknown"),
                code_result=code_r,
                executable=True,
                solvable=solvable,
            )
        memory.save()

    # Copy solvable examples to solvable/ directory
    solvable_manifest = {}
    for ex in examples:
        if ex["id"] not in solvable_ids:
            continue
        domain = ex.get("_domain", "unknown")
        domain_dir = os.path.join(solvable_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        path = os.path.join(domain_dir, f"{ex['id']}.json")
        with open(path, "w") as f:
            json.dump(ex, f, indent=2)
        solvable_manifest.setdefault(domain, []).append(ex["id"])

    # Write solvable manifest
    solvable_manifest_path = os.path.join(solvable_dir, "manifest.json")
    with open(solvable_manifest_path, "w") as f:
        json.dump(solvable_manifest, f, indent=2)

    logger.info(f"Verification complete: {len(solvable_ids)} solvable, {len(unsolvable_ids)} unsolvable, {len(errored_ids)} errored (will retry) out of {len(by_id)} total")
    logger.info(f"Solvable examples saved to {solvable_dir}")
    logger.info(f"All raw results saved to {agg_path}")
    return results


# ---------------------------------------------------------------------------
# Parallel verification (--verify-mode run)
# ---------------------------------------------------------------------------

def _run_verify_parallel(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Multi-process verification: workers pull examples from a queue and run code execution."""
    global _processes

    total_workers = args.num_workers
    logger.info(f"Verification (parallel): {len(examples)} examples across {total_workers} workers")

    manager = Manager()
    shared_results = manager.list()
    task_queue = manager.Queue()

    for ex in examples:
        task_queue.put(ex)

    _processes = []
    for pidx in range(total_workers):
        p = Process(
            target=worker,
            args=(task_queue, args, shared_results),
            name=f"SynthWorker-{pidx}",
            daemon=True,
        )
        p.start()
        _processes.append(p)
        logger.info(f"Started {p.name} (PID {p.pid})")

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

    return list(shared_results)


# ---------------------------------------------------------------------------
# Sequential verification (--verify-mode debug)
# ---------------------------------------------------------------------------

def _run_verify_sequential(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sequential verification in the main process (debugger-friendly)."""
    logger.info(f"Verification (sequential/debug): {len(examples)} examples")

    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    results: List[Dict[str, Any]] = []

    for ex_idx, example in enumerate(examples):
        logger.info(f"[{ex_idx + 1}/{len(examples)}] Running code for {example['id']} ({example.get('instruction', '')})")
        result_dir = os.path.join(
            args.output_dir, example.get("_domain", "unknown"),
            "trajectories", example["id"], "code",
        )
        os.makedirs(result_dir, exist_ok=True)

        try:
            res = run_code_task(
                args.server_url, example,
                args.max_steps, args.sleep_after_execution,
                llm_config, result_dir,
            )
            results.append(res)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — stopping verification")
            return results
        except Exception as e:
            logger.error(f"Error on code/{example['id']}: {e}")
            logger.error(traceback.format_exc())
            results.append({
                "id": example["id"], "mode": "code",
                "score": 0.0, "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# Verify dispatcher
# ---------------------------------------------------------------------------

def run_verify(args: argparse.Namespace, examples: Optional[List[Dict[str, Any]]] = None,
               memory: Optional[SynthesisMemory] = None) -> List[Dict[str, Any]]:
    """Execute synthetic examples on VMs via the API server and evaluate.

    Dispatches to parallel (--verify-mode run) or sequential
    (--verify-mode debug) execution based on the CLI flag.
    """
    if examples is None:
        examples = _load_synthetic_examples(args)
    if not examples:
        logger.error("No examples to verify")
        return []

    # Filter: only verify examples whose solvable status is still unknown
    if memory is not None:
        already_tested = {
            e["id"] for e in memory.entries
            if e.get("solvable") is not None
        }
        before = len(examples)
        examples = [ex for ex in examples if ex["id"] not in already_tested]
        if before != len(examples):
            logger.info(f"Skipped {before - len(examples)} already-verified examples ({len(examples)} remaining)")
    if not examples:
        logger.info("All examples already verified — nothing to do")
        return []

    # Dispatch based on verify mode
    verify_mode = getattr(args, "verify_mode", "run")
    if verify_mode == "debug":
        results = _run_verify_sequential(args, examples)
    else:
        results = _run_verify_parallel(args, examples)

    return _process_verify_results(results, examples, args, memory)


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
    logger.info(f"Signal {signum} received - shutting down")
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

    # API server
    p.add_argument("--server-url", type=str, default="http://localhost:20000",
                    help="URL of the OSGym API server (main.py)")

    # Execution
    p.add_argument("--sleep-after-execution", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=15,
                    help="Max steps for code execution; also guides task complexity during synthesis")

    # Parallelism / verification mode
    p.add_argument("--num-workers", type=int, default=2,
                    help="Number of parallel worker processes for verification")
    p.add_argument("--verify-mode", choices=["run", "debug"], default="run",
                    help="'run' = multi-process workers (production), "
                         "'debug' = sequential in main process (debugger-friendly)")

    # Output
    p.add_argument("--output-dir", type=str, default="synthetic_evaluation_examples")
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
