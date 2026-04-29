"""Task synthesis: LLM-driven generation, static validation, batched loop.

Responsibilities:
  * Discover domains and load reference examples from
    ``synthetic_demos/`` (hand-curated, download-free demo tasks).
  * Catalog the setup / getter / metric function libraries the LLM is allowed
    to use.
  * Generate new task examples + verifiers via LLM calls.
  * Statically validate generated scripts before they reach the VM.
  * Drive the batched synthesis loop (``run_synthesize``), including
    vector-DB dedup and writing validated examples to disk.
"""

from __future__ import annotations

import argparse
import ast
import glob
import inspect
import json
import logging
import os
import random
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from mm_agents.utils.call_llm import call_llm_with_single_response
from mm_agents.utils.utils import parse_json_response

from desktop_env.evaluators.schema import (
    _first_doc_line,
    get_schema,
    required_config_keys,
    required_rules_keys,
)

from .prompts import TASK_GEN_SYSTEM
from .shared_memory import SynthesisMemory, VectorDedupStore

logger = logging.getLogger("desktopenv.synthesis.task_creator")

# Hand-curated demo tasks that the LLM reads as ground-truth references in the
# TASK_GEN prompt. Each ``<domain>/`` subdirectory holds download-free example
# tasks that statically pass validation and exercise the most common setup
# helpers / getters / metrics for that domain. This directory is also used for
# domain discovery — the synthesis pipeline targets exactly the domains that
# have a folder here.
EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "synthetic_demos",
)


# ═══════════════════════════════════════════════════════════════════════════
# Domain ↔ getter / metric module maps
#
# Goal: when the prompt for a single-app domain (e.g. ``gimp``) is built, we
# only feed the LLM the getters/metrics that actually apply. Generic helpers
# stay visible everywhere; cross-app modules (chrome's getters, slides metrics)
# only show up for the matching domain. ``multi_apps`` is the escape hatch —
# it sees the full catalog because tasks there span apps.
#
# Module names below match ``obj.__module__.rsplit('.', 1)[-1]`` for each
# function exported from ``desktop_env.evaluators.{getters,metrics}``. Two
# special leaves carry universal helpers:
#   - ``"metrics"``: the package __init__ itself (e.g. ``infeasible``).
#   - ``"getters"``: reserved for the same role on the getters side.
# ═══════════════════════════════════════════════════════════════════════════


# Cross-domain helpers — every single-app domain sees these.
SHARED_GETTER_MODULES: Tuple[str, ...] = ("file", "general", "misc", "replay")
SHARED_METRIC_MODULES: Tuple[str, ...] = ("general", "pdf", "others")

# Package-level leaves that always remain in scope. Functions defined directly
# in ``getters/__init__.py`` or ``metrics/__init__.py`` (such as the
# ``infeasible`` sentinel metric) carry these as their leaf module name.
_PACKAGE_ROOT_LEAVES: frozenset = frozenset({"getters", "metrics"})

# Single-app domains → their domain-specific getter modules. Empty tuple means
# the domain has no domain-specific getter file (it relies entirely on shared
# modules above). ``multi_apps`` is intentionally absent — it bypasses the
# filter via the ``"*"`` wildcard handled in ``_filter_for_domain``.
DOMAIN_GETTER_MODULES: Dict[str, Tuple[str, ...]] = {
    "chrome":              ("chrome",),
    "gimp":                ("gimp",),
    "libreoffice_calc":    ("calc",),
    "libreoffice_impress": ("impress",),
    "libreoffice_writer":  (),
    "os":                  ("info",),
    "thunderbird":         (),
    "vlc":                 ("vlc",),
    "vs_code":             ("vscode",),
}

# Single-app domains → their domain-specific metric modules. The libreoffice
# leaf carries cross-suite helpers (``check_libre_locale``) so all three
# libreoffice domains include it alongside their own.
DOMAIN_METRIC_MODULES: Dict[str, Tuple[str, ...]] = {
    "chrome":              ("chrome",),
    "gimp":                ("gimp",),
    "libreoffice_calc":    ("libreoffice", "table"),
    "libreoffice_impress": ("libreoffice", "slides"),
    "libreoffice_writer":  ("libreoffice", "docs"),
    "os":                  ("basic_os",),
    "thunderbird":         ("thunderbird",),
    "vlc":                 ("vlc",),
    "vs_code":             ("vscode",),
}

# Sentinel domain that disables filtering (full catalog rendered for the LLM).
_FULL_CATALOG_DOMAIN = "multi_apps"

# Maximum LLM attempts per ``generate_task_examples`` call. The endpoint can
# blip (transport errors, malformed JSON, lists of strings instead of dicts),
# and a single batch failure shouldn't stall the whole synthesis loop — the
# outer ``empty_streak`` already kicks in if every attempt yields nothing.
_LLM_GENERATION_RETRIES = 3


def _module_leaf(obj: Any) -> str:
    """Return the last dotted component of ``obj.__module__``.

    Used to bucket each cataloged function under its source file name (e.g.
    ``desktop_env.evaluators.getters.chrome`` → ``"chrome"``) so the prompt
    builder can filter by domain without re-importing the underlying modules.
    """
    return getattr(obj, "__module__", "").rsplit(".", 1)[-1]


def _filter_for_domain(
    funcs: List[Dict[str, Any]],
    domain: str,
    domain_modules: Dict[str, Tuple[str, ...]],
    shared_modules: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """Return the subset of ``funcs`` reachable from ``domain``.

    ``multi_apps`` (the cross-app domain) returns the full list unchanged —
    those tasks legitimately combine functions from many modules. Every other
    domain restricts to ``shared_modules`` plus its own
    ``domain_modules[domain]`` entries, with package-root leaves always
    included so universal sentinels (``infeasible``) stay reachable. An
    unknown domain falls back to shared-only — safer than exposing the whole
    surface area when we don't know what's appropriate.
    """
    if domain == _FULL_CATALOG_DOMAIN:
        return list(funcs)
    allowed = set(shared_modules) | _PACKAGE_ROOT_LEAVES | set(domain_modules.get(domain, ()))
    return [f for f in funcs if f.get("module") in allowed]


# ═══════════════════════════════════════════════════════════════════════════
# Domain discovery and function cataloging
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DomainInfo:
    name: str
    example_files: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FunctionCatalog:
    """Static catalog of the setup / getter / metric functions the LLM may use.

    Each entry is a dict with:
      - ``name``      str: function name as referenced from synthesized scripts
      - ``signature`` str: pretty signature for prompt context, e.g. "f(env, config)"
      - ``doc``       str: first line of the docstring
      - ``_sig``      Optional[inspect.Signature]: the actual Signature used by
        ``validate_example_scripts`` to bind argument lists. ``self`` has
        already been stripped on setup-controller methods.
    """
    setup_functions: List[Dict[str, Any]] = field(default_factory=list)
    getter_functions: List[Dict[str, Any]] = field(default_factory=list)
    metric_functions: List[Dict[str, Any]] = field(default_factory=list)

    def index_by_name(self) -> Dict[str, Dict[str, Any]]:
        """Return a flat name → entry map across all three function groups."""
        idx: Dict[str, Dict[str, Any]] = {}
        for f in self.setup_functions:
            idx[f["name"]] = f
        for f in self.getter_functions:
            idx[f["name"]] = f
        for f in self.metric_functions:
            idx[f["name"]] = f
        return idx


def _stamp_synthesized_examples(
    examples: List[Dict[str, Any]], domain: str
) -> None:
    """Assign post-synthesis identity fields in-place, exactly once per example.

    The merged ``TASK_GEN_SYSTEM`` prompt no longer asks the LLM to emit
    ``id`` or ``source`` (the output schema only lists ``snapshot`` /
    ``instruction`` / ``config`` / ``related_apps`` / ``evaluator``), so we
    deterministically stamp them here. Any pre-existing ``id`` from the LLM
    is OVERWRITTEN with a fresh UUID4 — the in-prompt reference examples
    carry real UUIDs the model could copy, and a duplicate ``id`` would
    corrupt the manifest, the synthesis memory, and the on-disk
    ``<domain>/<id>.json`` layout. ``source`` is fixed at ``"synthetic"``
    to distinguish these from curated examples, and ``_domain`` is the
    in-process tag the verify stage and memory writer key on.

    The function mutates ``examples`` in place and returns ``None``; it is
    called exactly once per batch (right after the LLM response is parsed)
    so downstream code can rely on every accepted example having a unique,
    canonical ``id``. Non-dict entries are skipped defensively so a malformed
    item from upstream cannot crash the whole batch — callers should already
    have filtered to dicts before reaching here.
    """
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        ex["id"] = str(uuid.uuid4())
        ex["source"] = "synthetic"
        ex["_domain"] = domain


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
    """Load the curated demo tasks for ``domain`` from ``synthetic_demos/``.

    These hand-vetted examples are the primary reference shown to the LLM in
    the TASK_GEN prompt. ``max_examples`` randomly samples up to that many
    files from the demo set; ``0`` (the default) loads them all.
    """
    domain_dir = os.path.join(EXAMPLES_DIR, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
    info = DomainInfo(name=domain)
    json_files = sorted(glob.glob(os.path.join(domain_dir, "*.json")))
    info.example_files = json_files
    to_load = json_files if max_examples <= 0 else random.sample(json_files, max_examples)
    for fp in to_load:
        with open(fp, "r") as f:
            info.examples.append(json.load(f))
    logger.info(f"Loaded {len(info.examples)}/{len(json_files)} examples for domain '{domain}'")
    return info


def _func_info(obj, name: str, *, drop_self: bool = False) -> Dict[str, Any]:
    """Build a catalog entry for one function.

    ``drop_self`` removes the leading ``self`` parameter from the live
    Signature so that ``Signature.bind(...)`` can be applied to LLM-emitted
    calls that look like ``_download_setup(files=...)`` (i.e. as the
    SetupController binds them — without an explicit ``self`` argument).

    Each entry also carries an evaluator ``schema`` (see
    ``desktop_env.evaluators.schema``) — populated from the
    ``@evaluator`` decorator when present, otherwise from a Sphinx-ish
    docstring parser. The schema describes the dict keys the function
    reads from its ``config`` / ``rules`` / ``options`` arguments and
    the value it returns; ``_fmt_funcs`` renders this for the LLM and
    ``validate_example_scripts`` consumes it to flag missing required
    keys.
    """
    sig: Optional[inspect.Signature]
    try:
        sig = inspect.signature(obj)
        if drop_self:
            params = [
                p for p_name, p in sig.parameters.items() if p_name != "self"
            ]
            sig = sig.replace(parameters=params)
        sig_str = str(sig)
    except (ValueError, TypeError):
        sig = None
        sig_str = "(...)"
    schema = get_schema(obj)
    # Prefer schema-declared summary; fall back to _first_doc_line so a
    # docstring that opens with a "Config:" header doesn't surface the header
    # itself as the summary line.
    summary = schema.get("summary") or _first_doc_line(inspect.getdoc(obj))
    return {
        "name": name,
        "signature": f"{name}{sig_str}",
        "doc": summary,
        "role": schema.get("role", ""),
        "schema": schema,
        "module": _module_leaf(obj),
        "_sig": sig,
    }


def catalog_functions() -> FunctionCatalog:
    cat = FunctionCatalog()

    from desktop_env.controllers.setup import SetupController
    from desktop_env.evaluators.schema import is_setup_action
    # Enumerate via the ``@setup_action`` marker so the catalog stays in lock-step
    # with the dispatcher in ``SetupController.setup`` — undecorated stubs (e.g.
    # NotImplementedError placeholders) are correctly hidden from the LLM and
    # flagged as unknown by the validator.
    for name in sorted(dir(SetupController)):
        obj = getattr(SetupController, name, None)
        if callable(obj) and is_setup_action(obj):
            # SetupController.setup() resolves these via local_namespace
            # (binding self implicitly), so the LLM never passes self —
            # drop it from the cataloged signature for binding checks.
            cat.setup_functions.append(_func_info(obj, name, drop_self=True))

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

    # Surface decorator coverage so it's obvious when @evaluator is missing or
    # when a decorated function takes config/rules but declares no key schema
    # (the validator's required-key check then has nothing to enforce, except
    # what the small fallback registries cover).
    def _coverage_stats(funcs: List[Dict[str, Any]], dict_arg: str) -> Tuple[int, int, int]:
        decorated = sum(1 for f in funcs if (f.get("schema") or {}).get("source") == "decorator")
        empty_schema = 0
        for f in funcs:
            sig = f.get("_sig")
            if sig is None or dict_arg not in sig.parameters:
                continue
            schema = f.get("schema") or {}
            if not schema.get(dict_arg):
                empty_schema += 1
        return len(funcs), decorated, empty_schema

    g_total, g_decorated, g_empty = _coverage_stats(cat.getter_functions, "config")
    m_total, m_decorated, m_empty = _coverage_stats(cat.metric_functions, "rules")
    s_total = len(cat.setup_functions)
    s_decorated = sum(
        1 for f in cat.setup_functions
        if (f.get("schema") or {}).get("source") == "decorator"
    )
    logger.info(
        f"Cataloged {s_total} setup ({s_decorated} decorated), "
        f"{g_total} getter ({g_decorated} decorated, {g_empty} take config but "
        f"declare no schema keys), "
        f"{m_total} metric ({m_decorated} decorated, {m_empty} take rules but "
        f"declare no schema keys)"
    )
    return cat


# ═══════════════════════════════════════════════════════════════════════════
# LLM-driven example + verifier generation
# ═══════════════════════════════════════════════════════════════════════════


def _fmt_funcs(funcs: List[Dict[str, Any]]) -> str:
    """Render catalog entries for the LLM prompt, including schema lines.

    Each function gets a header (``- name(sig): one-line summary``) and,
    when a schema is available, indented ``config.<key>: ...`` /
    ``rules.<key>: ...`` / ``options.<key>: ...`` lines plus an optional
    ``returns: ...`` line. Optional keys are surfaced with a trailing
    ``"?"`` so the LLM can tell required from optional.
    """
    lines: List[str] = []
    for f in funcs:
        header = f"- {f['signature']}: {f['doc']}"
        schema = f.get("schema") or {}
        for kind in ("config", "rules", "options", "args"):
            entries: Dict[str, str] = schema.get(kind, {}) or {}
            if not entries:
                continue
            for key, desc in entries.items():
                desc_one_line = (desc or "").splitlines()[0] if desc else ""
                header += f"\n    {kind}.{key}: {desc_one_line}"
        returns_desc = (schema.get("returns") or "").strip()
        if returns_desc:
            header += f"\n    returns: {returns_desc}"
        lines.append(header)
    return "\n".join(lines)


def generate_task_examples(
    domain_info: DomainInfo,
    catalog: FunctionCatalog,
    llm_config: Dict[str, Any],
    num_to_generate: int = 1,
    max_steps: int = 15,
    memory: Optional[SynthesisMemory] = None,
    memory_block: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate full task examples — instruction + setup config + evaluator —
    in a single LLM call.

    The merged ``TASK_GEN_SYSTEM`` prompt covers task selection AND verifier
    construction, so the model returns each example with its ``evaluator``
    (``postconfig`` + ``eval``) already populated. There is no second pass:
    examples without an ``eval`` field are rejected by the static validator
    later in ``run_synthesize``.

    ``memory_block`` is the prior-experience text injected into the user
    prompt. Callers may pass a pre-formatted block (computed under a lock in
    parallel mode so the read happens atomically with respect to concurrent
    ``memory.record`` calls); otherwise it is derived from ``memory`` here.
    """
    # Curated demos (loaded from ``synthetic_demos/<domain>/``) are the LLM's
    # primary source of correct call patterns — they showcase how to pair
    # file-creation setup with `_open_setup`, how to populate
    # `get_vm_file(env, config={'path':..., 'dest':...})` with both required
    # keys, and how to emit rules dicts that match the metric's schema.
    demos = domain_info.examples[:5]
    ref = json.dumps(demos, indent=2)

    # Build memory context (empty string if no prior experience). If the
    # caller supplied ``memory_block`` directly, use it verbatim — that path
    # is what parallel-mode workers take, so the LLM call below runs outside
    # the shared lock.
    if memory_block is None:
        memory_block = memory.format_for_prompt(domain_info.name) if memory is not None else ""
    if memory_block:
        memory_block = f"\n{memory_block}\n"

    # Restrict the getter / metric surface area shown to the LLM to what
    # actually applies to this domain. multi_apps gets the full catalog;
    # every other domain is limited to its own modules plus the shared
    # helpers (file/general/misc/replay for getters; general/pdf/others for
    # metrics). Setup functions are not filtered — they're domain-agnostic
    # OS plumbing and any task may legitimately call any of them.
    domain_getters = _filter_for_domain(
        catalog.getter_functions, domain_info.name,
        DOMAIN_GETTER_MODULES, SHARED_GETTER_MODULES,
    )
    domain_metrics = _filter_for_domain(
        catalog.metric_functions, domain_info.name,
        DOMAIN_METRIC_MODULES, SHARED_METRIC_MODULES,
    )
    logger.info(
        f"Domain '{domain_info.name}' prompt scope: "
        f"{len(domain_getters)}/{len(catalog.getter_functions)} getters, "
        f"{len(domain_metrics)}/{len(catalog.metric_functions)} metrics"
    )

    user = (
        f"Domain: {domain_info.name}\n"
        f"## Complexity budget\n"
        f"Each task must be completable by a GUI agent in at most **{max_steps} steps** "
        f"(each step = one mouse click, keystroke, or typed string). "
        f"Keep tasks focused — each should explore one aspect of the software and "
        f"produce a clear, verifiable outcome.\n\n"
        f"{memory_block}"
        f"## Reference examples (each shows a complete task with its evaluator — copy these patterns)\n{ref}\n\n"
        f"## Setup functions\n{_fmt_funcs(catalog.setup_functions)}\n\n"
        f"## Getter functions\n{_fmt_funcs(domain_getters)}\n\n"
        f"## Metric functions\n{_fmt_funcs(domain_metrics)}\n\n"
        f"Generate {num_to_generate} new, diverse task example(s) for "
        f"\"{domain_info.name}\".\n"
        f"Each task must:\n"
        f"  - target a different feature or setting of the application,\n"
        f"  - produce a distinct, observable state change, AND\n"
        f"  - ship with a fully-populated `evaluator` object containing both "
        f"`postconfig` (list of setup-call strings) and a non-empty `eval` "
        f"expression that follows the verifier-strength rubric in the system "
        f"prompt.\n"
        f"Return a JSON array of task objects (no markdown fences)."
    )
    # breakpoint()
    messages = [
        {"role": "system", "content": TASK_GEN_SYSTEM},
        {"role": "user", "content": user},
    ]

    logger.info(
        f"Generating {num_to_generate} task+verifier example(s) for "
        f"'{domain_info.name}' ..."
    )

    # Retry loop: an LLM hiccup (transport error, malformed JSON, a list of
    # bare strings instead of task dicts) shouldn't sink the whole batch.
    # We try up to ``_LLM_GENERATION_RETRIES`` times and return the first
    # attempt that yields at least one well-formed dict; any non-dict entries
    # in an otherwise valid response are dropped with a warning.
    last_failure = ""
    for attempt in range(1, _LLM_GENERATION_RETRIES + 1):
        # Randomize temperature per attempt so retries (and successive batches)
        # explore a wider distribution; this fights mode collapse on smaller
        # models that otherwise emit near-identical task lists across calls.
        temperature = random.uniform(0.6, 1.0)
        try:
            raw = call_llm_with_single_response(
                messages=messages, llm_config=llm_config,
                max_tokens=8000, temperature=temperature,
            )
        except Exception as e:
            last_failure = f"LLM call raised {type(e).__name__}: {e}"
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}' failed: {last_failure}"
            )
            continue

        if not isinstance(raw, str) or not raw.strip():
            last_failure = "LLM returned empty response"
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}': {last_failure}"
            )
            continue
        logger.info(f"LLM response (T={temperature:.2f}): {len(raw)} chars")

        try:
            parsed = parse_json_response(raw)
        except Exception as e:
            last_failure = f"JSON parser raised {type(e).__name__}: {e}"
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}': {last_failure}"
            )
            continue

        if parsed is None:
            last_failure = "LLM response did not contain parseable JSON"
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}': {last_failure}"
            )
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            last_failure = (
                f"LLM response is {type(parsed).__name__}, not a JSON array"
            )
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}': {last_failure}"
            )
            continue

        valid = [ex for ex in parsed if isinstance(ex, dict)]
        if len(valid) < len(parsed):
            logger.warning(
                f"Dropped {len(parsed) - len(valid)}/{len(parsed)} non-dict "
                f"entries from LLM response for '{domain_info.name}'"
            )
        if not valid:
            last_failure = "LLM response had no dict task objects"
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}': {last_failure}"
            )
            continue

        # Stamp identity fields (id / source / _domain) once, here. The LLM
        # is not asked to emit them, and any model-supplied id is overwritten
        # so reference-example UUID copying can't collide with the manifest.
        try:
            _stamp_synthesized_examples(valid, domain_info.name)
        except Exception as e:
            last_failure = f"identity stamping failed: {type(e).__name__}: {e}"
            logger.warning(
                f"Attempt {attempt}/{_LLM_GENERATION_RETRIES} for "
                f"'{domain_info.name}': {last_failure}"
            )
            continue

        missing_eval = sum(
            1 for ex in valid if not (ex.get("evaluator") or {}).get("eval")
        )
        if missing_eval:
            logger.warning(
                f"{missing_eval}/{len(valid)} generated example(s) lack an "
                f"`evaluator.eval` field — they will be rejected by static "
                f"validation. Check the system prompt if this recurs."
            )
        logger.info(
            f"Generated {len(valid)} example(s) for '{domain_info.name}' "
            f"on attempt {attempt}/{_LLM_GENERATION_RETRIES}"
        )
        return valid

    logger.error(
        f"Giving up on '{domain_info.name}' after "
        f"{_LLM_GENERATION_RETRIES} attempts; last failure: {last_failure}"
    )
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Static validation of generated scripts
# ═══════════════════════════════════════════════════════════════════════════


_EVAL_BUILTINS = {"float", "int", "str", "len", "bool", "abs", "max", "min"}

# Sentinel for AST-extracted argument values that are not literal-evaluable
# (e.g. a Name like ``env`` or a nested Call). These pass through Signature.bind
# as opaque objects so binding still validates arity and kwarg names without
# tripping on runtime values that don't exist statically.
_OPAQUE = object()

# Last-resort fallback registries for functions that don't yet declare a
# schema (no ``@evaluator`` decorator and no ``Config:``/``Rules:`` block in
# the docstring). The validator prefers schema-derived keys; this list is
# only consulted when the schema yields nothing.
_GETTER_FALLBACK_CONFIG_KEYS: Dict[str, List[str]] = {
    "get_vm_file": ["path"],
    "get_vm_command_line": ["command"],
    "get_vm_command_error": ["command"],
    "get_rule": ["rules"],
    "get_rule_relativeTime": ["rules"],
}
_METRIC_FALLBACK_RULES_KEYS: Dict[str, List[str]] = {
    "exact_match": ["expected"],
    "match_in_list": ["expected"],
    "is_in_list": ["expected"],
    "fuzzy_match": ["expected"],
}


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class _ParsedCall:
    """AST-extracted call form, with literal-evaluable values resolved."""
    func_name: str
    pos_args: List[Any]
    kwargs: Dict[str, Any]
    has_splat: bool  # *args / **kwargs prevent static binding


def _parse_call(node: ast.Call) -> _ParsedCall:
    """Extract a static call-form from an ``ast.Call`` node.

    Args / kwargs whose AST values aren't literal-evaluable (Names like
    ``env``, nested Calls, attribute lookups) are recorded as ``_OPAQUE``
    so ``Signature.bind`` still checks arity without requiring runtime
    values.
    """
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    else:
        func_name = "<expr>"

    pos_args: List[Any] = []
    has_splat = False
    for a in node.args:
        if isinstance(a, ast.Starred):
            has_splat = True
            continue
        try:
            pos_args.append(ast.literal_eval(a))
        except Exception:
            pos_args.append(_OPAQUE)

    kwargs: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:  # **kwargs splat
            has_splat = True
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            kwargs[kw.arg] = _OPAQUE

    return _ParsedCall(
        func_name=func_name,
        pos_args=pos_args,
        kwargs=kwargs,
        has_splat=has_splat,
    )


def _check_signature(call: _ParsedCall, sig: Optional[inspect.Signature], label: str) -> List[str]:
    """Bind the parsed call against the live Signature; report mismatches."""
    if sig is None:
        return []  # signature unavailable; best-effort skip
    if call.has_splat:
        return []  # *args/**kwargs erase static info
    try:
        bound = sig.bind(*call.pos_args, **call.kwargs)
        bound.apply_defaults()
    except TypeError as e:
        return [
            f"{label}: call to {call.func_name}(...) does not match the real "
            f"implementation's signature {call.func_name}{sig}: {e}"
        ]
    return []


def _resolve_dict_arg(
    call: _ParsedCall, kw_name: str, pos_index: int
) -> Optional[Dict[Any, Any]]:
    """Pull the literal dict bound to ``kw_name`` (kwarg) or ``pos_index`` (positional)."""
    if kw_name in call.kwargs:
        val = call.kwargs[kw_name]
    elif len(call.pos_args) > pos_index:
        val = call.pos_args[pos_index]
    else:
        return None
    if val is _OPAQUE or not isinstance(val, dict):
        return None
    return val


def _resolve_list_arg(
    call: _ParsedCall, kw_name: str, pos_index: int
) -> Optional[List[Any]]:
    """Pull the literal list bound to ``kw_name`` (kwarg) or ``pos_index`` (positional)."""
    if kw_name in call.kwargs:
        val = call.kwargs[kw_name]
    elif len(call.pos_args) > pos_index:
        val = call.pos_args[pos_index]
    else:
        return None
    if val is _OPAQUE or not isinstance(val, list):
        return None
    return val


# How long to wait when probing a `_download_setup` URL. Each task may carry
# multiple URLs and the validator runs once per generated example, so the
# timeout is intentionally short.
_DOWNLOAD_PROBE_TIMEOUT = 20.0
# Read at most this many bytes after the response opens — we only need to
# confirm the body is reachable, not pull the whole asset.
_DOWNLOAD_PROBE_BYTES = 4096


def _validate_download_urls(files: List[Any], label: str) -> List[str]:
    """Probe every URL that ``_download_setup`` would fetch.

    The TASK_GEN prompt asks the LLM not to use `_download_setup`; this is the
    enforcement when the model emits one anyway. Each entry in ``files`` is
    expected to be a ``{"url": ..., "path": ...}`` dict (matching the
    SetupController signature). For every literal-string URL we issue a
    streaming GET and read a small chunk so a fabricated or 404 link surfaces
    as a validation error before the example reaches the VM stage.

    Errors carry the ``download_url:`` category prefix so callers and memory
    summaries can distinguish network-fetch failures from other setup
    problems (e.g. ``missing_file_setup``).

    Non-literal entries (resolved to ``_OPAQUE`` by the AST parser) are left
    alone — we can't tell statically what URL would be passed.
    """
    errors: List[str] = []
    for i, f in enumerate(files):
        if f is _OPAQUE:
            errors.append(
                f"{label}: download_url: _download_setup files[{i}] is not a "
                f"literal dict — the synthesis pipeline cannot statically "
                f"verify the URL"
            )
            continue
        if not isinstance(f, dict):
            errors.append(
                f"{label}: download_url: _download_setup files[{i}] is not a "
                f"dict (got {type(f).__name__}); expected "
                f"{{'url': ..., 'path': ...}}"
            )
            continue
        url = f.get("url")
        path = f.get("path")
        if not isinstance(url, str) or not url.strip():
            errors.append(
                f"{label}: download_url: _download_setup files[{i}] has "
                f"missing or empty 'url'"
            )
            continue
        if not isinstance(path, str) or not path.strip():
            errors.append(
                f"{label}: download_url: _download_setup files[{i}] has "
                f"missing or empty 'path'"
            )
            # still probe the URL — both fields are required, but the URL probe
            # is the more interesting failure to surface.
        if not url.lower().startswith(("http://", "https://")):
            errors.append(
                f"{label}: download_url: {url!r} is not http(s) — only web "
                f"URLs are probeable; pick a real download endpoint"
            )
            continue
        try:
            resp = requests.get(
                url,
                stream=True,
                timeout=_DOWNLOAD_PROBE_TIMEOUT,
                allow_redirects=True,
            )
            try:
                resp.raise_for_status()
                # Read a single chunk so the body actually has to start
                # streaming — catches presigned links that 200 on HEAD but
                # 403 mid-stream and HTML error pages served as 200.
                chunk = next(resp.iter_content(chunk_size=_DOWNLOAD_PROBE_BYTES), b"")
                if not chunk:
                    errors.append(
                        f"{label}: download_url: {url!r} returned an empty "
                        f"body — the URL likely no longer exists"
                    )
            finally:
                resp.close()
        except requests.RequestException as e:
            errors.append(
                f"{label}: download_url: {url!r} is not reachable "
                f"({type(e).__name__}: {e}) — the URL is fabricated or no "
                f"longer hosted; remove the download step or pick an existing "
                f"resource"
            )
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Setup-sequence file lifecycle
#
# The Ubuntu VM boots from a clean snapshot — there are NO pre-existing app
# files. Every file the task touches must be created or uploaded inside
# ``config`` BEFORE it's opened or referenced. The sets below classify each
# setup helper by how it interacts with the VM filesystem so the sequence
# validator can detect ``_open_setup(path=X)`` calls that reference paths no
# earlier setup entry actually produced.
#
#   Producers — declare a literal destination ``path`` that we can track.
#   Opaque    — run shell commands; could create anything, so they disable the
#               "missing file" check for subsequent consumers.
#   Consumers — require the named path to already exist on the VM.
# ─────────────────────────────────────────────────────────────────────────────

# Setup helpers whose ``files=[{"url":..., "path":...}, ...]`` (or
# ``{"local_path":..., "path":...}``) argument names a literal destination on
# the VM. The dict's ``"path"`` key is the on-VM path that becomes available
# after the helper runs.
_FILES_LIST_PRODUCERS: Tuple[str, ...] = ("_download_setup", "_upload_file_setup")

# Helpers that run arbitrary shell or launch a process. We can't know what
# files they leave behind, so seeing one disables ``missing_file_setup``
# checks for everything that follows in the same setup list (avoids false
# positives — the LLM may legitimately write a file via ``cat <<EOF`` and
# then open it).
_OPAQUE_SHELL_SETUPS: frozenset = frozenset({
    "_command_setup",
    "_execute_setup",
    "_execute_with_verification_setup",
    "_launch_setup",
    "_googledrive_setup",
})

# Helpers whose first ``path`` argument MUST already exist on the VM.
_PATH_CONSUMERS: Tuple[str, ...] = ("_open_setup", "_change_wallpaper_setup")

# Helpers that take a single literal ``path`` argument and create that file
# on the VM. The validator registers the path so a downstream ``_open_setup``
# referencing the same path won't trigger ``missing_file_setup``. Keep this
# in sync with new ``_create_*_setup`` helpers in
# ``desktop_env/controllers/setup.py``.
_PATH_PRODUCERS: Tuple[str, ...] = (
    "_create_calc_file_setup",
    "_create_writer_file_setup",
    "_create_impress_file_setup",
    "_create_gimp_image_setup",
)


def _resolve_rules_dict_for_metric(
    metric_node: ast.Call, parsed: _ParsedCall
) -> Optional[Dict[Any, Any]]:
    """Static resolution of a metric's rules dict, following ``get_rule``.

    The canonical synthesis pattern is
    ``metric(getter(...), get_rule(env, config={'rules': {...}}))`` — the
    rules dict for the metric is the inner ``rules`` config of the
    ``get_rule`` call. ``_resolve_dict_arg`` only handles literal dicts, so
    by itself it can't see through this indirection. This helper unpacks
    one level of ``get_rule(...)`` and returns the inner literal dict when
    statically available; otherwise falls back to ``_resolve_dict_arg`` for
    the literal-dict case (``rules={...}`` written directly in the eval).
    """
    # 1. Literal dict written inline (rules={...}) — already covered.
    direct = _resolve_dict_arg(parsed, "rules", pos_index=1)
    if direct is not None:
        return direct

    # 2. Indirection via get_rule(env, config={'rules': {...}}).
    if "rules" in parsed.kwargs:
        # The kwarg was an opaque non-literal value; we need the AST node
        # itself to recognise get_rule, which the parsed form has discarded.
        # Locate the AST node for the rules kwarg.
        for kw in metric_node.keywords:
            if kw.arg == "rules":
                rules_node = kw.value
                break
        else:
            return None
    elif len(metric_node.args) > 1:
        rules_node = metric_node.args[1]
    else:
        return None

    if not isinstance(rules_node, ast.Call):
        return None
    if not isinstance(rules_node.func, ast.Name) or rules_node.func.id != "get_rule":
        return None
    inner = _parse_call(rules_node)
    cfg = _resolve_dict_arg(inner, "config", pos_index=1)
    if not cfg:
        return None
    rules = cfg.get("rules")
    return rules if isinstance(rules, dict) else None


def _check_required_keys(
    call: _ParsedCall,
    label: str,
    fn_info: Optional[Dict[str, Any]],
    metric_node: Optional[ast.Call] = None,
) -> List[str]:
    """Verify literal dicts contain the keys the implementation reaches for.

    Required keys are sourced from the function's evaluator schema (if it
    exposes one) and otherwise from the ``_GETTER_FALLBACK_CONFIG_KEYS`` /
    ``_METRIC_FALLBACK_RULES_KEYS`` baseline. Schema-driven checks expand
    automatically as functions are annotated; the fallback covers the
    most common cases for un-annotated functions.
    """
    errors: List[str] = []
    schema = (fn_info or {}).get("schema") or {}

    # ---- config (getter convention) ----------------------------------------
    required_config = required_config_keys(schema)
    if not required_config:
        required_config = _GETTER_FALLBACK_CONFIG_KEYS.get(call.func_name, [])
    if required_config:
        cfg = _resolve_dict_arg(call, "config", pos_index=1)
        if cfg is not None:
            missing = [k for k in required_config if k not in cfg]
            if missing:
                errors.append(
                    f"{label}: {call.func_name}(...) reads config[{missing[0]!r}] "
                    f"in its implementation but the literal config dict only "
                    f"has keys {sorted(cfg.keys())} — call will raise KeyError "
                    f"at runtime"
                )

    # ---- rules (metric convention) -----------------------------------------
    required_rules = required_rules_keys(schema)
    if not required_rules:
        required_rules = _METRIC_FALLBACK_RULES_KEYS.get(call.func_name, [])
    if required_rules:
        rules = (
            _resolve_rules_dict_for_metric(metric_node, call)
            if metric_node is not None
            else _resolve_dict_arg(call, "rules", pos_index=1)
        )
        if rules is not None:
            missing = [k for k in required_rules if k not in rules]
            if missing:
                errors.append(
                    f"{label}: {call.func_name}(...) reads rules[{missing[0]!r}] "
                    f"in its implementation but the literal rules dict only "
                    f"has keys {sorted(rules.keys())} — call will raise KeyError "
                    f"at runtime"
                )

    return errors


def _validate_setup_entry(
    entry: str,
    setup_index: Dict[str, Dict[str, Any]],
    label: str,
) -> Tuple[List[str], Optional[_ParsedCall]]:
    """Parse one setup-config string, look up the function, bind args.

    Returns ``(errors, parsed_call_or_None)``. The parsed call is returned so
    sequence-level checks (e.g. ``missing_file_setup``) can inspect it
    without re-parsing.
    """
    if not isinstance(entry, str) or not entry.strip():
        return [f"{label}: setup entry is empty or not a string: {entry!r}"], None
    try:
        tree = ast.parse(entry, mode="eval")
    except SyntaxError as e:
        return (
            [
                f"{label}: setup entry {entry!r} is not valid Python "
                f"(syntax error: {e.msg} at offset {e.offset})"
            ],
            None,
        )
    if not isinstance(tree.body, ast.Call):
        return (
            [
                f"{label}: setup entry {entry!r} parses as Python but is not a "
                f"function call (expected something like _open_setup(path=...))"
            ],
            None,
        )
    call = _parse_call(tree.body)
    info = setup_index.get(call.func_name)
    if info is None:
        return (
            [
                f"{label}: unknown setup function {call.func_name!r} — "
                f"the SetupController has no method by that name; the LLM may "
                f"have invented it"
            ],
            call,
        )
    errs = _check_signature(call, info.get("_sig"), label)
    # `_download_setup` is discouraged by the TASK_GEN prompt; when the LLM
    # emits one anyway, fail fast on fabricated/unreachable URLs by actually
    # fetching them here, before the example reaches the VM. URL errors are
    # tagged with ``download_url:`` to distinguish them from other setup
    # failures (notably ``missing_file_setup`` raised by the sequence walker).
    if call.func_name == "_download_setup":
        files_arg = _resolve_list_arg(call, "files", pos_index=0)
        if files_arg is None:
            errs.append(
                f"{label}: download_url: _download_setup must be called with "
                f"a literal `files=[{{'url': ..., 'path': ...}}, ...]` list so "
                f"the validator can probe each URL"
            )
        else:
            errs.extend(_validate_download_urls(files_arg, label))
    return errs, call


@dataclass
class _SetupSequenceState:
    """Accumulated lifecycle context as the validator walks a setup list.

    ``created_paths`` are literal VM paths that an earlier producer setup
    helper (download / upload) has registered. ``opaque_shell`` records
    whether any ``_command_setup``-class helper has appeared — once true,
    subsequent ``missing_file_setup`` checks are skipped because we can't
    statically know what files the shell actually wrote.
    """
    created_paths: set = field(default_factory=set)
    opaque_shell: bool = False


def _files_destinations(call: _ParsedCall) -> List[str]:
    """Extract destination ``path`` values from a producer's ``files`` list.

    Used for ``_download_setup`` / ``_upload_file_setup`` whose argument is a
    list of ``{"url"|"local_path": ..., "path": ...}`` dicts. Only literal
    string paths are returned — opaque entries cannot be tracked statically.
    """
    files = _resolve_list_arg(call, "files", pos_index=0)
    if not files:
        return []
    out: List[str] = []
    for f in files:
        if isinstance(f, dict):
            p = f.get("path")
            if isinstance(p, str) and p.strip():
                out.append(p)
    return out


def _consumer_path(call: _ParsedCall) -> Optional[str]:
    """Return the literal ``path`` argument for a consumer setup, if any."""
    if "path" in call.kwargs:
        val = call.kwargs["path"]
    elif call.pos_args:
        val = call.pos_args[0]
    else:
        return None
    if val is _OPAQUE or not isinstance(val, str) or not val.strip():
        return None
    return val


def _validate_setup_sequence(
    entries: List[Any],
    setup_index: Dict[str, Dict[str, Any]],
    label_prefix: str,
    state: Optional[_SetupSequenceState] = None,
) -> Tuple[List[str], _SetupSequenceState]:
    """Validate a setup list with cross-entry file-lifecycle awareness.

    Each entry runs through ``_validate_setup_entry`` for the per-call
    signature/URL checks, and the parsed call is then folded into a running
    ``_SetupSequenceState`` so ``_open_setup``-style consumers can be checked
    against the paths earlier producers actually created. The check only
    fires when ALL of:

    - the consumer's ``path`` argument is a literal string,
    - no opaque shell helper has appeared earlier in the sequence,
    - the path is not in ``state.created_paths``.

    Producers register their literal destination paths even when they emit
    other errors (e.g. an unreachable download URL still announces the path
    so a downstream `_open_setup` reading that path doesn't double-fault as
    missing-file).
    """
    state = state or _SetupSequenceState()
    errors: List[str] = []
    if not isinstance(entries, list):
        return errors, state
    for i, entry in enumerate(entries):
        label = f"{label_prefix}[{i}]"
        per_entry, call = _validate_setup_entry(entry, setup_index, label)
        errors.extend(per_entry)
        if call is None:
            continue

        if call.func_name in _FILES_LIST_PRODUCERS:
            for p in _files_destinations(call):
                state.created_paths.add(p)
        elif call.func_name in _PATH_PRODUCERS:
            p = _consumer_path(call)
            if p is not None:
                state.created_paths.add(p)
        elif call.func_name in _OPAQUE_SHELL_SETUPS:
            state.opaque_shell = True

        if call.func_name in _PATH_CONSUMERS:
            path = _consumer_path(call)
            if path is None:
                # Non-literal path — can't check; per-entry signature check
                # already covered missing/empty values.
                continue
            if state.opaque_shell or path in state.created_paths:
                continue
            errors.append(
                f"{label}: missing_file_setup: {call.func_name}(path={path!r}) "
                f"references a file that no earlier setup entry created. The "
                f"VM boots from a clean snapshot with no pre-existing app "
                f"files; add a producer step that creates {path!r} first — "
                f"for office/image fixtures prefer the domain-specific "
                f"`_create_calc_file_setup` / `_create_writer_file_setup` / "
                f"`_create_impress_file_setup` / `_create_gimp_image_setup` "
                f"helpers, otherwise use `_command_setup` (heredoc/printf) or "
                f"`_upload_file_setup`."
            )
    return errors, state


# Builtins that pass the inner expression's score through unchanged. When one
# of these wraps a metric call, the metric is still the "outer" score-producing
# call; ``_outer_score_calls`` recurses through them to find the real outer.
_SCORE_PRESERVING_BUILTINS = {"float", "int", "bool", "abs"}


def _outer_score_calls(tree: ast.AST) -> set:
    """Identify call nodes whose return value is a conjunct of the eval.

    A well-formed eval reduces to one or more 0/1-returning calls combined
    with ``and`` / ``or``. Each such call must be a metric. This walker
    descends through:

    - ``and`` / ``or``  → every operand is a separate conjunct.
    - ``not``           → the operand is the conjunct.
    - comparisons       → the left side is the conjunct (e.g. ``metric(...) == 1``).
    - score-preserving builtins (``float`` / ``int`` / ``bool`` / ``abs``) →
      the first arg is the conjunct.

    Anything else encountered as a top-level call is recorded as outer.
    """
    out: set = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                visit(v)
        elif isinstance(node, ast.UnaryOp):
            visit(node.operand)
        elif isinstance(node, ast.Compare):
            visit(node.left)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _SCORE_PRESERVING_BUILTINS
                and node.args
            ):
                visit(node.args[0])
            else:
                out.add(id(node))
        # Other nodes (Name, Constant, ...) at the top level can't produce a
        # score on their own; the empty-eval branch already rejects those.

    if isinstance(tree, ast.Expression):
        visit(tree.body)
    else:
        visit(tree)
    return out


def _check_eval_call_role(
    node: ast.Call,
    call: _ParsedCall,
    fn_info: Optional[Dict[str, Any]],
    is_outer: bool,
    parent_metric_name: Optional[str],
) -> List[str]:
    """Enforce the role contract declared by the ``@evaluator`` decorator.

    - Every outer (score-producing) call must have ``role == "metric"`` so
      the eval actually returns a 0/1 score per conjunct. A getter at the
      outer position means there is no comparison happening at all.
    - A call that appears as a positional argument to a metric must NOT
      itself be a metric. Nesting metrics is almost always a synthesis
      mistake — the outer metric receives a 0/1 score where it expects raw
      VM state.
    """
    role = (fn_info or {}).get("role", "")
    if not role:
        return []  # role unknown (un-decorated; nothing to enforce)
    errors: List[str] = []
    if is_outer and role != "metric":
        errors.append(
            f"eval/{call.func_name}: top-level call has role={role!r}, but "
            f"each conjunct of the eval must be a metric that returns a 0/1 "
            f"score (e.g. exact_match, check_json, "
            f"is_expected_url_pattern_match). Wrap the getter output in a "
            f"metric, or replace the outer call entirely."
        )
    if (not is_outer) and role == "metric" and parent_metric_name is not None:
        errors.append(
            f"eval/{call.func_name}: metric call is nested inside "
            f"{parent_metric_name}(...), which feeds it a 0/1 score where raw "
            f"VM state is expected. Use a getter (role='getter') to extract "
            f"state for the outer metric."
        )
    return errors


def _build_metric_arg_parents(tree: ast.AST, fn_index: Dict[str, Dict[str, Any]]) -> Dict[int, str]:
    """Map ``id(call_node) -> parent_metric_name`` for each call passed as a
    positional argument to a metric. Only direct positional args are tracked;
    kwargs (rules={...}, config={...}) are usually literal dicts, not calls.
    """
    parents: Dict[int, str] = {}
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.Call):
            continue
        if not isinstance(parent.func, (ast.Name, ast.Attribute)):
            continue
        parent_name = parent.func.id if isinstance(parent.func, ast.Name) else parent.func.attr
        info = fn_index.get(parent_name)
        if not info or info.get("role") != "metric":
            continue
        for arg in parent.args:
            if isinstance(arg, ast.Call):
                parents[id(arg)] = parent_name
    return parents


def validate_example_scripts(example: Dict[str, Any], catalog: FunctionCatalog) -> ValidationResult:
    """Statically validate the setup + evaluator scripts of a synthesized example.

    Beyond syntax, every call is bound against the real function's
    Signature, and a small registry of well-known dict-key requirements
    (e.g. ``get_vm_file`` indexing ``config["path"]``) is checked against
    literal dict arguments. The goal is to reject examples whose runtime
    invocation would predictably raise — saving a VM round-trip and
    surfacing a precise reason that gets stored in synthesis memory.

    Returns a ``ValidationResult`` whose ``errors`` are full sentences
    suitable for memory persistence and for prompting the LLM in
    subsequent rounds.
    """
    setup_index = {f["name"]: f for f in catalog.setup_functions}
    fn_index = catalog.index_by_name()
    eval_allowed = (
        {f["name"] for f in catalog.getter_functions}
        | {f["name"] for f in catalog.metric_functions}
        | _EVAL_BUILTINS
    )

    errors: List[str] = []

    # --- setup config + evaluator postconfig ---------------------------------
    # ``config`` runs against a clean snapshot, so the sequence walker enforces
    # the file-lifecycle rule (every consumer path must be created upstream).
    # ``postconfig`` runs after the agent's task — the agent may have produced
    # arbitrary files we can't see statically — so we only run per-entry checks
    # there and skip the missing_file_setup pass.
    config_entries = example.get("config", [])
    if not isinstance(config_entries, list):
        errors.append(
            "config: must be a list of setup-call strings, "
            f"got {type(config_entries).__name__}"
        )
        config_entries = []
    config_errors, _ = _validate_setup_sequence(
        config_entries, setup_index, "config",
    )
    errors.extend(config_errors)

    evaluator = example.get("evaluator", {})
    if not isinstance(evaluator, dict):
        errors.append(
            "evaluator: must be a dict with keys 'eval' and optionally "
            f"'postconfig', got {type(evaluator).__name__}"
        )
        evaluator = {}

    postconfig_entries = evaluator.get("postconfig", []) or []
    if not isinstance(postconfig_entries, list):
        errors.append(
            "evaluator.postconfig: must be a list of setup-call strings, "
            f"got {type(postconfig_entries).__name__}"
        )
        postconfig_entries = []
    for i, entry in enumerate(postconfig_entries):
        per_entry, _ = _validate_setup_entry(
            entry, setup_index, f"postconfig[{i}]",
        )
        errors.extend(per_entry)

    # --- eval expression -----------------------------------------------------
    eval_str = (evaluator.get("eval") or "").strip()
    if not eval_str:
        errors.append(
            "eval: the verifier eval expression is empty — without it nothing "
            "checks the resulting VM state"
        )
        return ValidationResult(valid=False, errors=errors)

    try:
        tree = ast.parse(eval_str, mode="eval")
    except SyntaxError as e:
        errors.append(
            f"eval: expression {eval_str!r} is not valid Python "
            f"(syntax error: {e.msg} at offset {e.offset})"
        )
        return ValidationResult(valid=False, errors=errors)

    outer_call_ids = _outer_score_calls(tree)
    metric_arg_parents = _build_metric_arg_parents(tree, fn_index)

    calls_found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            continue
        calls_found += 1
        call = _parse_call(node)

        # Builtins are allowed but we don't bind their signatures (they
        # accept too many shapes to be useful for static checks).
        if call.func_name in _EVAL_BUILTINS:
            continue

        if call.func_name not in eval_allowed:
            errors.append(
                f"eval: expression calls unknown function {call.func_name!r}; "
                f"only getters/metrics from the cataloged library plus the "
                f"builtins {sorted(_EVAL_BUILTINS)} are permitted"
            )
            continue

        info = fn_index.get(call.func_name)
        if info is not None:
            errors.extend(_check_signature(
                call, info.get("_sig"), f"eval/{call.func_name}",
            ))
        errors.extend(_check_required_keys(
            call, f"eval/{call.func_name}", info, metric_node=node,
        ))
        errors.extend(_check_eval_call_role(
            node, call, info,
            is_outer=id(node) in outer_call_ids,
            parent_metric_name=metric_arg_parents.get(id(node)),
        ))

    if calls_found == 0:
        errors.append(
            "eval: expression parses but contains no function calls — "
            "a verifier without a getter+metric composition cannot evaluate "
            "anything"
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ═══════════════════════════════════════════════════════════════════════════
# Batched synthesis loop
# ═══════════════════════════════════════════════════════════════════════════


def _synthesize_domain(
    domain: str,
    args: argparse.Namespace,
    memory: Optional[SynthesisMemory],
    vector_store: Optional[VectorDedupStore],
    on_batch_complete: Optional[
        Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]
    ],
    catalog: FunctionCatalog,
    llm_config: Dict[str, Any],
    batch_size: int,
    total_examples: int,
    max_empty_batches: int,
    write_lock: Optional[threading.Lock] = None,
) -> List[Dict[str, Any]]:
    """Run the batched synthesis loop for one domain end-to-end.

    Per batch: generate via LLM → static-validate → vector-dedup → persist
    accepted to disk → verify (when ``on_batch_complete`` is set) → record
    outcomes into ``memory`` and persist memory + vector store. The memory
    JSON file and the ChromaDB store are flushed once per batch so concurrent
    domain workers (parallel mode) can pick up the latest state on their
    next iteration.

    ``write_lock`` (optional) — held around every read/write of ``memory``,
    ``vector_store``, and ``on_batch_complete`` so parallel workers don't
    corrupt shared state. The LLM call (``generate_task_examples``) and
    static validation run outside the lock so synthesis stays parallel; only
    the bookkeeping serializes. Sequential callers pass ``None``.
    """
    hold = (lambda: write_lock) if write_lock is not None else (lambda: nullcontext())

    logger.info("=" * 60)
    logger.info(
        f"Synthesizing for domain: {domain} "
        f"(target={total_examples}, batch_size={batch_size})"
    )

    domain_info = load_domain_examples(domain, max_examples=args.max_ref_examples)
    domain_dir = os.path.join(args.output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)

    with hold():
        existing_valid = sum(
            1 for e in (memory.get_domain_entries(domain) if memory else [])
            if e.get("executable")
        )
    logger.info(
        f"Domain '{domain}': {existing_valid} validated examples already in memory"
    )

    session_valid: List[Dict[str, Any]] = []
    empty_streak = 0
    batch_idx = 0
    while existing_valid + len(session_valid) < total_examples:
        remaining = total_examples - (existing_valid + len(session_valid))
        n_this_batch = min(batch_size, remaining)
        batch_idx += 1

        # Snapshot the prior-experience prompt block under the lock so the
        # read sees a consistent memory snapshot; the LLM call below then
        # runs OUTSIDE the lock so parallel workers overlap on generation.
        memory_block: Optional[str] = None
        if memory is not None:
            with hold():
                memory_block = memory.format_for_prompt(domain)

        examples = generate_task_examples(
            domain_info, catalog, llm_config,
            n_this_batch, args.max_steps,
            memory=None, memory_block=memory_block,
        )
        
        # breakpoint()

        # 1) Static validation (pure CPU work — runs outside the lock)
        batch_valid: List[Dict[str, Any]] = []
        statically_invalid: List[Tuple[Dict[str, Any], str]] = []
        for ex in examples:
            vr = validate_example_scripts(ex, catalog)
            if vr.valid:
                batch_valid.append(ex)
            else:
                err = "; ".join(vr.errors)
                logger.warning(
                    f"SCRIPT VALIDATION FAILED for {ex.get('id', '?')}: {err} – skipping"
                )
                statically_invalid.append((ex, err))

        # 2) Vector-DB dedup — locked so the read sees writes from peers'
        #    add_solvable calls and the per-store collection cache stays
        #    consistent across threads.
        duplicates: List[Tuple[Dict[str, Any], Any]] = []
        if vector_store is not None and batch_valid:
            with hold():
                decision = vector_store.filter_batch(domain, batch_valid)
            if decision.rejected:
                logger.info(
                    f"DEDUP: rejected {len(decision.rejected)}/"
                    f"{len(batch_valid)} as near-duplicates"
                )
                for ex_rej, match in decision.rejected:
                    logger.info(
                        f"  - skip {str(ex_rej.get('id', '?'))[:8]} "
                        f"sim={match.similarity:.3f} [{match.source}] "
                        f"-> {str(match.id)[:8]} ({(match.instruction or '')[:80]!r})"
                    )
            batch_valid = decision.accepted
            duplicates = decision.rejected

        # 3) Persist accepted examples to the vector store BEFORE writing
        #    them to disk, so the next batch (and the next run) can dedup
        #    against them. Without this step the Chroma collection only grows
        #    after verification, which means --mode synthesize runs never
        #    persist anything and re-running the script regenerates
        #    near-duplicates of what was already produced.
        if vector_store is not None and batch_valid:
            added = 0
            with hold():
                for ex in batch_valid:
                    if vector_store.add_solvable(ex, domain):
                        added += 1
            logger.info(
                f"[vector-dedup] persisted {added}/{len(batch_valid)} "
                f"accepted example(s) for domain '{domain}'"
            )

        # 4) Persist accepted examples to disk (unique filenames per id, so
        #    no cross-thread file collision — runs outside the lock).
        for ex in batch_valid:
            path = os.path.join(domain_dir, f"{ex['id']}.json")
            with open(path, "w") as f:
                json.dump(ex, f, indent=2)
            logger.info(f"Saved {path}")

        # 5) Verification first — gather results before touching memory.
        #    Locked because _process_verify_results does a read-modify-write
        #    on verification_results.json / solvable manifest, and pushes
        #    new entries into the vector store; concurrent calls would
        #    race on those files and the Chroma collection cache.
        verify_results: Optional[List[Dict[str, Any]]] = None
        if on_batch_complete is not None and batch_valid:
            try:
                with hold():
                    verify_results = on_batch_complete(domain, batch_valid)
            except Exception as e:
                logger.error(
                    f"Domain '{domain}' batch {batch_idx}: "
                    f"on_batch_complete callback failed: {e}"
                )
                logger.error(
                    "Continuing synthesis; batch left unverified "
                    "(rerun --mode verify to pick it up)."
                )

        # 6) Memory record + persist (single pass, after verification).
        #    memory.save() rewrites the whole JSON file, so this whole block
        #    must be atomic w.r.t. peer threads.
        if memory is not None:
            with hold():
                for ex, err in statically_invalid:
                    memory.record(
                        example=ex, domain=domain,
                        code_result={"score": -1, "error": f"script_validation: {err}"},
                        executable=False,
                    )
                for ex, match in duplicates:
                    memory.record(
                        example=ex, domain=domain,
                        code_result={
                            "score": -1,
                            "error": f"duplicate_of={match.id} sim={match.similarity:.3f} [{match.source}]",
                        },
                        executable=False,
                    )
                results_by_id = {
                    r["id"]: r for r in (verify_results or []) if r.get("id")
                }
                for ex in batch_valid:
                    r = results_by_id.get(ex["id"])
                    if r is None:
                        # Synthesize-only mode (no verification ran).
                        memory.record(
                            example=ex, domain=domain,
                            code_result={"score": -1},
                            executable=True, solvable=None,
                        )
                    else:
                        solvable = None if "error" in r else (r.get("score", -1) >= 0)
                        memory.record(
                            example=ex, domain=domain,
                            code_result=r,
                            executable=True, solvable=solvable,
                        )
                memory.save()

        session_valid.extend(batch_valid)
        logger.info(
            f"Domain '{domain}' batch {batch_idx}: "
            f"{len(batch_valid)}/{len(examples)} passed validation "
            f"(progress: {existing_valid + len(session_valid)}/{total_examples})"
        )
        
        # breakpoint()

        if len(batch_valid) == 0:
            empty_streak += 1
            if empty_streak >= max_empty_batches:
                logger.warning(
                    f"Domain '{domain}': {empty_streak} consecutive batches "
                    f"yielded no valid examples — stopping early."
                )
                break
        else:
            empty_streak = 0

    return session_valid


def _run_synthesize_sequential(
    args: argparse.Namespace,
    memory: Optional[SynthesisMemory],
    vector_store: Optional[VectorDedupStore],
    on_batch_complete: Optional[
        Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]
    ],
    targets: List[str],
    catalog: FunctionCatalog,
    llm_config: Dict[str, Any],
    batch_size: int,
    total_examples: int,
    max_empty_batches: int,
) -> List[Dict[str, Any]]:
    """Sequential synthesis: process domains one at a time in the main thread."""
    logger.info(
        f"Synthesis (sequential): {len(targets)} domain(s) "
        f"(target={total_examples}, batch_size={batch_size})"
    )
    all_examples: List[Dict[str, Any]] = []
    for domain in targets:
        all_examples.extend(_synthesize_domain(
            domain, args, memory, vector_store, on_batch_complete,
            catalog, llm_config, batch_size, total_examples,
            max_empty_batches, write_lock=None,
        ))
    return all_examples


def _run_synthesize_parallel(
    args: argparse.Namespace,
    memory: Optional[SynthesisMemory],
    vector_store: Optional[VectorDedupStore],
    on_batch_complete: Optional[
        Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]
    ],
    targets: List[str],
    catalog: FunctionCatalog,
    llm_config: Dict[str, Any],
    batch_size: int,
    total_examples: int,
    max_empty_batches: int,
) -> List[Dict[str, Any]]:
    """Parallel synthesis: a thread pool runs one domain per worker.

    Threads are appropriate here because the per-batch hot path is dominated
    by an I/O-bound LLM call (``generate_task_examples``); the synthesis
    bookkeeping is serialized via a single ``write_lock`` shared by every
    worker.
    """
    synthesize_workers = max(1, min(getattr(args, "synthesize_workers", 1), len(targets)))
    # breakpoint()
    logger.info(
        f"Synthesis (parallel): {len(targets)} domain(s) across "
        f"{synthesize_workers} worker thread(s) "
        f"(target={total_examples}, batch_size={batch_size})"
    )

    write_lock = threading.Lock()
    all_examples: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=synthesize_workers) as executor:
        future_to_domain = {
            executor.submit(
                _synthesize_domain,
                domain, args, memory, vector_store, on_batch_complete,
                catalog, llm_config, batch_size, total_examples,
                max_empty_batches, write_lock,
            ): domain
            for domain in targets
        }
        for fut in as_completed(future_to_domain):
            domain = future_to_domain[fut]
            try:
                all_examples.extend(fut.result())
            except Exception as e:
                logger.error(f"Domain '{domain}' synthesis failed: {e}")
                logger.error(traceback.format_exc())
    return all_examples


def run_synthesize(
    args: argparse.Namespace,
    memory: Optional[SynthesisMemory] = None,
    vector_store: Optional[VectorDedupStore] = None,
    on_batch_complete: Optional[
        Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]
    ] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic examples + verifiers via LLM in fixed-size batches.

    Loops until ``--total-examples`` valid (script-validated, non-duplicate)
    examples exist in memory for each domain. Between batches, previously-seen
    instructions are surfaced via ``SynthesisMemory.format_for_prompt`` and
    near-duplicates of solvable tasks are filtered via ``vector_store``.

    Order of operations per batch (see ``_synthesize_domain``):
      1. Generate, validate, dedup, persist accepted examples to disk.
      2. Run verification (via ``on_batch_complete``) on accepted examples.
      3. THEN record every batch outcome — script-invalid, duplicate,
         executable-only, solvable, errored — into memory in a single pass
         and persist it. Verification always happens before any memory
         write so the persisted record reflects the final status.

    ``args.synthesize_mode`` selects the dispatcher:
      - ``"sequential"`` (default): one domain at a time in the main thread.
      - ``"parallel"``: a thread pool of size ``args.synthesize_workers``
        runs one domain per worker; a shared lock serializes memory +
        vector-store bookkeeping while LLM calls run concurrently.

    ``on_batch_complete`` (optional) returns the verification results for
    the accepted examples so this function can fold them into memory.
    """
    all_domains = discover_domains()
    catalog = catalog_functions()
    targets = args.domains if args.domains else all_domains
    targets = [d for d in targets if d in all_domains]

    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    os.makedirs(args.output_dir, exist_ok=True)

    batch_size = args.batch_size if args.batch_size > 0 else args.num_examples
    total_examples = args.total_examples if args.total_examples > 0 else args.num_examples
    max_empty_batches = max(1, args.max_empty_batches)

    synthesize_mode = getattr(args, "synthesize_mode", "sequential")
    # breakpoint()
    if synthesize_mode == "parallel":
        all_examples = _run_synthesize_parallel(
            args, memory, vector_store, on_batch_complete, targets,
            catalog, llm_config, batch_size, total_examples, max_empty_batches,
        )
    else:
        all_examples = _run_synthesize_sequential(
            args, memory, vector_store, on_batch_complete, targets,
            catalog, llm_config, batch_size, total_examples, max_empty_batches,
        )

    # Rebuild manifest from every validated example on disk so accumulated
    # runs are all visible to the verify stage.
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest: Dict[str, List[str]] = {}
    for domain in targets:
        domain_dir = os.path.join(args.output_dir, domain)
        if not os.path.isdir(domain_dir):
            continue
        ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(domain_dir, "*.json")))
        ]
        if ids:
            manifest[domain] = ids
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    total_on_disk = sum(len(v) for v in manifest.values())
    logger.info(
        f"Manifest rebuilt from disk: {total_on_disk} examples across "
        f"{len(manifest)} domains → {manifest_path}"
    )
    return all_examples
