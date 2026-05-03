"""Synthesis utilities: domain/function discovery, prompt formatters, and the
static validator (syntax + signature + setup-sequence + eval-role checks).

Everything in this module is independent of the LLM-driven generation loop —
it only depends on the static catalog of setup/getter/metric functions and
the AST shape of the strings the LLM emits. ``task_creator.py`` consumes
these helpers; nothing here imports back from it.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import glob
import inspect
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

from desktop_env.evaluators.schema import (
    _first_doc_line,
    get_schema,
    required_config_keys,
    required_rules_keys,
)

logger = logging.getLogger("desktopenv.synthesis.utils")

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
    domain_dir = os.path.join(EXAMPLES_DIR, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
    info = DomainInfo(name=domain)
    json_files = sorted(glob.glob(os.path.join(domain_dir, "*.json")))
    info.example_files = json_files
    to_load = json_files
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
        return []
    if call.has_splat:
        return []
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

_FILES_LIST_PRODUCERS: Tuple[str, ...] = ("_download_setup", "_upload_file_setup")

_OPAQUE_SHELL_SETUPS: frozenset = frozenset({
    "_command_setup",
    "_execute_setup",
    "_execute_with_verification_setup",
    "_launch_setup",
    "_googledrive_setup",
})

_PATH_CONSUMERS: Tuple[str, ...] = ("_open_setup", "_change_wallpaper_setup")

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
    direct = _resolve_dict_arg(parsed, "rules", pos_index=1)
    if direct is not None:
        return direct

    if "rules" in parsed.kwargs:
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
        return []
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
# Extra-function (LLM-declared) execution verification
#
# The TASK_GEN prompt allows the LLM to declare its own setup / getter /
# metric helpers via the optional ``extra_functions`` array when the catalog
# lacks what it needs. The functions below verify that:
#
#   * The implementation source compiles and exec's cleanly.
#   * It produces a top-level callable matching the declared name.
#   * Its module body is restricted to safe top-level constructs (imports,
#     defs, classes, assignments, docstrings) — i.e. no module-level side
#     effects like ``os.system(...)`` running at exec time.
#
# Higher-level orchestration (referencing checks, catalog merging) lives in
# ``task_creator.py``; this module only provides the executable-verification
# primitive plus the catalog-extension helper that turns verified extras into
# entries the static validator can resolve against.
# ═══════════════════════════════════════════════════════════════════════════


# Top-level statements allowed in an ``extra_functions`` implementation. Other
# statement kinds — notably ``ast.Expr`` wrapping a call (e.g.
# ``os.system(...)``) — are rejected so a malformed or hostile implementation
# can't run code at exec time. Docstring ``Expr`` nodes are tolerated via an
# explicit Constant-string check inside ``_check_safe_extra_module``.
_ALLOWED_TOP_LEVEL_STMTS: Tuple[type, ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)

# Allowed values for the ``kind`` field of an extra_functions[] entry.
_EXTRA_FUNCTION_KINDS: frozenset = frozenset({"setup", "getter", "metric"})


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    """True if ``stmt`` is a top-level docstring expression."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _check_safe_extra_module(impl: str, name: str) -> List[str]:
    """Reject module-level side effects in an extra_functions implementation.

    Parses ``impl`` and walks the top-level statements; everything outside
    ``_ALLOWED_TOP_LEVEL_STMTS`` (and module/function docstrings) is rejected
    so ``exec`` doesn't run arbitrary expressions during validation. Function
    bodies are intentionally NOT scanned — they only run when the function is
    invoked by the verifier in a real VM, not during static checks.
    """
    try:
        tree = ast.parse(impl)
    except SyntaxError as e:
        return [
            f"extra_functions[{name!r}]: implementation has SyntaxError "
            f"({e.msg} at line {e.lineno})"
        ]
    errors: List[str] = []
    for stmt in tree.body:
        if isinstance(stmt, _ALLOWED_TOP_LEVEL_STMTS):
            continue
        if _is_docstring_expr(stmt):
            continue
        errors.append(
            f"extra_functions[{name!r}]: top-level statement at line "
            f"{stmt.lineno} is a {type(stmt).__name__} — only imports, "
            f"function/class defs, and assignments are allowed at module "
            f"scope (move side effects into the function body so they run "
            f"only when the verifier invokes the function)"
        )
    return errors


def referenced_function_names(example: Dict[str, Any]) -> Set[str]:
    """Collect every function name called from config / postconfig / eval.

    Used by the orchestration in ``task_creator.py`` to confirm that each
    ``extra_functions[].name`` is actually referenced from the example —
    declarations without a referrer are dead weight and surface as errors.

    Parsing failures are silently ignored here: the static validator already
    flags malformed config/eval strings with precise messages, so re-raising
    them here would just duplicate noise.
    """
    names: Set[str] = set()

    def _add_from_setup_str(s: Any) -> None:
        if not isinstance(s, str) or not s.strip():
            return
        try:
            tree = ast.parse(s, mode="eval")
        except SyntaxError:
            return
        if isinstance(tree.body, ast.Call):
            call = _parse_call(tree.body)
            if call.func_name and call.func_name != "<expr>":
                names.add(call.func_name)

    for s in example.get("config") or []:
        _add_from_setup_str(s)

    evaluator = example.get("evaluator") or {}
    for s in evaluator.get("postconfig") or []:
        _add_from_setup_str(s)

    eval_str = (evaluator.get("eval") or "").strip()
    if eval_str:
        try:
            tree = ast.parse(eval_str, mode="eval")
        except SyntaxError:
            return names
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def verify_extra_function_executable(
    spec: Any,
) -> Tuple[Optional[Callable[..., Any]], List[str]]:
    """Compile + exec one ``extra_functions[]`` entry and resolve its callable.

    The check is staged so each layer surfaces a precise error before the
    next one runs:

    1. ``spec`` is a dict with required string fields ``name``, ``kind``,
       ``implementation`` (and optional ``signature`` / ``description``).
    2. ``kind`` is one of ``setup`` / ``getter`` / ``metric``.
    3. The implementation parses, has only safe top-level constructs
       (``_check_safe_extra_module``), and compiles.
    4. ``exec`` runs in an isolated namespace with the standard builtins;
       any exception during exec is reported as an executability failure.
    5. The declared ``name`` resolves to a callable in that namespace.

    Returns ``(callable_or_None, errors)``. The callable is returned even
    when ``errors`` is empty so callers can graft it into an extended
    catalog without re-exec'ing.
    """
    errors: List[str] = []
    if not isinstance(spec, dict):
        return None, [
            f"extra_functions: expected dict, got {type(spec).__name__}"
        ]

    name = spec.get("name")
    kind = spec.get("kind")
    impl = spec.get("implementation")

    if not isinstance(name, str) or not name.strip():
        errors.append(
            "extra_functions: 'name' is required and must be a non-empty string"
        )
    if kind not in _EXTRA_FUNCTION_KINDS:
        errors.append(
            f"extra_functions[{name!r}]: 'kind' must be one of "
            f"{sorted(_EXTRA_FUNCTION_KINDS)}, got {kind!r}"
        )
    if not isinstance(impl, str) or not impl.strip():
        errors.append(
            f"extra_functions[{name!r}]: 'implementation' is required and "
            f"must be a non-empty Python source string"
        )
    if errors:
        return None, errors

    if not name.isidentifier():
        return None, [
            f"extra_functions[{name!r}]: 'name' is not a valid Python "
            f"identifier"
        ]

    safety_errors = _check_safe_extra_module(impl, name)
    if safety_errors:
        return None, safety_errors

    try:
        code = compile(impl, f"<extra_function:{name}>", "exec")
    except SyntaxError as e:
        return None, [
            f"extra_functions[{name!r}]: compile failed — SyntaxError "
            f"{e.msg} at line {e.lineno}"
        ]

    namespace: Dict[str, Any] = {
        "__name__": f"_extra_functions.{name}",
        "__builtins__": _builtins.__dict__,
    }
    try:
        exec(code, namespace)
    except Exception as e:
        return None, [
            f"extra_functions[{name!r}]: implementation raised "
            f"{type(e).__name__} at module-load time ({e}) — module body "
            f"must finish without error so the verifier can import the "
            f"function"
        ]

    fn = namespace.get(name)
    if fn is None:
        return None, [
            f"extra_functions[{name!r}]: implementation did not define a "
            f"top-level callable named {name!r} — confirm the def matches "
            f"the declared 'name' field"
        ]
    if not callable(fn):
        return None, [
            f"extra_functions[{name!r}]: top-level binding {name!r} is not "
            f"callable (got {type(fn).__name__})"
        ]
    try:
        inspect.signature(fn)
    except (ValueError, TypeError) as e:
        return None, [
            f"extra_functions[{name!r}]: callable has no introspectable "
            f"signature ({type(e).__name__}: {e}) — the verifier cannot "
            f"bind arguments to this function"
        ]
    return fn, []


def _extra_catalog_entry(
    spec: Dict[str, Any], fn: Callable[..., Any]
) -> Dict[str, Any]:
    """Wrap a verified extra function as a FunctionCatalog entry.

    Mirrors the shape ``_func_info`` produces so the static validator can
    treat extras as ordinary catalog members. ``role`` is set from the
    declared ``kind`` (empty for ``setup`` since role enforcement only
    applies to eval-side calls), ``schema`` is left empty (no decorator,
    no docstring contract), and ``module`` is tagged ``"<extra>"`` so logs
    can distinguish extras from real catalog entries.
    """
    try:
        sig: Optional[inspect.Signature] = inspect.signature(fn)
        sig_str = str(sig)
    except (ValueError, TypeError):
        sig = None
        sig_str = "(...)"
    kind = spec["kind"]
    role = kind if kind in ("getter", "metric") else ""
    name = spec["name"]
    return {
        "name": name,
        "signature": f"{name}{sig_str}",
        "doc": (spec.get("description") or "").strip(),
        "role": role,
        "schema": {},
        "module": "<extra>",
        "_sig": sig,
    }


def extend_catalog_with_extras(
    base: FunctionCatalog,
    extras: List[Tuple[Dict[str, Any], Callable[..., Any]]],
) -> FunctionCatalog:
    """Return a copy of ``base`` with ``(spec, fn)`` extras grafted in.

    Each extra is bucketed by its declared ``kind`` so the static validator
    finds it in the right list (setup_functions for ``kind == 'setup'``,
    etc.). The base catalog is not mutated — callers stay free to
    pass a single shared catalog into per-example validation without
    accumulating state across batches.
    """
    extended = FunctionCatalog(
        setup_functions=list(base.setup_functions),
        getter_functions=list(base.getter_functions),
        metric_functions=list(base.metric_functions),
    )
    for spec, fn in extras:
        entry = _extra_catalog_entry(spec, fn)
        kind = spec["kind"]
        if kind == "setup":
            extended.setup_functions.append(entry)
        elif kind == "getter":
            extended.getter_functions.append(entry)
        elif kind == "metric":
            extended.metric_functions.append(entry)
    return extended
