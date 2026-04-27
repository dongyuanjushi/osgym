"""Schema declarations for evaluator getters and metrics.

The synthesis catalog (``synthesis.task_creator``) reads schemas to teach
the LLM which dict keys each function expects and to statically validate
generated calls before they ever reach the VM. There are two ways to
contribute a schema:

1. The ``@evaluator(...)`` decorator (preferred for new code) — explicit,
   structured, easy to introspect, immune to docstring drift.

2. ``Config:`` / ``Rules:`` / ``Options:`` / ``Args:`` blocks in the
   function's docstring — automatic fallback so the many existing
   functions that already document their inputs in Sphinx-ish style
   contribute schema information without code churn.

Convention for both forms:
- Keys that are required at call time use their plain name (``"path"``).
- Optional keys are recorded with a trailing ``"?"`` (``"shell?"``) so
  ``validate_example_scripts`` only flags missing *required* keys.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

SCHEMA_ATTR = "__evaluator_schema__"
SETUP_SCHEMA_ATTR = "__setup_action_schema__"


def evaluator(
    *,
    role: str,
    config: Optional[Dict[str, str]] = None,
    rules: Optional[Dict[str, str]] = None,
    options: Optional[Dict[str, str]] = None,
    returns: str = "",
    summary: str = "",
) -> Callable[[F], F]:
    """Attach an evaluator schema to a getter or metric function.

    Args:
        role:    ``"getter"`` or ``"metric"``.
        config:  for getters whose second positional/kw arg is a config
                 dict — map of key name → human description. Suffix the key
                 with ``"?"`` (e.g. ``"shell?"``) to mark it optional.
        rules:   for metrics whose second positional/kw arg is a rules
                 dict (e.g. ``exact_match(result, rules)``).
        options: for metrics that accept ``**options`` keyword arguments.
        returns: plain-language description of the return value.
        summary: one-line summary; defaults to the function's first
                 docstring line when omitted.
    """
    if role not in {"getter", "metric"}:
        raise ValueError(f"role must be 'getter' or 'metric', got {role!r}")

    def deco(fn: F) -> F:
        setattr(fn, SCHEMA_ATTR, {
            "role": role,
            "config": dict(config or {}),
            "rules": dict(rules or {}),
            "options": dict(options or {}),
            "returns": returns,
            "summary": summary or _first_doc_line(fn.__doc__),
            "source": "decorator",
        })
        return fn

    return deco


def setup_action(
    *,
    args: Optional[Dict[str, str]] = None,
    returns: str = "",
    summary: str = "",
) -> Callable[[F], F]:
    """Mark a SetupController method as a setup action.

    The dispatcher in ``SetupController.setup`` auto-registers every method
    bearing this marker, and ``synthesis.task_creator`` uses the same marker
    to enumerate setup actions and to surface argument descriptions to the
    LLM prompt. Adding a new helper now only requires decorating it — there
    is no separate registration table to keep in sync.

    Args:
        args:    map of parameter name → human description. Suffix the key
                 with ``"?"`` (e.g. ``"shell?"``) to mark it optional. The
                 LLM prompt renders these as ``args.<name>: <description>``.
        returns: plain-language description of the return value, if any.
        summary: one-line summary; defaults to the function's first
                 docstring line when omitted.
    """

    def deco(fn: F) -> F:
        setattr(fn, SETUP_SCHEMA_ATTR, {
            "role": "setup",
            "config": {},
            "rules": {},
            "options": {},
            "args": dict(args or {}),
            "returns": returns,
            "summary": summary or _first_doc_line(fn.__doc__),
            "source": "decorator",
        })
        return fn

    return deco


def is_setup_action(fn: Callable[..., Any]) -> bool:
    """True iff ``fn`` carries a ``@setup_action`` schema."""
    return isinstance(getattr(fn, SETUP_SCHEMA_ATTR, None), dict)


_HEADER_ONLY_RE = re.compile(
    r"^(Config|Rules|Options|Args|Returns?|Yields?|Raises?|Examples?|Notes?)\s*:\s*$"
)


def _first_doc_line(doc: Optional[str]) -> str:
    """Return the first non-empty narrative line of a docstring.

    Skips lines that are *only* a block header (``Config:`` / ``Args:`` /
    ``Returns:`` / ...) so functions that document inputs but lack a
    proper one-line summary still get a useful summary in the catalog.
    """
    if not doc:
        return ""
    for line in inspect.cleandoc(doc).splitlines():
        s = line.strip()
        if not s:
            continue
        if _HEADER_ONLY_RE.match(s):
            continue
        return s
    return ""


# Header line of a doc block: e.g. "Config:" or "    Rules:"
# We deliberately do NOT match "Args:" / "Returns:" / "Yields:" here:
# - "Args:" documents *function parameters*, not dict-key contents. Files
#   that document dict-key contents under "Args:" mix levels (e.g. the
#   ``rules`` arg description includes a JSON-block prose); routing those
#   through the entry regex misclassifies top-level kwargs as dict keys.
#   Functions whose dict keys aren't already documented under Config:/
#   Rules:/Options: should declare a schema via the ``@evaluator`` decorator.
# - "Returns:" is parsed separately into the schema's ``returns`` field.
_BLOCK_HEADER_RE = re.compile(
    r"^[ \t]*(Config|Rules|Options)\s*:\s*$",
    re.MULTILINE,
)
_RETURNS_HEADER_RE = re.compile(
    r"^[ \t]*Returns?\s*:\s*$",
    re.MULTILINE,
)
# A single key line within a block:
#   "    path (str): absolute path on the VM to fetch"
#   "    shell (bool, optional): defaults to False"
#   "    multi: if True, treat path/dest as lists"
# The type annotation segment is a forgiving "anything except newline /
# colon" chunk so malformed annotations (e.g. mismatched brackets like
# "(str|List[str]))" we have seen in the wild) still parse cleanly.
# Leading whitespace is optional because ``inspect.cleandoc`` strips
# common indentation from docstring bodies; we scope to a single block
# when iterating, so indentation isn't needed as a guard.
_ENTRY_RE = re.compile(
    r"""^[ \t]*
        (?P<name>[A-Za-z_][\w]*)
        (?P<typ>[^\n:]*)
        :[ \t]*
        (?P<desc>.+?)\s*$
    """,
    re.VERBOSE,
)


def parse_docstring_schema(doc: Optional[str]) -> Dict[str, Any]:
    """Extract Config / Rules / Options / Returns blocks from a docstring.

    Returns ``{"config": {...}, "rules": {...}, "options": {...},
    "returns": "..."}``. Each dict value is a ``{key: description}`` map;
    optional keys are suffixed with ``"?"``.
    """
    out: Dict[str, Any] = {
        "config": {},
        "rules": {},
        "options": {},
        "returns": "",
    }
    if not doc:
        return out

    text = inspect.cleandoc(doc)
    headers = list(_BLOCK_HEADER_RE.finditer(text))

    for i, header in enumerate(headers):
        target = header.group(1).lower()
        if target not in out:
            continue

        body_start = header.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end]

        for raw_line in body.splitlines():
            entry = _ENTRY_RE.match(raw_line)
            if not entry:
                continue
            name = entry.group("name")
            desc = entry.group("desc").strip()
            typ = (entry.group("typ") or "").lower()
            is_optional = (
                "optional" in typ
                or "default" in desc.lower()
                or desc.lower().startswith("optional")
            )
            key = f"{name}?" if is_optional else name
            out[target][key] = desc

    # Returns: capture the first non-empty body line as a one-line summary.
    ret_match = _RETURNS_HEADER_RE.search(text)
    if ret_match:
        ret_body = text[ret_match.end():].strip().splitlines()
        for line in ret_body:
            s = line.strip()
            if s:
                out["returns"] = s
                break

    return out


def get_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Return the schema for ``fn`` (decorator output or parsed docstring).

    The result always has the keys ``role``, ``config``, ``rules``,
    ``options``, ``returns``, ``summary``, ``source`` so downstream code
    can read it without defensive checks. ``role`` may be ``""`` when only
    the docstring fallback fired (the parser cannot tell getter vs metric).
    """
    decl = getattr(fn, SCHEMA_ATTR, None)
    if isinstance(decl, dict):
        return decl

    setup_decl = getattr(fn, SETUP_SCHEMA_ATTR, None)
    if isinstance(setup_decl, dict):
        # Setup-action schemas use the same shape as evaluator schemas so
        # downstream renderers (``_fmt_funcs``) can iterate uniformly. The
        # extra ``args`` field is what the setup decorator populates.
        return setup_decl

    doc = inspect.getdoc(fn)
    parsed = parse_docstring_schema(doc)
    has_any = bool(
        parsed["config"] or parsed["rules"]
        or parsed["options"] or parsed["returns"]
    )
    return {
        "role": "",
        "config": parsed["config"],
        "rules": parsed["rules"],
        "options": parsed["options"],
        "returns": parsed["returns"],
        "summary": _first_doc_line(doc),
        "source": "docstring" if has_any else "none",
    }


def required_config_keys(schema: Dict[str, Any]) -> list:
    """Names of *required* (non-``?``-suffixed) keys in the config schema."""
    return [k for k in schema.get("config", {}) if not k.endswith("?")]


def required_rules_keys(schema: Dict[str, Any]) -> list:
    """Names of *required* (non-``?``-suffixed) keys in the rules schema."""
    return [k for k in schema.get("rules", {}) if not k.endswith("?")]
