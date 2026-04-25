"""Task synthesis: LLM-driven generation, static validation, batched loop.

Responsibilities:
  * Discover domains and load reference examples from
    ``refactored_evaluation_examples/examples/``.
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
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from mm_agents.utils.call_llm import call_llm_with_single_response
from mm_agents.utils.utils import parse_json_response

from .prompts import TASK_GEN_SYSTEM, VERIFIER_GEN_SYSTEM
from .shared_memory import SynthesisMemory, VectorDedupStore

logger = logging.getLogger("desktopenv.synthesis.task_creator")

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "refactored_evaluation_examples",
    "examples",
)


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

    logger.info(
        f"Cataloged {len(cat.setup_functions)} setup, "
        f"{len(cat.getter_functions)} getter, "
        f"{len(cat.metric_functions)} metric functions"
    )
    return cat


# ═══════════════════════════════════════════════════════════════════════════
# LLM-driven example + verifier generation
# ═══════════════════════════════════════════════════════════════════════════


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

    logger.info(f"Generating {num_to_generate} task example(s) for '{domain_info.name}' ...")

    raw = call_llm_with_single_response(
        messages=messages, llm_config=llm_config,
        max_tokens=8000, temperature=0.7,
    )

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
    raw = call_llm_with_single_response(
        messages=messages, llm_config=llm_config,
        max_tokens=8000, temperature=0.7,
    )
    parsed = parse_json_response(raw)
    if parsed is None:
        logger.error("Failed to parse verifier response")
        return {"postconfig": [], "eval": ""}
    ev = {"postconfig": parsed.get("postconfig", []), "eval": parsed.get("eval", "")}
    logger.info(f"Verifier eval: {ev['eval'][:120]}")
    return ev


# ═══════════════════════════════════════════════════════════════════════════
# Static validation of generated scripts
# ═══════════════════════════════════════════════════════════════════════════


_EVAL_BUILTINS = {"float", "int", "str", "len", "bool", "abs", "max", "min"}


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_example_scripts(example: Dict[str, Any], catalog: FunctionCatalog) -> ValidationResult:
    """Statically validate the setup config and evaluator scripts of a synthesized example.

    Checks:
      1. Every config / postconfig entry references a known setup function
         and is syntactically valid Python.
      2. The evaluator eval expression parses as Python and only calls known
         getter / metric functions (or a small whitelist of builtins).
    """
    setup_names = {f["name"] for f in catalog.setup_functions}
    eval_allowed = (
        {f["name"] for f in catalog.getter_functions}
        | {f["name"] for f in catalog.metric_functions}
        | _EVAL_BUILTINS
    )
    errors: List[str] = []

    def check_setup(entry: str, label: str) -> None:
        paren = entry.find("(")
        if paren == -1:
            errors.append(f"{label}: not a function call: {entry!r}")
            return
        func_name = entry[:paren].strip()
        if func_name not in setup_names:
            errors.append(f"{label}: unknown setup function: {func_name!r}")
            return
        try:
            ast.parse(entry, mode="eval")
        except SyntaxError as e:
            errors.append(f"{label}: syntax error in setup string: {e}")

    for i, entry in enumerate(example.get("config", [])):
        check_setup(entry, f"config[{i}]")

    evaluator = example.get("evaluator", {})
    for i, entry in enumerate(evaluator.get("postconfig", [])):
        check_setup(entry, f"postconfig[{i}]")

    eval_str = (evaluator.get("eval") or "").strip()
    if not eval_str:
        errors.append("eval: empty eval expression")
    else:
        try:
            tree = ast.parse(eval_str, mode="eval")
            called = [
                child.func.id for child in ast.walk(tree)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            ]
            if not called:
                errors.append("eval: expression contains no function calls")
            else:
                unknown = [n for n in called if n not in eval_allowed]
                if unknown:
                    errors.append(f"eval: unknown function(s): {unknown}")
        except SyntaxError as e:
            errors.append(f"eval: syntax error: {e}")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ═══════════════════════════════════════════════════════════════════════════
# Batched synthesis loop
# ═══════════════════════════════════════════════════════════════════════════


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

    Order of operations per batch:
      1. Generate, validate, dedup, persist accepted examples to disk.
      2. Run verification (via ``on_batch_complete``) on accepted examples.
      3. THEN record every batch outcome — script-invalid, duplicate,
         executable-only, solvable, errored — into memory in a single pass
         and persist it. Verification always happens before any memory
         write so the persisted record reflects the final status.

    ``on_batch_complete`` (optional) returns the verification results for
    the accepted examples so this function can fold them into memory.
    """
    all_domains = discover_domains()
    catalog = catalog_functions()
    targets = args.domains if args.domains else all_domains
    targets = [d for d in targets if d in all_domains]

    llm_config = {"model": args.model, "provider": args.provider, "endpoint": args.endpoint}
    os.makedirs(args.output_dir, exist_ok=True)
    all_examples: List[Dict[str, Any]] = []

    batch_size = args.batch_size if args.batch_size > 0 else args.num_examples
    total_examples = args.total_examples if args.total_examples > 0 else args.num_examples
    max_empty_batches = max(1, args.max_empty_batches)

    for domain in targets:
        logger.info("=" * 60)
        logger.info(
            f"Synthesizing for domain: {domain} "
            f"(target={total_examples}, batch_size={batch_size})"
        )
        domain_info = load_domain_examples(domain, max_examples=args.max_ref_examples)
        ref_evaluators = [ex.get("evaluator", {}) for ex in domain_info.examples]
        domain_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)

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
            logger.info(
                f"Domain '{domain}' batch {batch_idx}: generating {n_this_batch} "
                f"(progress: {existing_valid + len(session_valid)}/{total_examples})"
            )

            examples = generate_task_examples(
                domain_info, catalog, llm_config,
                n_this_batch, args.max_steps, memory,
            )

            for ex in examples:
                if not ex.get("evaluator", {}).get("eval"):
                    ex["evaluator"] = generate_verifier(
                        ex.get("instruction", ""), ex.get("config", []),
                        domain, catalog, ref_evaluators, llm_config,
                    )
                ex.setdefault("_domain", domain)

            # 1) Static validation
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

            # 2) Vector-DB dedup
            duplicates: List[Tuple[Dict[str, Any], Any]] = []
            
            breakpoint()
            
            if vector_store is not None and batch_valid:
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

            # 3) Persist accepted examples to disk (verification needs them)
            for ex in batch_valid:
                path = os.path.join(domain_dir, f"{ex['id']}.json")
                with open(path, "w") as f:
                    json.dump(ex, f, indent=2)
                logger.info(f"Saved {path}")

            # 4) Verification first — gather results before touching memory
            verify_results: Optional[List[Dict[str, Any]]] = None
            if on_batch_complete is not None and batch_valid:
                try:
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

            # 5) Memory record + persist (single pass, after verification)
            if memory is not None:
                for ex, err in statically_invalid:
                    memory.record(
                        example=ex, domain=domain,
                        code_result={"score": 0.0, "error": f"script_validation: {err}"},
                        executable=False,
                    )
                for ex, match in duplicates:
                    memory.record(
                        example=ex, domain=domain,
                        code_result={
                            "score": 0.0,
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
                            code_result={"score": 0.0},
                            executable=True, solvable=None,
                        )
                    else:
                        solvable = None if "error" in r else (r.get("score", 0) > 0)
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

        all_examples.extend(session_valid)

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
