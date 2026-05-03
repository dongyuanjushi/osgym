"""Task synthesis main workflow: LLM-driven generation + batched loop.

Responsibilities:
  * Generate new task examples + verifiers via LLM calls
    (``generate_task_examples``).
  * Drive the batched synthesis loop (``run_synthesize``), including
    vector-DB dedup and writing validated examples to disk.

Domain/function discovery, prompt formatting, identity stamping, and the
static validator live in ``utils.py`` — this module pulls them in but does
not own them, keeping ``task_creator.py`` focused on the generation workflow.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

from mm_agents.utils.call_llm import call_llm_with_single_response
from mm_agents.utils.utils import parse_json_response

from .prompts import TASK_GEN_SYSTEM
from .shared_memory import DedupHistory, SynthesisMemory, VectorDedupStore
from .utils import (
    DOMAIN_GETTER_MODULES,
    DOMAIN_METRIC_MODULES,
    EXAMPLES_DIR,
    SHARED_GETTER_MODULES,
    SHARED_METRIC_MODULES,
    DomainInfo,
    FunctionCatalog,
    _filter_for_domain,
    _fmt_funcs,
    _stamp_synthesized_examples,
    catalog_functions,
    discover_domains,
    extend_catalog_with_extras,
    load_domain_examples,
    referenced_function_names,
    validate_example_scripts,
    verify_extra_function_executable,
)

logger = logging.getLogger("desktopenv.synthesis.task_creator")


# Maximum LLM attempts per ``generate_task_examples`` call. The endpoint can
# blip (transport errors, malformed JSON, lists of strings instead of dicts),
# and a single batch failure shouldn't stall the whole synthesis loop — the
# outer ``empty_streak`` already kicks in if every attempt yields nothing.
_LLM_GENERATION_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════
# LLM-driven example + verifier generation
# ═══════════════════════════════════════════════════════════════════════════


def generate_task_examples(
    domain_info: DomainInfo,
    catalog: FunctionCatalog,
    llm_config: Dict[str, Any],
    num_to_generate: int = 1,
    max_steps: int = 15,
    memory: Optional[SynthesisMemory] = None,
    memory_block: Optional[str] = None,
    max_ref_examples: int = 5,
    dedup_block: Optional[str] = None
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

    ``dedup_block`` is a per-domain block listing the duplicates that
    ``VectorDedupStore.filter_batch`` has rejected for this domain across
    earlier batches. The history is stored as a set so identical
    rejection messages collapse automatically, and is capped at
    ``DedupHistory.max_per_domain`` (default 100). Surfacing those
    rejections to the LLM discourages it from re-proposing the same
    near-duplicates this round. Empty/None means nothing has been
    rejected for this domain yet.
    """
    # Curated demos are the LLM's primary source of correct call patterns —
    # they showcase how to pair file-creation setup with `_open_setup`, how
    # to populate `get_vm_file(env, config={'path':..., 'dest':...})` with
    # both required keys, and how to emit rules dicts that match the metric's
    # schema. Fall back to a slice of the legacy curated OSWorld examples if
    # no demo file exists for the domain so the prompt isn't empty.
    domain_info = load_domain_examples(domain_info.name)
    ref = json.dumps(random.sample(domain_info.examples, max_ref_examples), indent=2)
    if memory_block is None:
        memory_block = memory.format_for_prompt(domain_info.name) if memory is not None else ""
    if memory_block:
        memory_block = f"\n{memory_block}\n"

    dedup_section = f"\n{dedup_block}\n" if dedup_block else ""

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
        f"Keep tasks focused — each should explore one aspect of the software and "
        f"produce a clear, verifiable outcome.\n\n"
        f"## Reference examples (each shows a complete task with its evaluator)\n{ref}\n\n"
        f"## Setup functions\n{_fmt_funcs(catalog.setup_functions)}\n\n"
        f"## Getter functions\n{_fmt_funcs(domain_getters)}\n\n"
        f"## Metric functions\n{_fmt_funcs(domain_metrics)}\n\n"
        # f"{memory_block}"
        f"{dedup_section}"
        f"Generate {num_to_generate} new, diverse task example(s) for "
        f"\"{domain_info.name}\".\n"
        f"Each task must:\n"
        f"  - target a different feature or setting of the application,\n"
        f"  - produce a distinct, observable state change, AND\n"
        f"  - ship with a fully-populated `evaluator` object containing both "
        f"`postconfig` (list of setup-call strings) and a non-empty `eval` "
        f"expression that follows the verifier-strength rubric in the system "
        f"prompt.\n"
        f"Return a JSON array of task objects. "
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
        try:
            temperature = random.uniform(0.8, 1.0)
            raw = call_llm_with_single_response(
                messages=messages, llm_config=llm_config,
                max_tokens=8000, temperature=temperature
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
        logger.info(f"LLM response: {len(raw)} chars")

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
# Extra-function orchestration
#
# When the LLM cannot find a setup/getter/metric in the catalog, the
# TASK_GEN prompt allows it to declare its own helpers via the optional
# ``extra_functions`` array. Each entry must (a) be referenced from
# ``config`` / ``evaluator.postconfig`` / ``evaluator.eval``, and (b)
# carry an executable implementation. ``_resolve_extra_functions`` runs
# both checks for one example, returning an extended catalog the static
# validator can resolve against. Errors from this stage fold into the
# same statically-invalid bucket as syntax / signature failures so they
# land in synthesis memory with the same provenance.
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_extra_functions(
    example: Dict[str, Any], catalog: FunctionCatalog
) -> Tuple[Optional[FunctionCatalog], List[str]]:
    """Validate ``extra_functions`` and merge them into the catalog.

    Returns ``(extended_catalog, errors)``:

    * If the example has no ``extra_functions``, returns ``(catalog, [])``
      unchanged so the caller can use the same code path for every example.
    * If any check fails (shape, missing reference, name shadowing, exec
      failure, signature unintrospectable), returns ``(None, errors)`` and
      the caller should treat the example as statically invalid.
    * Otherwise returns the extended catalog with each verified extra grafted
      into ``setup_functions`` / ``getter_functions`` / ``metric_functions``
      based on its declared ``kind``.

    The match check is bidirectional in spirit but only one direction is
    enforced here: every declared extra MUST be referenced. The reverse
    direction (every referenced unknown name has a declaration) is left to
    ``validate_example_scripts``, which already emits a precise "unknown
    function" error for any catalog miss in eval.
    """
    extras_raw = example.get("extra_functions")
    if not extras_raw:
        return catalog, []

    if not isinstance(extras_raw, list):
        return None, [
            f"extra_functions: must be a list of function-spec dicts, got "
            f"{type(extras_raw).__name__}"
        ]
    if not extras_raw:
        return catalog, []

    errors: List[str] = []
    referenced = referenced_function_names(example)
    catalog_names = {
        f["name"] for f in catalog.setup_functions
    } | {
        f["name"] for f in catalog.getter_functions
    } | {
        f["name"] for f in catalog.metric_functions
    }

    seen: set = set()
    verified: List[Tuple[Dict[str, Any], Any]] = []
    for i, spec in enumerate(extras_raw):
        # Resolve a label up front so error messages stay readable even when
        # the spec is malformed (e.g. missing 'name').
        spec_name = spec.get("name") if isinstance(spec, dict) else None
        label = repr(spec_name) if isinstance(spec_name, str) and spec_name else f"index {i}"

        # 1) Match check — declared name must be referenced from the example.
        if isinstance(spec_name, str) and spec_name.strip():
            if spec_name not in referenced:
                errors.append(
                    f"extra_functions[{label}]: declared but never called "
                    f"from config, evaluator.postconfig, or evaluator.eval — "
                    f"either drop the declaration or reference the function "
                    f"in the example"
                )
            if spec_name in catalog_names:
                errors.append(
                    f"extra_functions[{label}]: shadows an existing "
                    f"setup/getter/metric in the catalog — reference the "
                    f"catalog version directly and remove this entry"
                )
                continue
            if spec_name in seen:
                errors.append(
                    f"extra_functions[{label}]: declared more than once in "
                    f"this example"
                )
                continue
            seen.add(spec_name)

        # 2) Execution check — implementation must compile, exec, and produce
        #    a callable matching the declared name.
        fn, exec_errors = verify_extra_function_executable(spec)
        if exec_errors:
            errors.extend(exec_errors)
            continue
        verified.append((spec, fn))

    if errors:
        return None, errors

    return extend_catalog_with_extras(catalog, verified), []


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
    dedup_history: Optional[DedupHistory] = None,
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
    max_ref_examples = getattr(args, "max_ref_examples", 5)

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

        # Pull THIS domain's accumulated dedup-rejection block. The history
        # is a per-domain set so rejections from one app don't bleed into
        # another and identical instructions are deduped automatically,
        # and capped at ``DedupHistory.max_per_domain`` (default 100) so
        # the prompt stays bounded even on long runs. The class is
        # internally locked, but we additionally hold the write_lock so
        # the snapshot can't change mid-build of the prompt.
        dedup_block: Optional[str] = None
        if dedup_history is not None:
            with hold():
                dedup_block = dedup_history.format_for_prompt(domain)
                # breakpoint()

        examples = generate_task_examples(
            domain_info, catalog, llm_config,
            n_this_batch, args.max_steps,
            memory=None, memory_block=memory_block,
            dedup_block=dedup_block,
            max_ref_examples=max_ref_examples
        )

        # 1) Static validation (pure CPU work — runs outside the lock).
        #    Per example: first resolve any ``extra_functions`` (LLM-declared
        #    setup/getter/metric helpers) into an extended catalog so the
        #    static validator can bind references to them. Extra-function
        #    failures (missing reference, exec error, signature unbindable)
        #    are recorded as static failures with the same provenance as
        #    syntax/signature errors.
        batch_valid: List[Dict[str, Any]] = []
        statically_invalid: List[Tuple[Dict[str, Any], str]] = []
        for ex in examples:
            ext_catalog, extra_errors = _resolve_extra_functions(ex, catalog)
            if extra_errors:
                err = "; ".join(extra_errors)
                logger.warning(
                    f"EXTRA FUNCTIONS INVALID for {ex.get('id', '?')}: "
                    f"{err} – skipping"
                )
                statically_invalid.append((ex, err))
                continue
            vr = validate_example_scripts(ex, ext_catalog)
            if vr.valid:
                batch_valid.append(ex)
            else:
                err = "; ".join(vr.errors)
                logger.warning(
                    f"SCRIPT VALIDATION FAILED for {ex.get('id', '?')}: {err} – skipping"
                )
                statically_invalid.append((ex, err))

        # 2) Vector-DB dedup. ``filter_batch`` queries the persistent Chroma
        #    collection for each example's nearest neighbour and upserts
        #    accepted ones immediately, so a duplicate emitted later in the
        #    same batch is caught via the same DB query (no separate
        #    in-memory intra-batch pass). The whole call runs under the
        #    write_lock so concurrent worker threads see consistent reads
        #    and the per-store collection cache stays coherent.
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
                        f"sim={match.similarity:.3f} "
                        f"-> {str(match.id)[:8]} ({(match.instruction or '')[:80]!r})"
                    )
            batch_valid = decision.accepted
            duplicates = decision.rejected

        # Add THIS batch's dedup rejections to this domain's rolling
        # history. ``extend`` calls ``set.add()`` per message so repeats
        # from earlier batches are silently absorbed, and the set is
        # trimmed to the cap inside ``extend`` so the prompt stays
        # bounded even on long runs. No-op when the batch had no
        # rejections (nothing to add).
        if dedup_history is not None and duplicates:
            messages = [
                match.instruction for ex, match in duplicates
            ]
            with hold():
                dedup_history.extend(domain, messages)

        # 3) Persist accepted examples to disk (unique filenames per id, so
        #    no cross-thread file collision — runs outside the lock).
        for ex in batch_valid:
            path = os.path.join(domain_dir, f"{ex['id']}.json")
            with open(path, "w") as f:
                json.dump(ex, f, indent=2)
            logger.info(f"Saved {path}")

        # 4) Verification first — gather results before touching memory.
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

        # 5) Memory record + persist (single pass, after verification).
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
                            "error": f"duplicate_of={match.id} sim={match.similarity:.3f}",
                        },
                        executable=False,
                    )
                results_by_id = {
                    r["id"]: r for r in (verify_results or []) if r.get("id")
                }
                for ex in batch_valid:
                    r = results_by_id.get(ex["id"])
                    if r is None:
                        memory.record(
                            example=ex, domain=domain,
                            code_result={"score": -1},
                            executable=True, solvable=None,
                        )
                    else:
                        if "error" in r:
                            solvable = None
                        elif r.get("mode") == "relevance_skip":
                            # Evaluator did not match the instruction's intent;
                            # the VM was never touched, so this is a hard
                            # not-solvable rather than an unknown.
                            solvable = False
                        else:
                            solvable = r.get("score", 0) > 0
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
    dedup_history: Optional[DedupHistory] = None,
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
            max_empty_batches, dedup_history=dedup_history, write_lock=None,
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
    dedup_history: Optional[DedupHistory] = None,
) -> List[Dict[str, Any]]:
    """Parallel synthesis: a thread pool runs one domain per worker.

    Threads are appropriate here because the per-batch hot path is dominated
    by an I/O-bound LLM call (``generate_task_examples``); the synthesis
    bookkeeping is serialized via a single ``write_lock`` shared by every
    worker.
    """
    synthesize_workers = max(1, min(getattr(args, "synthesize_workers", 1), len(targets)))
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
                max_empty_batches, dedup_history, write_lock,
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

    # Per-domain rolling history of dedup rejections, stored as a set so
    # repeated rejections collapse automatically (capped at 50 entries per
    # domain). Each domain worker adds its own batch's rejections via
    # ``set.add()``; the next batch in that same domain reads the
    # accumulated set and passes it to the LLM so the model sees every
    # distinct near-duplicate already flagged for this domain.
    dedup_history = DedupHistory(max_per_domain=200)

    synthesize_mode = getattr(args, "synthesize_mode", "sequential")
    if synthesize_mode == "parallel":
        all_examples = _run_synthesize_parallel(
            args, memory, vector_store, on_batch_complete, targets,
            catalog, llm_config, batch_size, total_examples, max_empty_batches,
            dedup_history=dedup_history
        )
    else:
        all_examples = _run_synthesize_sequential(
            args, memory, vector_store, on_batch_complete, targets,
            catalog, llm_config, batch_size, total_examples, max_empty_batches,
            dedup_history=dedup_history
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
