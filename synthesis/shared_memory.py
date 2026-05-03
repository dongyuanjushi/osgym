"""Persistence layer for the synthesis pipeline.

Two stores live here:

- ``SynthesisMemory`` - JSON-backed log of every task the pipeline has
  attempted in this output directory. Records static-validation outcomes
  and verification verdicts so subsequent rounds can avoid retrying
  already-resolved tasks and can surface prior experience in the LLM
  prompt.

- ``VectorDedupStore`` - local ChromaDB (PersistentClient) holding
  embeddings of tasks that have passed verification. Used to reject
  near-duplicates during synthesis before they're sent to a VM. Embeddings
  are produced by an OpenAI-compatible ``/v1/embeddings`` endpoint, which
  means any vLLM (``--task embed``) or sglang (``--is-embedding``) server
  works unchanged.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

import random

logger = logging.getLogger("desktopenv.synthesis.shared_memory")


# ═══════════════════════════════════════════════════════════════════════════
# SynthesisMemory - JSON-backed experience log
# ═══════════════════════════════════════════════════════════════════════════


class SynthesisMemory:
    """JSON-backed memory of past synthesis rounds.

    Each entry records one task's synthesis + verification outcome so that
    future rounds can:
      - avoid regenerating similar tasks,
      - avoid patterns that consistently fail,
      - steer toward unexplored UI areas.

    Entry lifecycle:
      executable=False              -> script failed static validation
      executable=True, solvable=False -> valid scripts but agent execution failed
      executable=True, solvable=True  -> fully verified via agent execution

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
            # "executable"/"solvable", and the unified "failure_reasons"
            # field instead of the split execution/verification reasons.
            for e in self.entries:
                if "executable" not in e and "verified" in e:
                    e["executable"] = e.pop("verified")
                if "solvable" not in e:
                    e["solvable"] = False
                if "failure_reasons" in e:
                    legacy = e.pop("failure_reasons") or []
                    if not e.get("executable"):
                        e.setdefault("execution_failure_reasons", []).extend(legacy)
                    elif e.get("solvable") is False:
                        e.setdefault("verification_failure_reasons", []).extend(legacy)
                e.setdefault("execution_failure_reasons", [])
                e.setdefault("verification_failure_reasons", [])
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
        error = (code_result or {}).get("error")

        # Build detailed, actionable failure reasons. ``error`` carries the
        # specific diagnosis (e.g. static-validation message, dedup match,
        # or a raised exception from /step). Surfacing it here turns the
        # generic "scripts not executable" line into something the next
        # synthesis round and a human reviewer can both act on.
        execution_failure_reasons: List[str] = []
        verification_failure_reasons: List[str] = []
        if not executable:
            if error:
                execution_failure_reasons.append(
                    f"Synthesized example failed pre-VM static validation: {error}"
                )
            else:
                execution_failure_reasons.append(
                    "Synthesized scripts (setup/evaluator) failed static "
                    "validation before the VM was touched"
                )
        else:
            if error:
                verification_failure_reasons.append(
                    f"Verification raised an error while running on the VM: {error}"
                )
            elif (code_result or {}).get("mode") == "relevance_skip":
                rel_reason = (code_result or {}).get("relevance_reason") or "no reason given"
                verification_failure_reasons.append(
                    f"Evaluator does not capture the instruction's intent — "
                    f"skipped before allocating a VM ({rel_reason})"
                )
            elif solvable is False:
                verification_failure_reasons.append(
                    "Verification ran without errors but the resulting state "
                    f"did not satisfy the verifier (score={code_score})"
                )

        entry = {
            "id": example.get("id", ""),
            "domain": domain,
            "instruction": example.get("instruction", ""),
            "evaluator_eval": example.get("evaluator", {}).get("eval", ""),
            "executable": executable,
            "execution_failure_reasons": execution_failure_reasons,
            "solvable": solvable,
            "verification_failure_reasons": verification_failure_reasons
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

    def format_for_prompt(self, domain: str, max_entries: int = 50) -> str:
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
        # lines.append(
        #     f"## Past synthesis experience for \"{domain}\" "
        #     f"({len(solvable)} solvable, {len(executable_only)} executable-only, "
        #     f"{len(not_executable)} not-executable)\n"
        # )

        # --- Solvable tasks: show instruction so LLM avoids duplicates ---
        # breakpoint()
        
        if solvable:
            lines.append("### Previously SOLVABLE tasks (do NOT generate similar ones):")
            for e in random.sample(solvable, min(max_entries, len(solvable))):
                lines.append(f"  - \"{e['instruction']}\"")
            lines.append("")

        # --- Executable but not solvable: valid scripts, agent failed ---
        if executable_only:
            lines.append(
                "### Previously EXECUTABLE tasks (The scripts are valid for reference but do not generate similar tasks):"
            )
            for e in random.sample(executable_only, min(max_entries, len(executable_only))):
                # reasons = "; ".join(
                #     e.get("verification_failure_reasons") or ["unknown"]
                # )
                lines.append(
                    f"  - \"{e['instruction']}\"\n"
                    f"    evaluator: {e.get('evaluator_eval', 'n/a')}\n"
                    # f"    verification_failure: {reasons}"
                )
            lines.append("")

        # --- Not executable: invalid scripts ---
        if not_executable:
            lines.append(
                "### Previously NOT EXECUTABLE tasks (invalid scripts — learn from the mistakes):"
            )
            for e in random.sample(not_executable, min(max_entries, len(not_executable))):
                reasons = "; ".join(
                    e.get("execution_failure_reasons") or ["unknown"]
                )
                lines.append(
                    f"  - \"{e['instruction']}\"\n"
                    f"    evaluator: {e.get('evaluator_eval', 'n/a')}\n"
                    f"    execution_failure: {reasons}"
                )
            lines.append("")

        # --- Aggregate failure patterns from all non-solvable entries ---
        # failed_entries = executable_only + not_executable
        # if failed_entries:
        #     failure_keywords: Dict[str, int] = {}
        #     for e in failed_entries:
        #         reasons = (
        #             (e.get("execution_failure_reasons") or [])
        #             + (e.get("verification_failure_reasons") or [])
        #         )
        #         for r in reasons:
        #             if "error=" in r:
        #                 err = r.split("error=")[-1][:60]
        #             else:
        #                 err = r[:60]
        #             failure_keywords[err] = failure_keywords.get(err, 0) + 1
        #     if failure_keywords:
        #         lines.append("### Common failure patterns (avoid these):")
        #         for kw, cnt in sorted(failure_keywords.items(), key=lambda x: -x[1])[:10]:
        #             lines.append(f"  - ({cnt}x) {kw}")
        #         lines.append("")

        # # --- Coverage summary ---
        # all_instructions = [e["instruction"] for e in domain_entries]
        # if all_instructions:
        #     lines.append(
        #         f"### Coverage: {len(all_instructions)} tasks generated so far. "
        #         f"Explore UI areas and menu paths NOT yet covered above.\n"
        #     )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# VectorDedupStore - ChromaDB-backed similarity store for solvable tasks
# ═══════════════════════════════════════════════════════════════════════════


class OpenAICompatEmbedder:
    """Call an OpenAI-compatible /v1/embeddings endpoint.

    Compatible with vLLM (``--task embed``) and sglang (``--is-embedding``),
    plus any other server that implements the OpenAI embeddings REST contract.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        inputs = [t if (t and t.strip()) else " " for t in texts]
        resp = requests.post(
            f"{self.endpoint}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": inputs},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return [row["embedding"] for row in payload["data"]]


def _task_text(instruction: str, evaluator_eval: str = "") -> str:
    """Canonical text used to embed a task: instruction only.

    Earlier revisions concatenated the verifier expression so tasks with
    similar instructions but different state checks were treated as distinct.
    That made dedup too lenient — near-paraphrased instructions slipped
    through whenever the verifier wording differed. Embedding the instruction
    alone keeps the catalog focused on what the user is actually asked to do;
    the ``evaluator_eval`` argument is accepted for backwards-compatible
    callers but ignored.
    """
    return (instruction or "").strip()


@dataclass
class SimilarMatch:
    id: str
    instruction: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DedupDecision:
    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Tuple[Dict[str, Any], SimilarMatch]] = field(default_factory=list)


class VectorDedupStore:
    """Per-domain Chroma collections holding embeddings of solvable tasks."""

    _COLLECTION_PREFIX = "osgym_tasks_"

    def __init__(
        self,
        persist_path: str,
        embedder: OpenAICompatEmbedder,
        similarity_threshold: float = 0.7,
    ):
        import chromadb  # local import so the module stays optional

        os.makedirs(persist_path, exist_ok=True)
        self.persist_path = persist_path
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.client = chromadb.PersistentClient(path=persist_path)
        self._collections: Dict[str, Any] = {}

    # -- internal -----------------------------------------------------------

    def _collection(self, domain: str):
        key = domain or "default"
        col = self._collections.get(key)
        if col is None:
            col = self.client.get_or_create_collection(
                name=f"{self._COLLECTION_PREFIX}{key}",
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[key] = col
        return col

    # -- queries ------------------------------------------------------------

    def filter_batch(
        self,
        domain: str,
        examples: List[Dict[str, Any]],
    ) -> DedupDecision:
        """Reject examples whose nearest neighbor in the persistent store
        exceeds ``self.similarity_threshold``, and upsert each accepted
        example into the store immediately so the next example in the same
        batch (and any subsequent run) dedups against it.

        Earlier revisions ran a second, in-memory ``_check_intra_batch``
        pass to catch near-duplicates produced inside a single LLM call.
        That approach was both redundant and incorrect: the source of truth
        is the Chroma collection, and any two passes that don't go through
        it can drift (e.g. the in-memory pass had no awareness of records
        added by sibling threads or earlier runs). Upserting per-example
        below collapses both checks into the DB query.

        Embeddings for the whole batch are computed in one call so the
        network overhead is a single round-trip, then each example is
        checked-and-upserted serially against the persistent collection.
        """
        if not examples:
            return DedupDecision()

        texts = [_task_text(ex.get("instruction", "")) for ex in examples]

        try:
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.warning(
                f"[vector-dedup] batch embedding failed: {e} - skipping dedup for this batch"
            )
            return DedupDecision(accepted=list(examples))

        col = self._collection(domain)
        decision = DedupDecision()

        for ex, emb, text in zip(examples, embeddings, texts):
            match = self._query_db(col, emb)
            if match is not None:
                decision.rejected.append((ex, match))
                continue

            # Upsert immediately so the next iteration's _query_db sees this
            # record and rejects near-duplicates within the same batch via
            # the same code path used for cross-run dedup. A failure here
            # only forfeits future dedup of this id — we still accept the
            # example so the caller can persist it to disk.
            self._upsert(col, ex, emb, text, domain)
            decision.accepted.append(ex)

        return decision

    def _query_db(self, col, emb: List[float]) -> Optional[SimilarMatch]:
        """Return the nearest persisted example whose similarity exceeds the
        configured threshold, or ``None`` if no such record exists.

        ``col.query`` on an empty collection returns empty result lists, so
        this function naturally handles the cold-start case without a
        separate ``count() > 0`` guard.
        """
        try:
            res = col.query(
                query_embeddings=[emb],
                n_results=1,
                include=["metadatas", "distances", "documents"],
            )
        except Exception as e:
            logger.warning(f"[vector-dedup] DB query failed: {e}")
            return None

        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return None
        dist = float((res.get("distances") or [[0.0]])[0][0])
        sim = 1.0 - dist  # Chroma cosine distance -> similarity
        if sim < self.similarity_threshold:
            return None
        meta = ((res.get("metadatas") or [[{}]])[0][0]) or {}
        doc = ((res.get("documents") or [[""]])[0][0]) or ""
        return SimilarMatch(
            id=ids[0],
            instruction=meta.get("instruction") or doc,
            similarity=sim,
            metadata=meta,
        )

    def _upsert(
        self,
        col,
        example: Dict[str, Any],
        embedding: List[float],
        text: str,
        domain: str,
    ) -> bool:
        """Persist a single example's embedding and metadata to ``col``.

        Idempotent — repeated calls for the same id overwrite the existing
        row. Returns ``False`` if the example has no id or the upsert call
        raises (the caller still accepts the example in the latter case so
        a transient Chroma failure doesn't drop work).
        """
        eid = example.get("id")
        if not eid:
            logger.warning("[vector-dedup] skipping upsert: example has no id")
            return False
        instruction = example.get("instruction", "") or ""
        evaluator_eval = (example.get("evaluator") or {}).get("eval", "") or ""
        try:
            col.upsert(
                ids=[eid],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "id": eid,
                    "domain": domain,
                    "instruction": instruction,
                    "evaluator_eval": evaluator_eval,
                }],
            )
        except Exception as e:
            logger.warning(f"[vector-dedup] upsert failed for {eid}: {e}")
            return False
        return True

    # -- writes -------------------------------------------------------------

    def add_solvable(self, example: Dict[str, Any], domain: str) -> bool:
        """Upsert ``example`` into the per-domain dedup collection.

        ``filter_batch`` already upserts every accepted example, so this
        path is mostly redundant for the synthesize→verify flow; it remains
        for callers (e.g. the standalone verify mode) that want to register
        an example without going through ``filter_batch``. Idempotent —
        repeated calls for the same id overwrite the row.
        """
        text = _task_text(example.get("instruction", "") or "")
        try:
            emb = self.embedder.embed([text])[0]
        except Exception as e:
            logger.warning(f"[vector-dedup] embedding failed during add: {e}")
            return False
        return self._upsert(self._collection(domain), example, emb, text, domain)

    def count(self, domain: Optional[str] = None) -> int:
        if domain is not None:
            try:
                return self._collection(domain).count()
            except Exception:
                return 0
        total = 0
        try:
            for item in self.client.list_collections():
                name = item if isinstance(item, str) else getattr(item, "name", None)
                if not name:
                    continue
                try:
                    total += self.client.get_collection(name=name).count()
                except Exception:
                    continue
        except Exception:
            pass
        return total


def build_vector_store(
    persist_path: str,
    endpoint: str,
    model: str,
    similarity_threshold: float = 0.7,
    api_key: Optional[str] = None,
) -> Optional[VectorDedupStore]:
    """Build a VectorDedupStore after smoke-testing the embedding endpoint.

    Returns None (and logs) on any failure, so the caller can keep running
    without dedup rather than crashing the whole synthesis run.
    """
    try:
        embedder = OpenAICompatEmbedder(endpoint=endpoint, model=model, api_key=api_key)
        vec = embedder.embed(["osgym-dedup-smoketest"])
        if not vec or not vec[0]:
            raise RuntimeError("embedding endpoint returned empty vector")
    except Exception as e:
        logger.error(
            f"[vector-dedup] embedding endpoint smoke test failed "
            f"(endpoint={endpoint} model={model}): {e}. Dedup disabled."
        )
        return None

    try:
        store = VectorDedupStore(
            persist_path=persist_path,
            embedder=embedder,
            similarity_threshold=similarity_threshold,
        )
    except Exception as e:
        logger.error(f"[vector-dedup] failed to initialize Chroma store at {persist_path}: {e}. Dedup disabled.")
        return None

    logger.info(
        f"[vector-dedup] initialized at {persist_path} "
        f"(model={model}, threshold={similarity_threshold}, "
        f"existing entries across all domains={store.count()})"
    )
    return store


# ═══════════════════════════════════════════════════════════════════════════
# DedupHistory - per-domain rolling set of dedup-rejection messages
# ═══════════════════════════════════════════════════════════════════════════


class DedupHistory:
    """Thread-safe per-domain set of dedup-rejection messages.

    Every batch ``extend``s its own domain's set with the rejection
    messages produced by ``VectorDedupStore.filter_batch``; each message
    is inserted via ``set.add()`` so repeats from one batch to the next
    are silently collapsed and the set only ever grows by *unique*
    instructions. The set is capped at ``max_per_domain`` entries
    (default 100); when the cap is exceeded, arbitrary elements are
    popped to bound memory and prompt length. The next batch for that
    domain reads ``format_for_prompt(domain)`` and surfaces the
    accumulated set to the LLM so the model can see every distinct
    near-duplicate it should avoid emitting again.

    Each domain has its own set; rejections from one domain do not bleed
    into another. The instruction text itself is stored verbatim and is
    domain-agnostic, so the prompt block contains no domain tags.
    """

    def __init__(self, max_per_domain: int = 100):
        if max_per_domain <= 0:
            raise ValueError("max_per_domain must be positive")
        self.max_per_domain = max_per_domain
        self._by_domain: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

    def extend(self, domain: str, messages: List[str]) -> None:
        """Add ``messages`` to ``domain``'s set via ``set.add()``.

        Each message is inserted individually so duplicates already in
        the set are no-ops. When the resulting set exceeds the cap,
        arbitrary elements are popped until it fits.
        """
        if not messages:
            return
        with self._lock:
            buf = self._by_domain.setdefault(domain, set())
            for m in messages:
                buf.add(m)
            while len(buf) > self.max_per_domain:
                buf.pop()

    def snapshot(self, domain: str) -> Set[str]:
        """Return a copy of ``domain``'s set (safe to render outside the lock)."""
        with self._lock:
            return set(self._by_domain.get(domain, set()))

    def format_for_prompt(self, domain: str) -> str:
        """Render ``domain``'s set as a user-prompt block; empty when none."""
        snap = self.snapshot(domain)
        if not snap:
            return ""
        lines = [
            "Avoid generating the tasks in the following list: "
        ]
        for m in snap:
            lines.append(f"  - {m}")
        lines.append("Pick a different feature, menu path, or observable state change to generate tasks. ")
        
        return "\n".join(lines) + "\n"
