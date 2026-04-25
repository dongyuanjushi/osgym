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
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

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
            "code_score": code_score,
            "code_steps": code_result.get("steps") if code_result else None,
            "executable": executable,
            "solvable": solvable,
            "failure_reasons": failure_reasons
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

    def format_for_prompt(self, domain: str, max_entries: int = 500) -> str:
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
    """Canonical text used to embed a task: instruction + verifier signature.

    Including the evaluator expression means two tasks whose instructions read
    similarly but whose verifiers check different state (e.g. different files
    or keys) are treated as distinct.
    """
    instr = (instruction or "").strip()
    ev = (evaluator_eval or "").strip()
    if ev:
        return f"{instr}\n\nVERIFIER: {ev}"
    return instr


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass
class SimilarMatch:
    id: str
    instruction: str
    similarity: float
    source: str  # "db" or "intra_batch"
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
        similarity_threshold: float = 0.88,
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
        """Drop examples whose nearest neighbor (in DB or earlier in this
        batch) exceeds the similarity threshold. Embeddings are computed in a
        single batched call so the network overhead is one round-trip."""
        if not examples:
            return DedupDecision()

        texts = [
            _task_text(
                ex.get("instruction", ""),
                (ex.get("evaluator") or {}).get("eval", ""),
            )
            for ex in examples
        ]

        try:
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.warning(
                f"[vector-dedup] batch embedding failed: {e} - skipping dedup for this batch"
            )
            return DedupDecision(accepted=list(examples))

        col = self._collection(domain)
        have_db = False
        try:
            have_db = col.count() > 0
        except Exception as e:
            logger.warning(f"[vector-dedup] collection count failed: {e}")

        decision = DedupDecision()
        accepted_embs: List[List[float]] = []

        for ex, emb in zip(examples, embeddings):
            # 1) Check against persisted solvable set
            db_match = self._query_db(col, emb) if have_db else None
            if db_match is not None:
                decision.rejected.append((ex, db_match))
                continue

            # 2) Check against examples already accepted in this batch
            intra_match = self._check_intra_batch(emb, decision.accepted, accepted_embs)
            if intra_match is not None:
                decision.rejected.append((ex, intra_match))
                continue

            decision.accepted.append(ex)
            accepted_embs.append(emb)

        return decision

    def _query_db(self, col, emb: List[float]) -> Optional[SimilarMatch]:
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
            source="db",
            metadata=meta,
        )

    def _check_intra_batch(
        self,
        emb: List[float],
        accepted: List[Dict[str, Any]],
        accepted_embs: List[List[float]],
    ) -> Optional[SimilarMatch]:
        for prev_ex, prev_emb in zip(accepted, accepted_embs):
            sim = _cosine(emb, prev_emb)
            if sim >= self.similarity_threshold:
                return SimilarMatch(
                    id=prev_ex.get("id", ""),
                    instruction=prev_ex.get("instruction", ""),
                    similarity=sim,
                    source="intra_batch",
                )
        return None

    # -- writes -------------------------------------------------------------

    def add_solvable(self, example: Dict[str, Any], domain: str) -> bool:
        eid = example.get("id")
        if not eid:
            logger.warning("[vector-dedup] skipping add: example has no id")
            return False

        instruction = example.get("instruction", "") or ""
        evaluator_eval = (example.get("evaluator") or {}).get("eval", "") or ""
        text = _task_text(instruction, evaluator_eval)

        try:
            emb = self.embedder.embed([text])[0]
        except Exception as e:
            logger.warning(f"[vector-dedup] embedding failed during add: {e}")
            return False

        try:
            self._collection(domain).upsert(
                ids=[eid],
                embeddings=[emb],
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
    similarity_threshold: float = 0.88,
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
