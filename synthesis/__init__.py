"""OSWorld synthesis pipeline.

Package layout:
    cli.py            - entrypoint: argparse + dispatcher + signal handling.
    prompts.py        - LLM system prompts.
    shared_memory.py  - SynthesisMemory (JSON) + VectorDedupStore (ChromaDB)
                        + DedupHistory (per-domain rolling rejection log).
    utils.py          - domain/function discovery, prompt formatters, and
                        the static validator (syntax + signature + setup
                        sequence + eval-role checks).
    task_creator.py   - LLM-driven example generation and the batched
                        synthesis loop (sequential/parallel dispatchers).
    verifier.py       - API client, code-execution worker, parallel and
                        sequential verify dispatchers, result processing.
"""
