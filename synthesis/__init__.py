"""OSWorld synthesis pipeline.

Package layout:
    cli.py            - entrypoint: argparse + dispatcher + signal handling.
    prompts.py        - LLM system prompts.
    shared_memory.py  - SynthesisMemory (JSON) + VectorDedupStore (ChromaDB).
    task_creator.py   - domain/function discovery, LLM generation, static
                        validation, batched synthesis loop.
    verifier.py       - API client, code-execution worker, parallel and
                        sequential verify dispatchers, result processing.
"""
