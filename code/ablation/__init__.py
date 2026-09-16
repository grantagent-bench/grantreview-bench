"""Compute-matched ablation harness for the GrantAgent controlled study.

Modules:
    envload   -- load backend/.env into os.environ (zero-dependency)
    budget    -- LLM call/token accounting (the "same total compute" axis)
    dataset   -- build/load the sectionized eval cache from runs/paper_eval
    systems   -- the systems-under-test (baselines + ablations of the full pipeline)
    metrics   -- accuracy / P-R-F1 / score-gap / ECE with bootstrap CIs and McNemar
    run       -- CLI orchestrator

IMPORTANT: call ``envload.load()`` BEFORE importing anything from ``grantagent``,
because ``grantagent.service.llm`` instantiates its singleton at import time.
"""
