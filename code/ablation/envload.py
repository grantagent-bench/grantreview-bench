"""Zero-dependency .env loader.

python-dotenv is not installed in this environment, and ``grantagent.service.llm``
builds its singleton at import time from ``os.environ``.  So we parse ``backend/.env``
by hand and inject values *before* any grantagent import happens.

Shell-exported variables win over the file (so ``OPENAI_API_KEY=... python -m ...``
still overrides the file), matching python-dotenv's default ``override=False``.
"""
from __future__ import annotations

import os
from pathlib import Path

# backend/  (two levels up from scripts/ablation/envload.py)
REPO_DIR = Path(__file__).resolve().parents[2]


def load(path: str | os.PathLike | None = None, override: bool = False) -> dict:
    """Load KEY=VALUE lines from an .env file into os.environ.

    Returns a dict of the keys that were set (values omitted for safety).
    """
    env_path = Path(path) if path else REPO_DIR / ".env"
    applied: dict[str, bool] = {}
    if not env_path.exists():
        return applied

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if not value:
            # Empty placeholder (e.g. an unset ANTHROPIC_API_KEY) — skip so we
            # don't clobber a real shell export with "".
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = True
    return applied


def status() -> dict:
    """Non-secret summary of what the LLM layer will see."""
    return {
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
        "LLM_MODEL": os.environ.get("LLM_MODEL", "(provider default)"),
        "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
