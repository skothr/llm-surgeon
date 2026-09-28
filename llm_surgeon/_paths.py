"""User-writable default locations for llm-surgeon's on-disk state.

Everything lives under one root:

- ``$LLM_SURGEON_HOME`` if set, else
- ``$XDG_CACHE_HOME/llm-surgeon`` if ``XDG_CACHE_HOME`` is set, else
- ``~/.cache/llm-surgeon``.

Individual locations keep their own overrides (``LLM_SURGEON_CACHE_DIR`` for
model downloads, ``LLM_SURGEON_DB`` for the experiment database). Nothing is
computed relative to the package directory, so a regular (non-editable)
install never writes into site-packages.
"""

import os
from pathlib import Path


def surgeon_home() -> Path:
    """Root directory for llm-surgeon's caches and databases."""
    env = os.environ.get("LLM_SURGEON_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "llm-surgeon"


def model_cache_dir() -> str:
    """HF-hub cache directory for model downloads (``LLM_SURGEON_CACHE_DIR`` overrides)."""
    return os.environ.get("LLM_SURGEON_CACHE_DIR") or str(surgeon_home() / "models")


def default_db_path() -> str:
    """Experiment-tracking SQLite path (``LLM_SURGEON_DB`` overrides)."""
    return os.environ.get("LLM_SURGEON_DB") or str(surgeon_home() / "experiments.db")
