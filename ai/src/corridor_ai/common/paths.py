"""Repository paths, resolved relative to this file (independent of the CWD)."""

from __future__ import annotations

from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = AI_DIR.parent
CONFIG_DIR = AI_DIR / "configs"
FIXTURE_DIR = AI_DIR / "sim_fixtures"
RUNS_DIR = AI_DIR / "runs"
CHECKPOINT_DIR = AI_DIR / "checkpoints"
CACHE_DIR = RUNS_DIR / "_cache"
CONTRACTS_DIR = REPO_ROOT / "shared" / "contracts"
SCHEMA_DIR = CONTRACTS_DIR / "schemas"


def resolve(path: str | Path, base: Path = AI_DIR) -> Path:
    """Resolve `path` against `base` unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()
