# scripts/: repo-wide helper scripts

Module-specific scripts belong in `<module>/scripts/`. This folder is only for tools that apply to the whole repository.

| Script | Purpose | Usage |
|---|---|---|
| `validate_contracts.py` | Validates `shared/contracts/` schemas and examples, and checks intersection id consistency | `pip install jsonschema pyyaml && python scripts/validate_contracts.py` |
| `check_large_files.sh` | Fails if a tracked file exceeds 2 MB (gzipped SUMO nets exempt) | `scripts/check_large_files.sh [limit_bytes]` |

Both run in CI.
