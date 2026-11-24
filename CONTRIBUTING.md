# Contributing

This is a team monorepo. Each top-level module has one owner (see the [README](README.md#team-and-ownership) and [`.github/CODEOWNERS`](.github/CODEOWNERS)). These rules keep `main` runnable and stop us from breaking each other's work.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Protected. Always runnable. Updated only by merging `develop` after a team check. |
| `develop` | Integration branch. All feature PRs target `develop`. |
| `<type>/<module>-<short-desc>` | Feature branches, e.g. `feat/ai-marl-env`, `fix/backend-ws-reconnect`, `docs/edge-readme`. |

`<type>` is one of the Conventional Commit types below. `<module>` is the top-level folder (`ai`, `simulation`, `backend`, `dashboard`, `edge`, `hardware`, `analysis`) or `repo`, `contracts`, `docs`, `ci`.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) with the module as scope:

```
feat(ai): add PPO trainer
fix(backend): reconnect WebSocket after broker restart
docs(edge): document ROI calibration
chore(contracts): add speed field to queue_telemetry (DRAFT)
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`, `build`. Keep commits small and focused. Write the subject in the imperative mood.

## Pull requests

1. Open a PR from your feature branch into `develop`.
2. Fill in the PR template (what, why, how it was tested, contracts touched).
3. Get **at least one approving review from another team member**. If you touch another member's folder, that module owner must review (CODEOWNERS requests them automatically).
4. CI must be green.
5. Prefer squash-merge, with a Conventional Commit title.

## Touching another member's folder

- Only modify your own module. For other modules, open an issue or ask the owner.
- Small fixes in someone else's folder (a typo, a broken import caused by your change) are fine **if the owner reviews the PR**.
- Never move, delete or rewrite another member's implementation without their agreement.
- Cross-team interfaces live in [`shared/contracts/`](shared/contracts/). Anyone may propose a change, but every affected owner must approve it. Breaking changes bump `schema_version`.

## Code style

| Language | Tools | Command |
|---|---|---|
| Python | **ruff** (lint + import sorting) and **black** (format), line length 100, configured in the root `pyproject.toml` | `ruff check . && black --check .` |
| JavaScript / TypeScript | **eslint** + **prettier** (configured in `dashboard/` when it is created) | `npm run lint && npx prettier --check .` |

Other conventions:

- Python ≥ 3.10 (**ASSUMPTION**); development is tested on 3.14.
- Unit tests go in `<module>/tests/` and must run **without SUMO, hardware or network access**. Mark tests that need SUMO with `@pytest.mark.sumo` so CI can skip them.
- Put configuration in files or environment variables, never in hard-coded constants. Add every new environment variable to `.env.example` with an empty or safe default.
- Use the signal safety limits everywhere: min green 10 s, max green 60 s, yellow 3 s, all-red 1 s.
- Prefix anything assumed rather than specified with `ASSUMPTION` (in code comments and docs).

## Never commit

- **Secrets**: API keys, tokens, passwords, MQTT credentials, HMAC/AES keys, `.env` files, private keys. If you commit one by accident, tell the team immediately, **rotate the key**, and clean the history with `git filter-repo`. Deleting the file in a later commit is not enough.
- **Data**: raw or processed datasets (see [data/README.md](data/README.md)). Licences must be confirmed first.
- **Model weights and run outputs**: `*.pt`, `*.pth`, `*.onnx`, `*.engine`, `runs/`, `checkpoints/`.
- **Simulation outputs**: tripinfo, edgeData, stats and other SUMO output files.
- **Large binaries**: anything over 2 MB needs team agreement. Commit SUMO networks gzipped (`*.net.xml.gz`). Check with `scripts/check_large_files.sh`.

The `.gitignore` covers these, but it is not a substitute for checking `git status` before you commit.

## Issues

Use the issue templates (bug report or feature request) and add the module label.
