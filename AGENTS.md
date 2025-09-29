# Repository Guidelines

## Project Structure & Module Organization
- `vtol_rl/` is the installable package; task environments sit in `envs/`, shared dynamics and controllers live in `envs/base/`, and tensor helpers reside in `utils/`.
- JSON presets in `vtol_rl/config/` define aircraft, scenes, and objects—clone the closest sample, document new keys inline, and keep diffs targeted.
- Tests mirror runtime scope: `vtol_rl/tests/` covers modules with unit checks, while top-level `tests/` executes system smoke scenarios.
- Vendor code remains under `aerial_gym_simulator/`; treat modifications as upstream patches and summarise differences in pull requests.

## Build, Test, and Development Commands
- `uv sync` provisions the virtual environment and resolves runtime plus dev dependencies; `environment.yml` is retained for future Conda/mamba needs but is not active.
- `uv run python -m pip install -e .` keeps the package editable so code and tests stay in lockstep.
- `uv run pytest` runs the entire suite; add `-k hover` or `--lf` when iterating inside the red→green cycle.
- `UV_CACHE_DIR=.uv_cache PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit run --all-files` runs the pre-commit check as the last check.

## Test-Driven Development Workflow
- Begin every feature or fix by writing or tightening a failing test that describes the intended behaviour at the smallest feasible scope.
- Implement just enough code to satisfy the test, verify with `uv run pytest`, then refactor while the suite stays green.
- During refactors or large sweeps, prefer `uv run pytest --lf` to replay recent failures and guard against regressions.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents, explicit type hints, and device-aware tensor handling; mirror signatures already used in environments such as `HoverEnv`.
- Name files with `snake_case.py`, classes with `PascalCase`, and helper functions plus fixtures with descriptive `snake_case` verbs (`build_hover_env`).
- Use `setup_logger` for structured logging and isolate side effects to keep tests deterministic across platforms.

## Commit & Pull Request Guidelines
- Keep commit subjects imperative and concise (e.g., `Add hover reward guard`), adding deeper context in the body when behaviour changes.
- Confirm `uv run pytest` and `uv run ruff format` succeed before pushing; include reproduction commands, new fixtures, and checklist updates in the PR description.
- Update `ToDos.md` checkboxes as tasks complete, leaving items unchecked until the work is fully delivered.
- Highlight cross-module impacts—especially config updates that influence training scripts—to guide reviewers toward targeted verification.
