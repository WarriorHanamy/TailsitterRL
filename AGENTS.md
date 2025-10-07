# Repository Guidelines

## Project Structure & Module Organization
- `vtol_rl/` houses the installable package; task environments live in `vtol_rl/envs/`, with shared dynamics and controllers under `vtol_rl/envs/base/`, and tensor utilities in `vtol_rl/utils/`.
- JSON presets in `vtol_rl/config/` describe aircraft, scenes, and assets. Duplicate the closest preset, document new keys in-line, and keep diffs limited to the scenario you touch.
- Tests mirror runtime scope: unit coverage belongs in `vtol_rl/tests/`, while integration and smoke runs sit under the top-level `tests/` directory.

## Build, Test, and Development Commands
- `uv sync` resolves runtime and dev dependencies into the project-managed virtual environment.
- `uv run python -m pip install -e .` installs the package in editable mode so code edits and imports stay aligned.
- `UV_CACHE_DIR=.uv_cache uv run pytest uv run pytest` executes the full suite twice to surface caching issues; use `uv run pytest --lf` for focused reruns.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, explicit type hints, and device-aware tensor logic that mirrors `HoverEnv` and peers.
- Name modules `snake_case.py`, classes `PascalCase`, and helper functions with descriptive verbs (`build_hover_env`).
- Prefer `setup_logger` for structured output and avoid side effects at import time to keep trainings reproducible.

## Testing Guidelines
- Write the failing test first, choosing the narrowest scope possible; unit checks live beside their modules, and scenario smoke tests reside in `tests/`.
- Pytest discovers functions prefixed with `test_`; use fixtures for shared setup and guard against nondeterminism with seeded randomness.
- Always run `uv run pytest` before submitting changes and capture new regression cases when fixing bugs.

## Commit & Pull Request Guidelines
- Keep commit subjects imperative and concise (`Add hover reward guard`), with bodies summarising behaviour changes or migration steps.
- PRs should link issues when available, include reproduction commands, note cross-module impacts (configs, training scripts), and update `ToDos.md` only when work is complete.
- Confirm the full test suite passes locally before requesting review and flag any intentionally skipped checks.
