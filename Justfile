# Shared development commands for TailsitterRL.

export UV_CACHE_DIR := ".uv_cache"

k:
	tmux kill-session
work:
	tmuxp load tmux_config/agents.yaml

# Run the full pytest suite once.
test:
	uv run pytest

# Run the suite twice to catch caching issues.
test-all:
	uv run pytest
	uv run pytest

# Rerun only the last failing tests.
test-lf:
	uv run pytest --lf
