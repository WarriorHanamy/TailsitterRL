# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TailsitterRL is a reinforcement learning framework for VTOL (Vertical Take-Off and Landing) aircraft, specifically tailsitter UAVs. The project implements simulation environments, dynamics models, and control systems for training RL agents on complex aerial maneuvers.

## Environment Setup

### Hybrid Package Management
- **Runtime dependencies**: Managed with conda/mamba via `environment.yml`
- **Development tools**: Managed with uv for modern Python tooling

### Installation Commands
```bash
# Create conda environment
mamba env create -f environment.yml
mamba activate rl

# Install package in editable mode
pip install -e .

# Sync development dependencies
uv sync

# Install package after uv sync
uv run python -m pip install -e .
```

## Development Commands

### Testing
```bash
# Run full test suite (recommended to run twice to catch caching issues)
UV_CACHE_DIR=.uv_cache uv run pytest
UV_CACHE_DIR=.uv_cache uv run pytest

# Run only failed tests
uv run pytest --lf

# Run specific test file
uv run pytest vtol_rl/tests/test_specific_file.py
```

### Code Quality
```bash
# Run linting and formatting (auto-fixes issues)
uv run ruff --fix
uv run ruff format

# Pre-commit hooks automatically run ruff on commits
```

## Architecture Overview

### Core Components

#### Environment Hierarchy (`vtol_rl/envs/`)
- **Base Classes**:
  - `TailsitterEnvsBase`: Main vectorized environment class extending Gymnasium VecEnv
  - `DroneEnvsBase`: Base functionality for drone environments
- **Task-Specific Environments**: `HoverEnv`, `NavigationEnv`, `LandingEnv`, `RacingEnv`, `CatchEnv`, `TrackingEnv`, `MultiNavigationEnv`, `DynamicEnv`
- **Core Systems**:
  - `dynamics.py`: Physics simulation with quaternion-based orientation handling
  - `controller.py`: Abstract controller base classes (ThrustController, BodyrateController)
  - `droneEnv.py`: Base drone environment implementation

#### Configuration System (`vtol_rl/config/`)
- JSON-based configuration system for aircraft parameters, scenes, and assets
- Use `JsonResource` class for loading configuration files
- Conventions: Duplicate closest preset, document new keys inline

#### Utilities (`vtol_rl/utils/`)
- `maths.py`: Core mathematical utilities including `Quaternion` and `Integrator` classes
- `type.py`: Type definitions, state slices, and action type enums
- `logger.py`: Structured logging with `setup_logger`

### Key Architectural Patterns

#### Tensor Conventions
- Actions normalized to `[-1, 1]` range, then denormalized for physical control
- Batch dimension first: tensors shaped as `(N, ...)` where N is batch size
- Device-aware operations: code supports both CPU and GPU execution
- Shape guards in `reset()` methods, compute-bound paths in `step()` methods

#### State Representation
- 13-dimensional state vector: position (3), orientation/quaternion (4), velocity (3), angular velocity (3)
- State slices defined in `type.py`: `STATE_POS`, `STATE_ORI`, `STATE_VEL`, `STATE_ANG_VEL`

#### Control Pipeline
1. Actions from RL agent (normalized [-1,1])
2. Controller processes actions into commands
3. Dynamics simulation updates state
4. Environment returns observations and rewards

## Testing Strategy

- **Unit Tests**: Located in `vtol_rl/tests/` alongside source modules
- **Integration Tests**: Would typically go in top-level `tests/` directory
- Test discovery: Functions prefixed with `test_`, classes prefixed with `Test`
- Use fixtures for shared setup, seeded randomness for reproducibility

## Code Standards

- **Style**: PEP 8 with 4-space indentation, explicit type hints required
- **Naming**: `snake_case` for modules/files, `PascalCase` for classes
- **Device Awareness**: All tensor operations should handle both CPU and GPU
- **Logging**: Use `setup_logger` for structured output, avoid import-time side effects
- **Batch Processing**: All functions should support batched operations

## Development Workflow

1. **Feature Development**: Create new environment by extending base classes
2. **Configuration**: Add new JSON presets to `vtol_rl/config/` following existing patterns
3. **Testing**: Write failing tests first, then implement functionality
4. **Verification**: Run `UV_CACHE_DIR=.uv_cache uv run pytest` before commits
5. **Documentation**: Update relevant documentation and configuration files

## Common Gotchas

- GPU vs CPU: Automatically defaults to CPU for <1000 environments for performance
- Quaternion Operations: Always return batched tensors with shape `(N, ...)`
- Action Normalization: Remember actions are [-1,1] and must be denormalized
- Shape Consistency: Ensure all tensor operations maintain consistent batch dimensions