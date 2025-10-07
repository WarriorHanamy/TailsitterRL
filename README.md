# Foundations
- [VisFly](https://github.com/SJTU-ViSYS-team/VisFly/tree/beta)
- [warp](https://github.com/NVIDIA/warp)
- [spinningup](https://github.com/openai/spinningup)
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [newton](https://github.com/newton-physics/newton)
- [OmniDrones](https://github.com/btx0424/OmniDrones)


# Convetions
1. actions are always normalized into [-1, 1]
2. commands are denormalized from actions, serving for the physical purpose.


# Prerequisites

Install Rust and uv using curl:

```shell
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install just using cargo (Rust package manager)
cargo install just
```

# Installation
```shell
mamba env create -f environment.yml
mamba activate rl
```
Check which pip (Linux/macOS) or where pip (Windows) first to confirm you’re using the virtual environment’s pip—avoid breaking your system Python.

```shell
pip install -e .
```

```shell
python -m ipykernel install --user --name=rl-notebook --display-name "rl-notebok"
```


# Hybrid Packages Management
1. conda/mamba to manage the runtime depencies and compile tools.
2. uvx to manage development tools. e.g., ruff, black, pre-commit.

# ToDo Lists
- [] quadrotor forward flight maintaining height (drag).
- [] limit the commands bandwidth.
- [] tailsitter forward flight maintaining height (aerodynamics-agnostic).
- [] vision-based tailsitter navigation based on depth map.
