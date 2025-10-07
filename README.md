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
uv sync
uv run python -m pip install -e .
```

# ToDo Lists
- [] quadrotor forward flight maintaining height (drag).
- [] limit the commands bandwidth.
- [] tailsitter forward flight maintaining height (aerodynamics-agnostic).
- [] vision-based tailsitter navigation based on depth map.
