# Foundations
- [VisFly](https://github.com/SJTU-ViSYS-team/VisFly/tree/beta)
- [warp](https://github.com/NVIDIA/warp)
- [spinningup](https://github.com/openai/spinningup)
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [newton](https://github.com/newton-physics/newton)


# Convetions
1. actions are always normalized into [-1, 1]
2. commands are denormalized from actions, serving for the physical purpose.


# Installation
```shell
mamba env create -f environment.yml
mamba activate rl
```
Check which pip (Linux/macOS) or where pip (Windows) first to confirm you’re using the virtual environment’s pip—avoid breaking your system Python.

```shell
pip install -e .
```

# ToDo Lists
- [] quadrotor forward flight maintaining height (drag).
- [] limit the commands bandwidth.
- [] tailsitter forward flight maintaining height (aerodynamics-agnostic).
- [] vision-based tailsitter navigation based on depth map.