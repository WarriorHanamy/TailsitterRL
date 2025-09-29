# To-Dos

- [x] Remove all `habitat_sim` references and related scene control from `vtol_rl/envs` (project forked from third-party source).
- [x] Fix latest pre-commit error.
- [x] Ensure the full test suite passes via `uv run pytest`.
- [x]  Make sure the action shape is (N,4) in `envs/base/dynamics.py` and `uv run pytest` works normally.
```python
    def step(self, action) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Step the simulation forward by one control period given the action.
        Args:
            action (torch.Tensor): shape (N, 4), in range [-1, 1]
        Returns:
            state (torch.Tensor): shape (N, 13): (p,q,v,r)
        """
```

- [x] Make sure the quaternion functionalities in `maths.Quaternion` runs normally.
1. It supports batches operations, return shape should be (batch_size, ...), as follows:
```python
    def R(self):
        """
        Args: self (Quaternion): shape (N, (w,x,y,z))
        Returns: R (torch.Tensor): shape (N, 3, 3)
        """
```
You should compare all shape returned is predictably and first write test in vtol_rl/test folder.
