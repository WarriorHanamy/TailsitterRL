from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from vtol_rl.envs.base.dynamics import Dynamics


@dataclass
class HoverConfig:
    horizon: int = 200
    rollout_dt: float = 0.02
    sim_time_step: float = 0.005
    learning_rate: float = 1e-3
    num_epochs: int = 300
    early_stop_patience: int = 25
    early_stop_min_delta: float = 1e-1
    log_dir: Path = Path("logs/bptt_hover")
    device: torch.device = torch.device("cpu")
    seed: int = 42
    position_weight: float = 5.0
    velocity_weight: float = 1.0
    angular_velocity_weight: float = 0.5
    orientation_weight: float = 0.5
    action_weight: float = 0.05
    grad_clip: float = 5.0
    eval_horizon: int = 400
    train_enabled: bool = True


class HoverPolicy(nn.Module):
    def __init__(self, state_dim: int = 13, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        error = target_state - state
        stacked = torch.cat([state, error], dim=-1)
        raw_action = self.net(stacked)
        return torch.tanh(raw_action)


class HoverBPTTExperiment:
    def __init__(self, cfg: HoverConfig | None = None):
        self.cfg = cfg or HoverConfig()
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.cfg.log_dir / "training_log.csv"
        self.fig_file = self.cfg.log_dir / "hover_trajectory.png"

        torch.manual_seed(self.cfg.seed)

        self.device = self.cfg.device
        self.dynamics = Dynamics(
            num=1,
            sim_time_step=self.cfg.sim_time_step,
            ctrl_period=self.cfg.rollout_dt,
            ctrl_delay=False,
            comm_delay=0.0,
            drag_random=0.0,
            acc_noise_std=0.0,
            device=self.device,
            state_clamp=False,
        )

        self.policy = HoverPolicy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)

        self.target_state = torch.tensor(
            [[0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )

        if not self.log_file.exists():
            with self.log_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch",
                        "loss",
                        "mean_position_error",
                        "final_position_error",
                        "mean_velocity_norm",
                        "mean_action_norm",
                    ]
                )

    def reset_dynamics(self) -> None:
        self.dynamics.reset(
            pos=torch.tensor([[0.0, 0.0, 1.2]], device=self.device),
            vel=torch.zeros((1, 3), device=self.device),
            ori=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
            ori_vel=torch.zeros((1, 3), device=self.device),
        )

    def compute_loss(
        self,
        positions: list[torch.Tensor],
        velocities: list[torch.Tensor],
        angular_velocities: list[torch.Tensor],
        quaternions: list[torch.Tensor],
        actions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_pos = self.target_state[:, :3]
        pos_errors = [((pos - target_pos) ** 2).sum(dim=1) for pos in positions]
        mean_position_error = torch.stack(pos_errors).mean()
        final_position_error = pos_errors[-1].mean()

        vel_norms = [vel.norm(dim=1) for vel in velocities]
        ang_vel_norms = [ang_vel.norm(dim=1) for ang_vel in angular_velocities]

        quat = torch.stack(quaternions)
        quat_norm_error = ((1.0 - quat[..., 0].abs()) ** 2).mean()

        action_norms = [action.norm(dim=1) for action in actions]

        loss = (
            self.cfg.position_weight * mean_position_error
            + self.cfg.velocity_weight * torch.stack(vel_norms).mean()
            + self.cfg.angular_velocity_weight * torch.stack(ang_vel_norms).mean()
            + self.cfg.orientation_weight * quat_norm_error
            + self.cfg.action_weight * torch.stack(action_norms).mean()
        )

        metrics = {
            "mean_position_error": mean_position_error.item(),
            "final_position_error": final_position_error.item(),
            "mean_velocity_norm": torch.stack(vel_norms).mean().item(),
            "mean_action_norm": torch.stack(action_norms).mean().item(),
        }
        return loss, metrics

    def log_metrics(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
        with self.log_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    loss,
                    metrics["mean_position_error"],
                    metrics["final_position_error"],
                    metrics["mean_velocity_norm"],
                    metrics["mean_action_norm"],
                ]
            )

    def train(self) -> None:
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.reset_dynamics()
            self.optimizer.zero_grad()

            positions: list[torch.Tensor] = []
            velocities: list[torch.Tensor] = []
            angular_velocities: list[torch.Tensor] = []
            quaternions: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []

            for _ in range(self.cfg.horizon):
                state = self.dynamics.state
                action = self.policy(state, self.target_state)
                states_next = self.dynamics.step(action)

                positions.append(states_next[:, :3])
                velocities.append(states_next[:, 7:10])
                angular_velocities.append(states_next[:, 10:13])
                quaternions.append(states_next[:, 3:7])
                actions.append(action)

            loss, metrics = self.compute_loss(
                positions=positions,
                velocities=velocities,
                angular_velocities=angular_velocities,
                quaternions=quaternions,
                actions=actions,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            loss_value = loss.item()
            improved = loss_value < (best_loss - self.cfg.early_stop_min_delta)
            if improved:
                best_loss = loss_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            should_log = epoch % 10 == 0 or epoch == 1 or improved
            if should_log:
                self.log_metrics(epoch, loss_value, metrics)
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[Epoch {epoch:03d}] loss={loss_value:.4f} "
                    f"mean_pos_err={metrics['mean_position_error']:.4f} "
                    f"final_pos_err={metrics['final_position_error']:.4f}"
                )
            elif improved:
                print(
                    f"[Epoch {epoch:03d}] loss improved to {loss_value:.4f} "
                    f"(best so far), patience reset."
                )

            if (
                self.cfg.early_stop_patience > 0
                and epochs_without_improvement >= self.cfg.early_stop_patience
            ):
                if not should_log:
                    self.log_metrics(epoch, loss_value, metrics)
                print(
                    f"Early stopping at epoch {epoch:03d}; "
                    f"no improvement greater than {self.cfg.early_stop_min_delta:.2e} "
                    f"for {self.cfg.early_stop_patience} epochs. "
                    f"Best loss={best_loss:.4f}"
                )
                break

    def evaluate(self) -> dict[str, torch.Tensor]:
        self.reset_dynamics()

        positions = []
        velocities = []
        actions = []
        times = []

        with torch.no_grad():
            for step_idx in range(self.cfg.eval_horizon):
                state = self.dynamics.state
                action = self.policy(state, self.target_state)
                self.dynamics.step(action)

                positions.append(self.dynamics.position.clone())
                velocities.append(self.dynamics.velocity.clone())
                actions.append(action.clone())
                times.append(step_idx * self.cfg.rollout_dt)

        trajectory = {
            "time": torch.tensor(times),
            "positions": torch.cat(positions, dim=0),
            "velocities": torch.cat(velocities, dim=0),
            "actions": torch.cat(actions, dim=0),
        }
        return trajectory

    def plot_trajectory(self, trajectory: dict[str, torch.Tensor]) -> None:
        times = trajectory["time"].cpu()
        positions = trajectory["positions"].cpu()
        velocities = trajectory["velocities"].cpu()

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(times, positions[:, 0], label="x")
        axes[0].plot(times, positions[:, 1], label="y")
        axes[0].plot(times, positions[:, 2], label="z", linewidth=2)
        axes[0].axhline(self.target_state[0, 2].item(), color="tab:red", linestyle="--", label="z target")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend()
        axes[0].set_title("Position Tracking")

        axes[1].plot(times, velocities[:, 0], label="vx")
        axes[1].plot(times, velocities[:, 1], label="vy")
        axes[1].plot(times, velocities[:, 2], label="vz")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend()
        axes[1].set_title("Velocity")

        axes[2].plot(times, trajectory["actions"][:, 0].cpu(), label="thrust")
        axes[2].plot(times, trajectory["actions"][:, 1].cpu(), label="ωx")
        axes[2].plot(times, trajectory["actions"][:, 2].cpu(), label="ωy")
        axes[2].plot(times, trajectory["actions"][:, 3].cpu(), label="ωz")
        axes[2].set_ylabel("Normalized Action")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].set_title("Control Inputs")

        fig.tight_layout()
        fig.savefig(self.fig_file, dpi=150)
        plt.close(fig)

    def run(self) -> None:
        if self.cfg.train_enabled:
            self.train()
        else:
            print("Training disabled via configuration; skipping training loop.")

        trajectory = self.evaluate()
        self.plot_trajectory(trajectory)
        print(f"Training log saved to {self.log_file}")
        print(f"Trajectory plot saved to {self.fig_file}")


def main() -> None:
    experiment = HoverBPTTExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
