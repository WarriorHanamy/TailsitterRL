import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ---
# Define System and Controller Architectures
# ---


class ThirdOrderSystem(nn.Module):
    """Defines the third-order system dynamics."""

    def __init__(self, zeta, omega_n, noise_level=1.0, sim_dt=0.01):
        super(ThirdOrderSystem, self).__init__()
        self.zeta = torch.tensor(zeta, dtype=torch.float32)
        self.omega_n = torch.tensor(omega_n, dtype=torch.float32)
        self.sim_dt = sim_dt
        self.state = torch.zeros(3).detach()
        self.noise_level = noise_level

    def simulate(self, cmd):
        """Simulates one time step for inference/testing, detaching the state."""
        if not isinstance(cmd, torch.Tensor):
            cmd = torch.tensor(cmd, dtype=torch.float32)

        # Detach state to prevent it from being part of the graph in subsequent steps
        current_state = self.state.detach()

        angle_next = current_state[0] + self.sim_dt * current_state[1]
        y, dy_dt = current_state[1], current_state[2]

        dy_dt2 = (
            -(self.omega_n**2) * y
            - 2 * self.zeta * self.omega_n * dy_dt
            + self.omega_n**2 * cmd
        )

        y_next = y + dy_dt * self.sim_dt + torch.randn(1).item() * self.noise_level
        dy_dt_next = dy_dt + dy_dt2 * self.sim_dt
        # Add noise to the velocity update for realism
        self.state = torch.stack([angle_next, y_next, dy_dt_next])
        return self.state

    def forward_train(self, current_state, cmd):
        """Calculates the next state for training, keeping operations in the graph."""
        angle_next = current_state[0] + self.sim_dt * current_state[1]
        y, dy_dt = current_state[1], current_state[2]

        dy_dt2 = (
            -(self.omega_n**2) * y
            - 2 * self.zeta * self.omega_n * dy_dt
            + self.omega_n**2 * cmd
        )

        y_next = y + dy_dt * self.sim_dt
        dy_dt_next = dy_dt + dy_dt2 * self.sim_dt
        next_state = torch.stack([angle_next, y_next, dy_dt_next])
        return next_state


class ControlPolicy(nn.Module):
    """A simple neural network to act as a control policy."""

    def __init__(self):
        super(ControlPolicy, self).__init__()
        self.control_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
        )

        self.Kp = nn.Parameter(torch.tensor(5.0), requires_grad=False)

    def forward(self, target, current_state):
        """Calculates control command based on target and current state."""
        error = target - current_state[0]
        velocity = current_state[1]
        net_input = torch.stack([error, velocity])
        cmd = self.control_net(net_input) + self.Kp * error
        return cmd.squeeze()


# ---
# Training Section
# ---

# Training parameters
zeta_train_values = [0.8]  # Different zeta values for training
omega_n_train = 100.0
target_value_train = torch.tensor(1.0)
num_epochs = 1000
rho1 = 1000.0  # Weight for position error
rho2 = 10000.0  # Weight for velocity penalty
rho3 = 1e-4  # Weight for jerk penalty
velocity_limit = 12.0  # Velocity limit for penalty

# Instantiate models and optimizer
policy = ControlPolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

# Create three system instances for training, one for each zeta value
training_systems = [
    ThirdOrderSystem(zeta=z, omega_n=omega_n_train) for z in zeta_train_values
]

print(
    f"Starting training on systems with zeta={zeta_train_values}, omega_n={omega_n_train}..."
)
for epoch in range(num_epochs):
    optimizer.zero_grad()

    total_epoch_loss = 0

    # Loop through each system instance for training in each epoch
    for system in training_systems:
        # Use a temporary state for building the computation graph for this system
        temp_system_state = torch.zeros(3)
        trajectory_states = []

        # Forward pass over a time horizon to collect trajectory
        for _ in range(200):
            cmd = policy(target_value_train, temp_system_state)
            # Use the forward_train method to keep operations in the graph
            temp_system_state = system.forward_train(temp_system_state, cmd)
            temp_system_state[1] += torch.randn(1).item() * 0.5
            trajectory_states.append(temp_system_state)

        # Calculate loss for this specific trajectory
        trajectory_loss = 0
        for i, state in enumerate(trajectory_states):
            if (
                i > 30
            ):  # Allow some time for the system to settle before calculating position error
                error_y = (target_value_train - state[0]) ** 2
                trajectory_loss += rho1 * error_y
                jerk_penalty = rho3 * (state[2] ** 2)
                trajectory_loss += jerk_penalty

            over_limit = torch.relu(torch.abs(state[1]) - velocity_limit)
            velocity_penalty = rho2 * over_limit**2
            trajectory_loss += velocity_penalty

        # Accumulate the loss from this zeta simulation
        total_epoch_loss += trajectory_loss

    # Average the loss over all system instances before backpropagation
    avg_epoch_loss = total_epoch_loss / len(training_systems)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    avg_epoch_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss.item():.4f}")

print("Training finished.")


# ---
# Simulation and Plotting Function for different scenarios
# ---
def run_and_plot_simulation(policy, test_zeta, test_omega_n, target_value_sim, kp_gain):
    """
    Simulates the trained policy and Kp controller for a given system
    and plots the results.
    """
    print(f"\n--- Running simulation for Zeta={test_zeta}, Omega_n={test_omega_n} ---")

    # Instantiate systems for this specific scenario
    system_trained = ThirdOrderSystem(zeta=test_zeta, omega_n=test_omega_n)
    system_kp = ThirdOrderSystem(zeta=test_zeta, omega_n=test_omega_n)

    # Data logging lists
    time_points = []
    positions_trained, velocities_trained, commands_trained = [], [], []
    positions_kp, velocities_kp, commands_kp = [], [], []

    current_time = 0.0
    total_time = 5.0

    while current_time <= total_time:
        with torch.no_grad():
            # 1. Trained policy
            cmd_trained = policy(target_value_sim, system_trained.state)

            # 2. Kp controller
            error_kp = target_value_sim - system_kp.state[0]
            cmd_kp = kp_gain * error_kp

        # Simulate one step for all systems
        system_trained.simulate(cmd_trained)
        system_kp.simulate(cmd_kp)

        # Record data
        time_points.append(current_time)
        positions_trained.append(system_trained.state[0].item())
        velocities_trained.append(system_trained.state[1].item())
        commands_trained.append(cmd_trained.item())
        positions_kp.append(system_kp.state[0].item())
        velocities_kp.append(system_kp.state[1].item())
        commands_kp.append(cmd_kp.item())

        current_time += system_trained.sim_dt

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(
        f"Controller Response for Zeta={test_zeta}, Omega_n={test_omega_n}\n"
        f"(Trained on Zeta={zeta_train_values}, Omega_n={omega_n_train})",
        fontsize=16,
        fontweight="bold",
    )

    # Plot Position
    ax1.plot(time_points, positions_trained, label="Trained Policy", linewidth=2)
    ax1.plot(
        time_points, positions_kp, label=f"Kp Controller (Kp={kp_gain})", linestyle="--"
    )
    ax1.axhline(y=target_value_sim.item(), color="r", linestyle=":", label="Target")
    ax1.set_title("System Position Comparison", fontsize=14)
    ax1.set_ylabel("Position", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Plot Velocity
    ax2.plot(time_points, velocities_trained, label="Trained Policy", linewidth=2)
    ax2.plot(time_points, velocities_kp, label="Kp Controller", linestyle="--")
    ax2.set_title("System Velocity Comparison", fontsize=14)
    ax2.set_ylabel("Velocity", fontsize=12)
    ax2.legend()
    ax2.grid(True)

    # Plot Command
    ax3.plot(time_points, commands_trained, label="Trained Policy", linewidth=2)
    ax3.plot(time_points, commands_kp, label="Kp Controller", linestyle="--")
    ax3.set_title("Control Command Comparison", fontsize=14)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Command", fontsize=12)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create a directory to save plots if it doesn't exist
    if not os.path.exists("simulation_results"):
        os.makedirs("simulation_results")

    filename = (
        f"simulation_results/comparison_zeta_{test_zeta}_omega_{test_omega_n}.png"
    )
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close(fig)  # Close the figure to free up memory


# ---
# Define Test Scenarios and Run Simulations
# ---
target_value_sim = torch.tensor(2.0)
kp_gain = 5.0

# Define the different system dynamics to test
test_scenarios = [
    # Test on the three zeta values used during training
    {"zeta": 0.4, "omega_n": 0.5 * omega_n_train},
    {"zeta": 0.5, "omega_n": 0.5 * omega_n_train},
    {"zeta": 0.6, "omega_n": 0.5 * omega_n_train},
    {"zeta": 0.7, "omega_n": 0.5 * omega_n_train},
    {"zeta": 0.8, "omega_n": 0.5 * omega_n_train},
    {"zeta": 0.9, "omega_n": 0.5 * omega_n_train},
    {"zeta": 1.0, "omega_n": 0.5 * omega_n_train},
    {"zeta": 1.1, "omega_n": 0.5 * omega_n_train},
    {"zeta": 0.4, "omega_n": omega_n_train},
    {"zeta": 0.5, "omega_n": omega_n_train},
    {"zeta": 0.6, "omega_n": omega_n_train},
    {"zeta": 0.7, "omega_n": omega_n_train},
    {"zeta": 0.8, "omega_n": omega_n_train},
    {"zeta": 0.9, "omega_n": omega_n_train},
    {"zeta": 1.0, "omega_n": omega_n_train},
    {"zeta": 1.1, "omega_n": omega_n_train},
    {"zeta": 0.4, "omega_n": 2.0 * omega_n_train},
    {"zeta": 0.5, "omega_n": 2.0 * omega_n_train},
    {"zeta": 0.6, "omega_n": 2.0 * omega_n_train},
    {"zeta": 0.7, "omega_n": 2.0 * omega_n_train},
    {"zeta": 0.8, "omega_n": 2.0 * omega_n_train},
    {"zeta": 0.9, "omega_n": 2.0 * omega_n_train},
    {"zeta": 1.0, "omega_n": 2.0 * omega_n_train},
    {"zeta": 1.1, "omega_n": 2.0 * omega_n_train},
]

# Run simulation and plotting for each scenario
for scenario in test_scenarios:
    run_and_plot_simulation(
        policy,
        test_zeta=scenario["zeta"],
        test_omega_n=scenario["omega_n"],
        target_value_sim=target_value_sim,
        kp_gain=kp_gain,
    )

print(
    "\nAll simulations are complete. Check the 'simulation_results' folder for the plots."
)
