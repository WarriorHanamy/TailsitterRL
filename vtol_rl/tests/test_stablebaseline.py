import torch
from stable_baselines3 import PPO
import gymnasium as gym
import torch.nn as nn

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",  # Use "MlpPolicy" for simple observation spaces
    env,
    policy_kwargs={
        "activation_fn": nn.ReLU,
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),  # Pass dict directly
    },
    device=device,  # Explicitly set device (CPU or GPU)
)

print(f"Running on: {device}")