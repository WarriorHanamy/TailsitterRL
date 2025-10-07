from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .droneEnv import DroneEnvsBase
from vtol_rl.utils.type import ACTION_TYPE, TensorDict
from vtol_rl.utils.logger import setup_logger


class TailsitterEnvsBase(VecEnv):
    logger = setup_logger(__name__)

    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        max_episode_steps: int = 1000,
        seed: int = 42,  # control the randomzation of envs for consistency
        device: torch.device = torch.device("cpu"),
        dynamics_kwargs: Mapping[str, Any] | None = None,
        random_kwargs: Mapping[str, Any] | None = None,
        requires_grad: bool = False,
        scene_kwargs: Mapping[str, Any] | None = None,
        sensor_kwargs: Sequence[Mapping[str, Any]] | None = None,
        tensor_output: bool = True,
        is_train: bool = False,
    ):
        dynamics_kwargs = {} if dynamics_kwargs is None else dict(dynamics_kwargs)
        random_kwargs = {} if random_kwargs is None else dict(random_kwargs)
        scene_kwargs = {} if scene_kwargs is None else dict(scene_kwargs)
        sensor_kwargs = list(sensor_kwargs) if sensor_kwargs is not None else []

        super(VecEnv, self).__init__()

        # raise Warning if device is cuda while num_envs is less than 1e3
        device = torch.device(device)
        if num_agent_per_scene * num_scene < 1e3 and (device.type == "cuda"):
            _env_device = torch.device("cpu")
            TailsitterEnvsBase.logger.warning(
                "The number of envs is less than 1e3, cpu is faster than gpu. To make training faster, we have set device to cpu."
            )
        else:
            _env_device = device

        self.envs = DroneEnvsBase(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            device=_env_device,  # because at least under 1e3 (common useful range) envs, cpu is faster than gpu
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
        )

        self.device = device

        self.num_agent = num_agent_per_scene * num_scene
        self.num_scene = num_scene
        self.num_agent_per_scene = num_agent_per_scene
        self.num_envs = self.num_agent

        self.requires_grad = requires_grad
        self.max_sense_radius = 10

        self.tensor_output = tensor_output
        self.is_train = is_train

        # key interference of gym env
        state_size = 3 + 4 + 3 + 3

        # REC MARK: multi-modal observation space construction, though it is not implemented here yet (images)
        # self.observation_space = spaces.Dict({
        #     "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32),
        #     "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        # })
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
                )
            }
        )

        # if latent_dim is not None:
        #     # self.observation_space["latent"] = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)
        #     self.latent = torch.zeros((self.num_envs, latent_dim), device=self.device)

        # latent variables
        # REC MARK: deterministic and stochastic
        self.deter = None
        self.stoch = None

        # REC MARK: remove for simplicity.
        if self.envs.dynamics.action_type == ACTION_TYPE.BODYRATE:
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            raise ValueError(
                "action_type should be one of ['bodyrate', 'thrust', 'velocity', 'position'], but got {}".format(
                    self.envs.dynamics.action_type
                )
            )

        self._step_count = torch.zeros((self.num_agent,), dtype=torch.int32)
        self._reward = torch.zeros((self.num_agent,))
        self._rewards = torch.zeros((self.num_agent,))
        self._action = torch.zeros((self.num_agent, 4))

        # TensorDict is a dict for tensor.
        self._observations = TensorDict({})

        self._success = torch.zeros(self.num_agent, dtype=bool)
        self._failure = torch.zeros(self.num_agent, dtype=bool)
        self._episode_done = torch.zeros(self.num_agent, dtype=bool)
        self._done = torch.zeros(self.num_agent, dtype=bool)
        self._info = [{"TimeLimit.truncated": False} for _ in range(self.num_agent)]

        # For convenience of intuitive visualization of reward components
        # self._indiv_reward = self.get_reward()  # 例如 {"reward": 1.0, "speed": 0.5, "penalty": -0.2}
        self._indiv_rewards: dict[str, torch.Tensor] | None = None
        self._indiv_reward: dict[str, torch.Tensor] | None = None
        self.max_episode_steps = max_episode_steps

        # necessary for gym compatibility
        self.render_mode = ["None" for _ in range(self.num_agent)]

        self._is_initial = False

    def step(self, _action, is_test=False, latent_func=None):
        """
        Args:
            _action: (torch.Tensor) shape (num_envs, action_dim)
            is_test: (bool) whether the env is in test mode.
            latent_func: (callable) function to update latent variables.
        Returns:
            observation: (dict or TensorDict) shape (num_envs, obs_dim)
            reward: (torch.Tensor) shape (num_envs,)
            done: (torch.Tensor) shape (num_envs,)
            info: (list) length num_envs
        """
        assert self._is_initial, "You should call reset() before step()"
        self._action = (
            _action if isinstance(_action, torch.Tensor) else torch.as_tensor(_action)
        )
        assert (
            self._action.abs().max() <= 1.0
        ), f"The action should be in range [-1, 1], but got {self._action}"
        # update state and observation and _done
        print(f"high-level call to step action.shape: {_action.shape}")
        self.envs.step(self._action)
        self.get_full_observation()
        if latent_func is not None:
            self.update_latent(latent_func=latent_func)

        self._step_count += 1

        # update success _done
        self._success = self.get_success()
        self._failure = self.get_failure()
        assert self._success.dtype == torch.bool and self._failure.dtype == torch.bool

        # REC MARK: 初始化时机​：
        # _indiv_rewards通常在第一次调用 step()时初始化，以匹配 _indiv_reward的键。
        # update _rewards
        if self._indiv_reward is None:
            self._reward = self.get_reward()
        else:
            self._indiv_reward = self.get_reward()
            assert (
                isinstance(self._indiv_reward, dict)
                and "reward" in self._indiv_reward.keys()
            )
            self._reward = self._indiv_reward["reward"]
            for key in self._indiv_reward.keys():
                self._indiv_rewards[key] += self._indiv_reward[key].detach()
        self._rewards += self._reward

        # update collision, timeout _done
        self._episode_done = (
            self._episode_done | self._success | self._failure | self.is_collision
        )
        # self._episode_done = self._episode_done | self._success | self._failure

        self._done = self._episode_done | (self._step_count >= self.max_episode_steps)

        # update and record _info: episode, timeout
        # full_state = self.full_state
        for indice in range(self.num_agent):
            # self._info[indice]["state"]= full_state[indice].cpu().clone().detach().numpy()
            # i don't know why, but whatever this returned info data address should be strictly independent with torch.
            if self._done[indice]:
                self._info[indice] = self.collect_info(indice, self._observations)

        # return and auto-reset
        _done, _reward, _info = (
            self._done.clone(),
            self._reward.clone(),
            self._info.copy(),
        )
        # _episode_done = self._episode_done.clone()
        # reset all the dead agents
        if self._done.any() and not is_test:
            self.examine()
        if self.requires_grad:
            # analytical gradient RL
            if not self.tensor_output:
                raise ValueError(
                    "requires_grad should be False if tensor_output is False"
                )
            return self._observations, _reward, _done, _info
        else:
            if self.tensor_output:
                return self._observations.detach(), _reward.detach(), _done, _info
            else:
                return (
                    self._observations,
                    _reward.cpu().numpy(),
                    _done.cpu().numpy().astype(np.int32),
                    _info,
                )

    @torch.no_grad()
    def update_latent(self, latent_func):
        next_stoch_post, next_deter = latent_func(
            action=self._action,
            stoch=self.stoch,
            deter=self.deter,
            deterministic=False,
            next_observation=self._observations,
            # return_prior=True,
        )
        self.deter = next_deter.to(self.device)
        self.stoch = next_stoch_post.to(self.device)
        self._observations["deter"] = self.deter.detach().cpu()
        self._observations["stoch"] = self.stoch.detach().cpu()

    def collect_info(self, indice, observations):
        _info = {}

        _info["episode_done"] = self._episode_done[indice].item()
        if self._success[indice]:
            _info["is_success"] = True
        else:
            _info["is_success"] = False

        step_count_val = self._step_count[indice].cpu().clone().detach().numpy()
        if step_count_val <= 2:
            Warning(
                "The length of the episode is too short, check the initial state randomization."
            )
        _info["episode"] = {
            "r": self._rewards[indice].cpu().clone().detach().numpy(),
            "l": step_count_val,
            "t": (self._step_count[indice] * self.envs.dynamics.ctrl_period)
            .cpu()
            .clone()
            .detach()
            .numpy(),
        }
        if self.requires_grad:
            _info["terminal_observation"] = {
                key: observations[key][indice].detach()
                if hasattr(observations[key][indice], "detach")
                else observations[key][indice]
                for key in observations.keys()
            }
        else:
            _info["terminal_observation"] = {
                key: observations[key][indice] for key in observations.keys()
            }

        if self._step_count[indice] >= self.max_episode_steps:
            _info["TimeLimit.truncated"] = True
        else:
            _info["TimeLimit.truncated"] = False

        _info["episode"]["extra"] = {}

        if self._indiv_rewards is not None:
            for key in self._indiv_rewards.keys():
                _info["episode"]["extra"][key] = (
                    self._indiv_rewards[key][indice].clone().detach()
                )

        return _info

    def initialize_latent(self, deter_dim, stoch_dim, world=None):
        self.deter = torch.zeros((self.num_agent, deter_dim), device=self.device)
        self.stoch = torch.zeros((self.num_agent, stoch_dim), device=self.device)
        self.observation_space["deter"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(deter_dim,), dtype=np.float32
        )
        self.observation_space["stoch"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(stoch_dim,), dtype=np.float32
        )

        if world:
            setattr(self, "world", world)

    def detach(self):
        self.envs.detach()
        self.simple_detach()

    def simple_detach(self):
        self._rewards = self._rewards.clone().detach()
        self._reward = self._reward.clone().detach()
        self._action = self._action.clone().detach()
        self._step_count = self._step_count.clone().detach()
        self._done = self._done.clone().detach()
        if hasattr(self, "latent"):
            self.latent = self.latent.clone().detach()
        if self.deter:
            self.deter = self.deter.clone().detach()
            self.stoch = self.stoch.clone().detach()

    def reset(self, state=None, obs=None, is_test=False):
        self._is_initial = True

        self.envs.reset()

        if isinstance(self.get_reward(), dict):
            self._indiv_reward: dict[str, torch.Tensor] = self.get_reward()
            self._indiv_rewards: dict[str, torch.Tensor] = self._indiv_reward
            self._indiv_rewards = {
                key: torch.zeros((self.num_agent,))
                for key in self._indiv_rewards.keys()
            }
            self._indiv_reward = {
                key: torch.zeros((self.num_agent,))
                for key in self._indiv_rewards.keys()
            }
        elif isinstance(self.get_reward(), torch.Tensor):
            self._indiv_rewards = None
            self._indiv_reward = None

        else:
            raise ValueError(
                "get_reward should return a dict or a tensor, but got {}".format(
                    type(self.get_reward())
                )
            )

        self.get_full_observation()
        self._reset_attr()
        self.get_full_observation()

        return self._observations

    def reset_env_by_id(self, scene_indices=None):
        assert not isinstance(scene_indices, bool)
        scene_indices = (
            scene_indices if scene_indices is not None else torch.arange(self.num_scene)
        )
        scene_indices = torch.atleast_1d(scene_indices)
        agent_indices = (
            (
                torch.tile(
                    torch.arange(self.num_agent_per_scene), (len(scene_indices), 1)
                )
                + (scene_indices.unsqueeze(1) * self.num_agent_per_scene)
            ).reshape(-1, 1)
        ).flatten()
        self.envs.reset_agents(agent_indices)
        self._reset_attr(indices=agent_indices)
        return self.get_full_observation(agent_indices)

    def reset_agent_by_id(self, agent_indices=None, state=None, reset_obs=None):
        assert ~(state is None and reset_obs is None) or (
            state is not None and reset_obs is not None
        )
        assert not isinstance(agent_indices, bool)
        self.envs.reset_agents(agent_indices, state=state)
        self.get_full_observation(agent_indices)
        self._reset_attr(indices=agent_indices)
        return self._observations

    def _format_obs(self, obs):
        if not self.tensor_output:
            return obs.detach().cpu().numpy()
        else:
            return obs

    @torch.no_grad()
    def _reset_attr(self, indices=None):
        if indices is None:
            self._reward = torch.zeros((self.num_agent,), device=self.device)
            self._rewards = torch.zeros((self.num_agent,), device=self.device)
            self._done = torch.zeros(self.num_agent, dtype=bool, device=self.device)
            self._episode_done = torch.zeros(
                self.num_agent, dtype=bool, device=self.device
            )
            self._step_count = torch.zeros(
                (self.num_agent,), dtype=torch.int32, device=self.device
            )
            if self.deter is not None:
                if hasattr(self, "world"):
                    latent = self.world.sequence_model.initial(self.num_agent)
                    # next_deter, next_stoch_post = latent["deter"], latent["stoch"]
                    next_stoch, next_deter = self.world.sequence_model(
                        action=torch.zeros(
                            (self.num_agent, 4), device=self.world.sequence_model.device
                        ),
                        stoch=latent["stoch"],
                        deter=latent["deter"],
                        deterministic=False,
                    )
                    next_stoch_post = self.world.encoder(
                        observation=self._observations,
                        deter=next_deter,
                        deterministic=False,
                    )
                    self.deter, self.stoch = (
                        next_deter.to(self.device),
                        next_stoch_post.to(self.device),
                    )
                else:
                    self.deter = torch.zeros_like(self.deter, device=self.device)
                    self.stoch = torch.zeros_like(self.stoch, device=self.device)
            if self._indiv_rewards is not None:
                for key in self._indiv_rewards.keys():
                    self._indiv_rewards[key] = torch.zeros(
                        (self.num_agent,), device=self.device
                    )
                    self._indiv_reward[key] = torch.zeros(
                        (self.num_agent,), device=self.device
                    )

        else:
            self._reward[indices] = 0
            self._rewards[indices] = 0
            self._done[indices] = False
            self._episode_done[indices] = False
            self._step_count[indices] = 0
            if self.deter is not None:
                if hasattr(self, "world"):
                    latent = self.world.sequence_model.initial(len(indices))
                    next_stoch, next_deter = self.world.sequence_model(
                        action=torch.zeros(
                            (len(indices), 4), device=self.world.sequence_model.device
                        ),
                        stoch=latent["stoch"],
                        deter=latent["deter"],
                        deterministic=False,
                    )
                    next_stoch_post = self.world.encoder(
                        observation=self._observations[indices],
                        deter=next_deter,
                        deterministic=False,
                    )
                    self.deter[indices], self.stoch[indices] = (
                        next_deter.to(self.device),
                        next_stoch_post.to(self.device),
                    )
                else:
                    self.deter[indices] = 0
                    self.stoch[indices] = 0
            if self._indiv_rewards is not None:
                for key in self._indiv_rewards.keys():
                    self._indiv_rewards[key][indices] = 0
                    self._indiv_reward[key][indices] = 0

        indices = range(self.num_agent) if indices is None else indices
        for indice in indices:
            self._info[indice] = {
                "TimeLimit.truncated": False,
                "episode_done": False,
            }

    def examine(self):
        if self._done.any():
            self.reset_agent_by_id(torch.where(self._done)[0])
        return self._observations

    def render(self, **kwargs):
        obs = self.envs.render(**kwargs)
        return obs

    def get_done(self):
        return torch.full((self.num_agent,), False, dtype=bool)

    @abstractmethod
    def get_success(self) -> torch.Tensor:
        _success = torch.full((self.num_agent,), False, dtype=bool)
        return _success

    def get_failure(self) -> torch.Tensor:
        _failure = torch.full((self.num_agent,), False, dtype=bool)
        return _failure

    @abstractmethod
    def get_reward(self) -> np.ndarray | torch.Tensor:
        _rewards = np.empty(self.num_agent)

        return _rewards

    @abstractmethod
    def get_observation(self, indice=None) -> TensorDict:
        raise NotImplementedError

    def get_full_observation(self, indice=None):
        obs = self.get_observation()
        assert isinstance(obs, TensorDict)

        if self.deter is not None:
            obs.update({"deter": self.deter})
            obs.update({"stoch": self.stoch})

        self._observations = self._format_obs(obs.as_tensor(device=self.device))
        return self._observations

    def close(self):
        self.envs.close()

    @property
    def reward(self):
        return self._reward

    @property
    def sensor_obs(self):
        return self.envs.sensor_obs

    @property
    def state(self):
        return self.envs.state

    @property
    def info(self):
        return self._info

    @property
    def is_collision(self):
        return self.envs.is_collision

    @property
    def done(self):
        return self._done

    @property
    def episode_done(self):
        return self._episode_done

    @property
    def success(self):
        return self._success

    @property
    def failure(self):
        return self._failure

    @property
    def direction(self):
        return self.envs.direction

    @property
    def position(self):
        return self.envs.position

    @property
    def orientation(self):
        return self.envs.orientation

    @property
    def velocity(self):
        return self.envs.velocity

    @property
    def angular_velocity(self):
        return self.envs.angular_velocity

    @property
    def t(self):
        return self.envs.t

    @property
    def visual(self):
        return self.envs.visual

    @property
    def collision_vector(self):
        return self.envs.collision_vector

    @property
    def collision_dis(self):
        return self.envs.collision_dis

    @property
    def collision_point(self):
        return self.envs.collision_point

    @property
    def full_state(self):
        return self.envs.full_state

    @property
    def dynamic_object_position(self):
        """
        Get the position of dynamic objects in the environment.
        :return: (Tensor) Position of dynamic objects, shape (num_envs, num_dynamic_objects, 3)
        """
        return self.envs.dynamic_object_position

    def env_is_wrapped(self):
        return False

    def step_async(self):
        raise NotImplementedError("This method is not implemented")

    def step_wait(self):
        raise NotImplementedError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        if indices is None:
            return getattr(self, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise NotImplementedError("This method is not implemented")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise NotImplementedError("This method is not implemented")

    def to(self, device):
        self.device = device if not isinstance(device, str) else torch.device(device)

    def eval(self):
        self.envs.eval()

    def __len__(self):
        return self.num_envs

    # brief description of the class
    def __repr__(self):
        return f"{self.__class__.__name__}(Env={self.envs.__class__},\
        NumAgentPerScene={self.num_agent_per_scene}, NumScene={self.num_scene}, \
        tensorOut={self.tensor_output}, RequiresGrad={self.requires_grad})"

    def set_requires_grad(self, requires_grad: bool):
        """
        Set whether the environment requires gradient computation.
        :param requires_grad: (bool) Whether to require gradients
        """
        self.requires_grad = requires_grad
