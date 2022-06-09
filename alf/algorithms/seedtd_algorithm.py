from alf.algorithms.config import TrainerConfig
from alf.networks.q_networks import QNetwork
from alf.networks.value_networks import ValueNetwork
from alf.tensor_specs import TensorSpec
from alf.algorithms.config import TrainerConfig
import torch
import torch.nn as nn
import numpy as np
import random

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.networks.relu_mlp import ReluMLP
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep
from alf.utils import value_ops, tensor_utils
from alf.utils.losses import element_wise_squared_loss

SeedTDInfo = namedtuple("TInfo", ["state", "action", "reward", "value",
                             "prev_obs", "step_type", "discount", "noise", "observation"], default_value=())

SeedTDState = namedtuple("TState", ["prev_obs"], default_value=())


@alf.configurable
class SeedTDAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec=TensorSpec(()),
                 action_spec=TensorSpec(()),
                 reward_spec=TensorSpec(()),
                 q_network_cls=QNetwork,
                 v=0.01,
                 gamma=0.99,
                 max_noise_buf_length=1000,
                 env=None,
                 optimizer=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="SeedTDAlgorithm"):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         train_state_spec=SeedTDState(),
                         reward_spec=reward_spec,
                         env=env,
                         optimizer=optimizer,
                         config=config,
                         debug_summaries=debug_summaries,
                         name=name)
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            q_network_cls (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            optimizer: The optimizer used for the network.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """

        self.num_actions = action_spec.maximum - action_spec.minimum + 1
        q_network_cls = QNetwork(input_tensor_spec=observation_spec, action_spec=action_spec)
        self._network = q_network_cls.make_parallel(config.num_parallel_agents)
        self._noise = torch.normal(0, v, (max_noise_buf_length, config.num_parallel_agents, config.num_parallel_agents))
        self._noise_index = 0

        env = self._env
        if env is not None:
            metric_buf_size = max(self._config.metric_min_buffer_size,
                                  self._env.batch_size)
            example_time_step = env.reset()
            self._metrics = [
                alf.metrics.NumberOfEpisodes(),
                alf.metrics.EnvironmentSteps(),
                alf.metrics.AverageReturnMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step),
                alf.metrics.AverageEpisodeLengthMetric(
                    example_time_step=example_time_step,
                    buffer_size=metric_buf_size),
                alf.metrics.AverageEnvInfoMetric(
                    example_time_step=example_time_step,
                    buffer_size=metric_buf_size),
                alf.metrics.AverageDiscountedReturnMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step),
                alf.metrics.AverageRegretMetric(
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step
                )
            ]

        self._gamma = gamma
        self._epsilon_greedy = config.epsilon_greedy
        self._config = config
    
    def predict_step(self, info, state: SeedTDState):
        value, _ = self._network(info.observation)
        action = torch.argmax(value, dim=2)
        
        if self._config.num_parallel_agents > 1:
            action = action[:, 0]
        
        return AlgStep(output=action,
                       state=state,
                       info=SeedTDInfo(action=action))
    
    def predict_action(self, observation):
        value, _ = self._network(observation)
        action = torch.argmax(value, dim=2)
        
        if self._config.num_parallel_agents > 1:
            action = action[:, 0]
        
        return action
    

    def rollout_step(self, inputs: TimeStep, state: SeedTDState):
        
        value, _ = self._network(inputs.observation)
        
        action = torch.argmax(value, dim=-1)
        action = torch.diagonal(action, 0)

        noise = self._noise[self._noise_index, :, :]
        self._noise_index = self._noise_index + 1

        return AlgStep(output=action,
                       state=state,
                       info=SeedTDInfo(state=inputs.observation,
                                  action=action,
                                  reward=inputs.reward,
                                  step_type=inputs.step_type,
                                  discount=inputs.discount,
                                  value=value,
                                  noise=noise))

    def train_step(self, inputs: TimeStep, state, rollout_info: SeedTDInfo):
        new_value, _ = self._network(inputs.observation)
        action = torch.argmax(new_value, dim=-1)
        action = action.to(torch.int64)

        new_value = new_value.gather(2, action.unsqueeze(2))
        new_value = torch.squeeze(new_value)

        ### [152, 10, 1]
        return AlgStep(output=action,
                       state=SeedTDState(),
                       info=SeedTDInfo(discount=inputs.discount,
                                  step_type=inputs.step_type,
                                  action=action,
                                  reward=inputs.reward,
                                  value=new_value,
                                  noise=rollout_info.noise))

    def calc_loss(self, info: SeedTDInfo):
        gaussian_noise = info.noise
        if self._config.num_parallel_agents == 1:
            returns = value_ops.discounted_return(
            info.reward, info.value, info.step_type, info.discount * self._gamma)
            returns = tensor_utils.tensor_extend(returns, info.value[-1])
            returns = returns + gaussian_noise
            loss = element_wise_squared_loss(returns, torch.squeeze(info.value))
        else:
            value = torch.reshape(info.value, (info.value.shape[0], info.value.shape[1], -1))

            returns = torch.zeros((info.value.shape))
            for n in range(info.value.shape[2]):
                returns_n = value_ops.discounted_return(info.reward, value[:, :, n], info.step_type, info.discount * self._gamma)
                returns_n = tensor_utils.tensor_extend(returns_n, info.value[-1, :, n])
                returns[:, :, n] = returns_n

            returns = returns + gaussian_noise
            loss = element_wise_squared_loss(returns, info.value)
            loss = loss.sum(dim=-1)

        """info.value, returns after extending, and loss should be the same shape"""

        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("reward", torch.mean(info.reward))
                alf.summary.scalar("returns", torch.mean(returns))

        return LossInfo(loss=loss)