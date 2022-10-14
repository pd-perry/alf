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

TInfo = namedtuple("TInfo", ["state", "action", "reward", "value",
                             "prev_obs", "step_type", "discount", "noise", "observation"], default_value=())

TState = namedtuple("TState", ["prev_obs"], default_value=())


@alf.configurable
class SeedTD(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec=TensorSpec(()),
                 action_spec=TensorSpec(()),
                 reward_spec=TensorSpec(()),
                 qnetwork=QNetwork,
                 learning_rate=0.003,
                 v=0.01,
                 gamma=0.99,
                 env=None,
                 optimizer=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="SeedTD"):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         train_state_spec=TState(),
                         reward_spec=reward_spec,
                         env=env,
                         optimizer=optimizer,
                         config=config,
                         debug_summaries=debug_summaries,
                         name=name)

        self.num_actions = action_spec.maximum - action_spec.minimum + 1
        qnetwork = QNetwork(input_tensor_spec=observation_spec, action_spec=action_spec)
        self._network = qnetwork.make_parallel(config.num_parallel_agents)
        self._noise = torch.normal(0, v, (1000, config.num_parallel_agents, config.num_parallel_agents))
        self._noise_index = 0

        self._lr = learning_rate
        self._v = v

        self._gamma = gamma
        self._epsilon_greedy = config.epsilon_greedy
        self._config = config

    def rollout_step(self, inputs: TimeStep, state: TState):
        
        value, _ = self._network(inputs.observation)
        
        action = torch.argmax(value, dim=2)
        action = torch.diagonal(action, 0)

        noise = self._noise[self._noise_index, :, :]
        self._noise_index = self._noise_index + 1

        return AlgStep(output=action,
                       state=state,
                       info=TInfo(state=inputs.observation,
                                  action=action,
                                  reward=inputs.reward,
                                  step_type=inputs.step_type,
                                  discount=inputs.discount,
                                  value=value,
                                  noise=noise))

    def train_step(self, inputs: TimeStep, state, rollout_info: TInfo):
        new_value, _ = self._network(inputs.observation)
        action = torch.argmax(new_value, dim=2)
        action = action.to(torch.int64)

        new_value = new_value.gather(2, action.unsqueeze(2))
        new_value = torch.squeeze(new_value)

        ### [152, 10, 1]
        return AlgStep(output=action,
                       state=TState(),
                       info=TInfo(discount=inputs.discount,
                                  step_type=inputs.step_type,
                                  action=action,
                                  reward=inputs.reward,
                                  value=new_value,
                                  noise=rollout_info.noise,
                                  observation=inputs.observation))

    def calc_loss(self, info: TInfo):
        if self._config.num_parallel_agents == 1:
            gaussian_noise = info.noise

            returns = value_ops.discounted_return(
            info.reward, info.value, info.step_type, info.discount * self._gamma)
            returns = tensor_utils.tensor_extend(returns, info.value[-1])
            returns = returns + gaussian_noise
            loss = element_wise_squared_loss(returns, torch.squeeze(info.value))
        else:
            value = torch.reshape(info.value, (info.value.shape[0], info.value.shape[1], -1))

            gaussian_noise = info.noise

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
    
    def predict_step(self, info, state: TState):
        value, _ = self._network(info.observation)
        action = torch.argmax(value, dim=2)
        
        if self._config.num_parallel_agents > 1:
            action = action[:, 0]
        
        return AlgStep(output=action,
                       state=state,
                       info=TInfo(action=action))
    