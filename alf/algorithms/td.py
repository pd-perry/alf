from turtle import pd
from torch._C import Value
from alf.algorithms.config import TrainerConfig
from alf.algorithms.multiagent_one_step_loss import MultiAgentOneStepTDLoss
from alf.algorithms.one_step_loss import OneStepTDLoss
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
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.networks.relu_mlp import ReluMLP
from alf.data_structures import AlgStep, LossInfo, namedtuple, StepType, TimeStep
from alf.utils import common, value_ops, tensor_utils
from alf.utils.losses import element_wise_squared_loss

SeedTDInfo = namedtuple("TInfo", ["target_value", "action", "reward", "value",
                             "prev_obs", "step_type", "discount", "noise", "observation"], default_value=())

SeedTDState = namedtuple("TState", ["prev_obs"], default_value=())


@alf.configurable
class TD(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec=TensorSpec(()),
                 action_spec=TensorSpec(()),
                 reward_spec=TensorSpec(()),
                 q_network_cls=QNetwork,
                 gamma=0.99,
                 epsilon_greedy=0.0,
                 max_episodic_reward=100,
                 target_update_tau=0.05,
                 target_update_period=1,
                 bootstrap=False,
                 env=None,
                 optimizer=None,
                 config: TrainerConfig = None,
                 debug_summaries=True,
                 name="TD"):
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
            gamma (float): the discount factor for future returns.
            max_noise_buf_length (int): the maximum length of the noise buffer. This number 
                can exceed the number of elements that can be in the replay buffer. When this 
                is the case, the extra noise terms will not be used. 
            max_episodic_reward (int): the maximum reward each episode. Used to calculate
                the cumulated training regret. 
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
        if bootstrap:
            bootstrap_index = np.random.randint(0, config.replay_buffer_length - 1, size=(config.num_parallel_agents, config.replay_buffer_length))
            bootstrap_index = torch.from_numpy(bootstrap_index)
            bootstrap_index = torch.sort(bootstrap_index, dim=-1).values
        else:
            bootstrap_index = None
        self._bootstrap = bootstrap
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         train_state_spec=SeedTDState(),
                         reward_spec=reward_spec,
                         env=env,
                         bootstrap_index=bootstrap_index,
                         optimizer=optimizer,
                         config=config,
                         debug_summaries=debug_summaries,
                         name=name)

        # assert action_spec.is_discrete, (
        #     "Only support discrete actions")

        self.num_actions = action_spec.maximum - action_spec.minimum + 1
        q_network_cls = QNetwork(input_tensor_spec=observation_spec, action_spec=action_spec)
        self._network = q_network_cls.make_parallel(config.num_parallel_agents)
        self._target_network = self._network.copy(
            name='target_networks')
        
        if env is not None:
            metric_buf_size = max(self._config.metric_min_buffer_size,
                                  self._env.batch_size)
            example_time_step = env.reset()
            self._metrics.append(alf.metrics.AverageRegretMetric(
                    max_episodic_reward=max_episodic_reward,
                    buffer_size=metric_buf_size,
                    example_time_step=example_time_step
                ))

        self._gamma = gamma
        self._epsilon_greedy = epsilon_greedy
        self._config = config
        
        self._update_target = common.TargetUpdater(
            models=[self._network],
            target_models=[self._target_network],
            tau=target_update_tau,
            period=target_update_period)
    
    # def predict_step(self, info, state: SeedTDState):
    #     value, _ = self._network(info.observation)
    #     if self._config.num_parallel_agents > 1:
    #         action =  torch.argmax(value, dim=-1)
    #         action = torch.diagonal(action, 0)
    #     else:
    #         #single agent is good
    #         action = torch.argmax(value, dim=-1)
    #         action = action.squeeze()
            

    #     if self._config.num_parallel_agents > 1:
    #         action = action[:, 0]
        
    #     return AlgStep(output=action,
    #                    state=state,
    #                    info=SeedTDInfo(action=action))
    
    def predict_action(self, observation):
        value, _ = self._network(observation)
        action = torch.argmax(value, dim=2)
        
        if self._config.num_parallel_agents > 1:
            action = action[:, 0]
        
        return action
    

    def rollout_step(self, inputs: TimeStep, state: SeedTDState):
        value, _ = self._network(inputs.observation)

        #currently no need for epsilon greedy, also does not work
        e = random.randint(0, 100)
        if e > self._epsilon_greedy*100:
            if self._config.num_parallel_agents > 1:
                action =  torch.argmax(value, dim=-1)
                action = torch.diagonal(action, 0)
            else:
                #single agent is good
                action = torch.argmax(value, dim=-1)
                action = action.squeeze()
            
        else:
            action = torch.tensor([random.randint(
                self._action_spec.minimum, self._action_spec.maximum) for i in range(inputs.observation.shape[0])])

        #TODO: how to find next state
        return AlgStep(output=action,
                       state=state,
                       info=SeedTDInfo(action=action,
                                  reward=inputs.reward,
                                  step_type=inputs.step_type,
                                  discount=inputs.discount,
                                  observation=inputs.observation))

    def train_step(self, inputs: TimeStep, state, rollout_info: SeedTDInfo):
        #inputs.observation is the same as rollout_info.observation
        #single agent is good
        target_value, _ = self._target_network(inputs.observation)
        action = torch.argmax(target_value, dim=-1)
        action = action.to(torch.int64)

        target_value = target_value.gather(dim=2, index=action.unsqueeze(2))
        target_value = torch.squeeze(target_value)

        value, _ = self._network(inputs.observation)
        rollout_action = rollout_info.action.repeat([value.shape[1], 1])
        rollout_action = rollout_action.transpose(0, 1)
        value = value.gather(dim=-1, index=rollout_action.unsqueeze(2))
        value = torch.squeeze(value)

        return AlgStep(output=action,
                       state=SeedTDState(),
                       info=SeedTDInfo(discount=inputs.discount,
                                  step_type=inputs.env_id,
                                  action=action,
                                  reward=inputs.reward,
                                  target_value=target_value,
                                  value=value))
    
    def after_update(self, root_inputs, info: SeedTDInfo):
        self._update_target()

    def calc_loss(self, info: SeedTDInfo):
        if self._config.num_parallel_agents > 1: 
            loss_fn = MultiAgentOneStepTDLoss(gamma=self._gamma, debug_summaries=True)
            loss = loss_fn(info, value=info.value, target_value=info.target_value, noise=None, bootstrap=self._bootstrap)            
        else:
            # import pdb; pdb.set_trace()
            # single agent is good
            loss_fn = OneStepTDLoss(gamma=self._gamma, debug_summaries=True)
            loss = loss_fn(info=info, value=info.value, target_value=info.target_value)

        
        """info.value, returns after extending, and loss should be the same shape"""
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("reward", torch.mean(info.reward))

        return LossInfo(loss=loss.loss)
    

    