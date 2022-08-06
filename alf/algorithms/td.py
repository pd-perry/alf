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
from alf.utils import common, summary_utils, value_ops, tensor_utils
from alf.utils.losses import element_wise_squared_loss
from alf.utils.math_ops import add_ignore_empty

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
            self._metrics.append(alf.metrics.CumulativeRegretMetric(
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
                #single agent
                action = torch.argmax(value, dim=-1)
                action = action.squeeze()
            
        else:
            action = torch.tensor([random.randint(
                self._action_spec.minimum, self._action_spec.maximum) for i in range(inputs.observation.shape[0])])

        #TODO: should value be stored from rollout?
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

        # import pdb; pdb.set_trace()

        return AlgStep(output=action,
                       state=SeedTDState(),
                       info=SeedTDInfo(discount=inputs.discount,
                                  step_type=inputs.step_type,
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
    
    def _trainable_attributes_to_ignore(self):
        return ['_target_network']
    
    def update_with_gradient(self, loss_info, valid_masks=None, weight=1, batch_info=None):
        """Overides update_with_gradient from algorithm.py
        Complete one iteration of training.

        Update parameters using the gradient with respect to ``loss_info``.

        Args:
            loss_info (LossInfo): loss with shape :math:`(T, B, N)`, where N 
                is the number of agents (except for ``loss_info.scalar_loss``)
            valid_masks (Tensor): masks indicating which samples are valid.
                (``shape=(T, B), dtype=torch.float32``)
            weight (float): weight for this batch. Loss will be multiplied with
                this weight before calculating gradient.
            batch_info (BatchInfo): information about this batch returned by
                ``ReplayBuffer.get_batch()``
        Returns:
            tuple:
            - loss_info (LossInfo): loss information.
            - params (list[(name, Parameter)]): list of parameters being updated.
        """

        if self._debug_summaries:
            summary_utils.summarize_per_category_loss(loss_info)

        loss_info = self.aggregate_loss(loss_info, valid_masks, batch_info)
        all_params = self._backward_and_gradient_update(
                loss_info.loss * weight)
        return loss_info, all_params
    
    def aggregate_loss(self, loss_info, valid_masks=None, batch_info=None):
        """Computed aggregated loss.

        Args:
            loss_info (LossInfo): loss with shape :math:`(T, B, N)` (except for
                ``loss_info.scalar_loss``)
            valid_masks (Tensor): masks indicating which samples are valid.
                (``shape=(T, B), dtype=torch.float32``)
            batch_info (BatchInfo): information about this batch returned by
                ``ReplayBuffer.get_batch()``
        Returns:
            loss_info (LossInfo): loss information, with the aggregated loss
                in the ``loss`` field (i.e. ``loss_info.loss``).
        """
        masks = valid_masks
        if self._bootstrap:
            masks = masks.reshape((2, -1, loss_info.loss.shape[-1])) #[T, B, N]
        else:
            masks = masks.unsqueeze(-1).repeat(1, 1, loss_info.loss.shape[-1]) #duplicates masks once for each
        if masks is not None:
            loss_info = alf.nest.map_structure(
                lambda l: torch.sum(torch.mean(l * masks, dim=(0, 1))), loss_info)
        else:
            loss_info = alf.nest.map_structure(lambda l: torch.mean(l),
                                               loss_info)
        if isinstance(loss_info.scalar_loss, torch.Tensor):
            assert len(loss_info.scalar_loss.shape) == 0
            loss_info = loss_info._replace(
                loss=add_ignore_empty(loss_info.loss, loss_info.scalar_loss))
        return loss_info
    