# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math

import alf
from alf.algorithms.tsabc_algorithm import TsabcAlgorithm
from alf.algorithms.multi_bootstrap_ensemble import MultiBootstrapEnsemble
from alf.algorithms.multiswag_algorithm import MultiSwagAlgorithm
from alf.networks import NormalProjectionNetwork, ActorNetwork, ActorDistributionNetwork
from alf.optimizers import Adam, AdamTF
from alf.utils.losses import element_wise_squared_loss
from alf.utils.math_ops import clipped_exp

from alf.examples import sac_conf

# experiment settings
use_multibootstrap = True
deterministic_actor = False
deterministic_critic = False
state_dependent_std = True
use_epistemic_alpha = False
critic_training_weight = None
use_q_mean_train_actor = True

# if env_name() == "Humanoid-v2":
#     fixed_alpha = 0.05 
# else:
#     fixed_alpha = 0.2

# environment config
alf.config(
    'create_environment',
    env_name="HalfCheetah-v2",
    num_parallel_environments=1)

# # algorithm config
# @alf.configurable
# def tsabc_deterministic_actor(value: bool = True):
#     return value
# # environment config
# alf.config('tsabc_deterministic_actor',
#            value=True)

# algorithm config
fc_layer_params = (256, 256)
joint_fc_layer_params = (256, 256)
# deterministic_actor = tsabc_deterministic_actor()

if deterministic_actor:
    actor_network_cls = partial(ActorNetwork, fc_layer_params=fc_layer_params)
else:
    actor_network_cls = partial(
        ActorDistributionNetwork,
        fc_layer_params=fc_layer_params,
        continuous_projection_net_ctor=partial(
            NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True,
            std_transform=clipped_exp))

alf.config('calc_default_target_entropy', min_prob=0.184)

# explore_network_cls = partial(
#     ActorDistributionNetwork,
#     fc_layer_params=fc_layer_params,
#     continuous_projection_net_ctor=partial(
#         NormalProjectionNetwork,
#         state_dependent_std=True,
#         scale_distribution=True,
#         std_transform=clipped_exp))

explore_network_cls = partial(ActorNetwork, fc_layer_params=fc_layer_params)

alf.config(
    'CriticDistributionParamNetwork',
    joint_fc_layer_params=joint_fc_layer_params,
    state_dependent_std=state_dependent_std)

# alf.config(
#     'FuncParVIAlgorithm',
#     num_particles=10)
# alf.config(
#     'MultiSwagAlgorithm',
#     num_samples_per_model=1,
#     subspace_max_rank=20,
#     subspace_after_update_steps=10000)

alf.config(
    'MultiBootstrapEnsemble',
    num_basins=5,  # grid search
    num_particles_per_basin=3,  # grid search
    initial_train_steps=1000000)

alf.config(
    'TsabcAlgorithm',
    actor_network_cls=actor_network_cls,
    explore_network_cls=explore_network_cls,
    critic_module_cls=MultiBootstrapEnsemble,
    beta_ub=1.,
    beta_lb=1.,  # grid search
    initial_log_alpha=math.log(0.2),
    # entropy_regularization_weight=1.,
    deterministic_actor=deterministic_actor,
    deterministic_critic=deterministic_critic,
    critic_training_weight=critic_training_weight,
    common_td_target=True,  # grid search
    use_q_mean_train_actor=use_q_mean_train_actor,
    use_entropy_reward=False,
    use_epistemic_alpha=use_epistemic_alpha,
    use_basin_mean_for_target_critic=True,  # grid search
    random_actor_every_step=False,
    target_update_tau=0.005,
    actor_optimizer=AdamTF(lr=3e-4),
    explore_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=Adam(lr=3e-4),  #, weight_decay=1e-4),
    alpha_optimizer=None,  # None means fixed alpha
    explore_alpha_optimizer=AdamTF(lr=3e-4))

alf.config('OneStepTDLoss', td_error_loss_fn=element_wise_squared_loss)

# training config
alf.config('Agent', rl_algorithm_cls=TsabcAlgorithm)

alf.config('TrainerConfig',
           version='normal',
           use_wandb=True,
           async_eval=True,
           entity="jiachenli",
           project="bayesian-critics",
           initial_collect_steps=10000,
           mini_batch_length=2,
           unroll_length=1,
           mini_batch_size=256,
           num_updates_per_train_iter=1,
           num_iterations=2500000,
           num_checkpoints=1,
           evaluate=True,
           eval_interval=1000,
           num_eval_episodes=5,
           debug_summaries=True,
           random_seed=1,
           summarize_grads_and_vars=True,
           summary_interval=1000,
           replay_buffer_length=int(1e6))
