# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Tests for sarsa_algorithm.py."""

from absl import logging
from absl.testing import parameterized
import functools
from functools import partial
import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_algorithm_test import unroll
from alf.algorithms.seedtd_algorithm import SeedTDAlgorithm
from alf.environments.suite_unittest import ActionType, PolicyUnittestEnv
from alf.networks.q_networks import QNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils.math_ops import clipped_exp

from alf.environments.suite_bsuite import load
from alf.environments.utils import create_environment
import alf.environments.suite_bsuite as bsuite
from bsuite import sweep


def _create_algorithm(env):
    env = create_environment(env_name=sweep.CARTPOLE_SWINGUP[0], env_load_fn=bsuite.load, num_parallel_environments=1)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    fc_layer_params = (50, 50)

    network = partial(
            alf.nn.QNetwork,
            fc_layer_params=fc_layer_params)

    config = TrainerConfig(
        root_dir="dir",
        initial_collect_steps=6,
        unroll_length=1,
        num_parallel_agents=1,
        algorithm_ctor=SeedTDAlgorithm,
        num_iterations=150,
        num_checkpoints=5,
        evaluate=False,
        eval_interval=20,
        debug_summaries=True,
        num_updates_per_train_iter=1,
        whole_replay_buffer_training=True,
        clear_replay_buffer=False,
        replay_buffer_length=1000,
        summarize_grads_and_vars=False,
        summary_interval=1,
        random_seed=2)

    return SeedTDAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        q_network_cls=network,
        env=env,
        config=config,
        debug_summaries=True)


class SeedTDTest(parameterized.TestCase, alf.test.TestCase):
    # TODO: on_policy=True is very unstable, try to figure out the possible
    # reason.
    def test_seedTD(self):
        env_class = PolicyUnittestEnv
        iterations = 500
        num_env = 128
        steps_per_episode = 12
        env = env_class(
            num_env, steps_per_episode, action_type=ActionType.Continuous)
        eval_env = env_class(
            100, steps_per_episode, action_type=ActionType.Continuous)

        algorithm = _create_algorithm(env)
        eval_env = create_environment(env_name=sweep.CARTPOLE_SWINGUP[0], env_load_fn=bsuite.load, num_parallel_environments=1)

        env.reset()
        eval_env.reset()
        for i in range(iterations):
            algorithm.train_iter()

            eval_env.reset()
            eval_time_step = unroll(eval_env, algorithm, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            0, float(eval_time_step.reward.mean()), delta=0.3)


def unroll(env, algorithm, steps):
    """Run `steps` environment steps using QrsacAlgorithm._predict_action()."""
    time_step = common.get_initial_time_step(env)
    trans_state = algorithm.get_initial_transform_state(algorithm._env.batch_size)
    for _ in range(steps):
        transformed_time_step, trans_state = algorithm.transform_timestep(
            time_step, trans_state)
        action = algorithm.predict_action(transformed_time_step.observation)
        time_step = env.step(torch.reshape(action, (1, )))
    return time_step

if __name__ == '__main__':
    alf.test.main()
