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

import gym
from gym import spaces

import numpy as np


class BipolarChains(gym.Env):

    def __init__(self, N=50):
        super().__init__()
        assert N%2 == 0

        self._N = N
        self._root = N/2
        self._position = int(N/2)
        self._game_over = False

        # self._sign = np.sign(np.random.random() - 0.5)
        self._sign = 1

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(N, ), dtype=np.float32)
        # self.observation_space = spaces.Discrete(self._N)
        self.action_space = spaces.Discrete(2)
        self.reset()

    def reset(self):
        # reset to position N/2
        self._position = int(self._root)
        self._game_over = False
        obs = np.zeros((self._N,))
        obs[self._position] = 1
        return obs

    def step(self, action):
        # auto_reset will be called by wrappers
        observation, reward = self._gen_observation(action)
        return observation, reward, self._game_over, {}
    
    def render(self, mode="human", close=False):
        return

    def _gen_observation(self, action):
        #TODO sample either left or right positive reward
        if action == 0:
            if self._position != 1:
                reward = -1
            else:
                reward = self._sign * self._N
                self._game_over = True
            self._position = self._position - 1
        elif action == 1:
            if self._position != self._N - 2:
                reward = -1
            else:
                reward = -self._sign * self._N
                self._game_over = True
            self._position = self._position + 1
        observation = np.zeros((self._N,))
        observation[self._position] = 1

        return observation, reward
