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

import unittest
from absl.testing import parameterized
from alf.environments.simple.bipolar_chains import BipolarChains

class BipolarChainTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters(4, 6)
    def test_bipolar_chain_environment(self, N):
        #Steps left until it terminates
        array = BipolarChains(N)
        array.reset()
        for _ in range(int(N/2)):
            step = array.step(0)
            done = step[2]
            reward1 = step[1]
            print(_)
        self.assertTrue(done)
        array.reset()

        #Steps right until it terminates
        for _ in range(int(N/2)-1):
            step = array.step(1)
            done = step[2]
            reward2 = step[1]
        self.assertTrue(done)

        self.assertEquals(reward1, -reward2)


if __name__ == "__main__":
    unittest.main()
