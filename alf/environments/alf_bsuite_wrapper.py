import bsuite
from bsuite.utils import gym_wrapper
from bsuite import sweep

from alf.environments.alf_gym_wrapper import AlfGymWrapper


class AlfBsuiteWrapper(AlfGymWrapper):
    def __init__(self,
                 env_id=None,
                 discount=1.0,
                 auto_reset=True,
                 simplify_box_bounds=True):
        if env_id == None:
            self._env_id = 0
        else:
            self._env_id = env_id
        self._env = bsuite.load_from_id('catch/' + str(self._env_id))
        self._gym_env = gym_wrapper.GymFromDMEnv(self._env)

        super(AlfBsuiteWrapper, self).__init__(self._gym_env,
                                               env_id=env_id,
                                               discount=discount,
                                               auto_reset=auto_reset,
                                               simplify_box_bounds=simplify_box_bounds)
