import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.trac_algorithm import TracAlgorithm
from alf.algorithms.data_transformer import RewardScaling
from alf.environments import suite_simple

from alf.environments.bipolar_chains import BipolarChains
from alf.environments.suite_simple import load
import alf.environments.suite_bsuite as bsuite
from bsuite import sweep

from alf.algorithms.seedtd_algorithm import SeedTDAlgorithm

from alf.environments.utils import create_environment


# environment config
# alf.config(
#     'create_environment', env_name=BipolarChains, env_load_fn=load, num_parallel_environments=87)

alf.config(
    'create_environment', env_name=sweep.CARTPOLE_SWINGUP[0], env_load_fn=bsuite.load, num_parallel_environments=5)

# reward scaling
alf.config('TrainerConfig', data_transformer_ctor=RewardScaling)
alf.config('RewardScaling', scale=0.01)

# algorithm config
alf.config('QNetwork', fc_layer_params=(50, 50 ))
alf.config(
    'SeedTDAlgorithm',
    optimizer=alf.optimizers.Adam(lr=5e-2, weight_decay=1e-5))

alf.config(
    'TrainerConfig',
    initial_collect_steps=6,
    unroll_length=1,
    num_parallel_agents=5,
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