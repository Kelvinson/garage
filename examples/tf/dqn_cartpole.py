"""
An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole. And uses a DQN with
1M steps.
"""
import gym

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import EpsilonGreedyStrategy
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
        

def run_task(*_):
    """Run task."""
    max_path_length = 1
    n_epochs = int(1e5)

    env = TfEnv(normalize(gym.make("CartPole-v0")))

    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=int(5e4),
        time_horizon=max_path_length)

    qf = DiscreteMLPQFunction(
        env_spec=env.spec, hidden_sizes=(64, 64), dueling=False)

    policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

    epilson_greedy_strategy = EpsilonGreedyStrategy(
        env_spec=env.spec,
        total_step=int(1e5),
        max_epsilon=1.0,
        min_epsilon=0.02,
        decay_ratio=0.1)

    # exploration_fraction = 0.1
    # exploration_final_eps = 0.02
    # exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * n_epochs),
    #                              initial_p=1.0,
    #                              final_p=exploration_final_eps)

    algo = DQN(
        env=env,
        policy=policy,
        qf=qf,
        exploration_strategy=epilson_greedy_strategy,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        n_epochs=n_epochs,
        n_epoch_cycles=20,
        qf_lr=1e-3,
        discount=1.,
        min_buffer_size=1000,
        train_freq=1,
        smooth_return=False,
        target_network_update_freq=500,
        buffer_batch_size=32)

    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
