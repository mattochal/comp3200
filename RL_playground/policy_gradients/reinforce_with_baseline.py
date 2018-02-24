import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
from RL_playground.policy_gradients.lib import plotting
from RL_playground.policy_gradients.rps_env import RockPaperScissorsEnv
from RL_playground.policy_gradients.ipd_env import IteratedPDEnv
from RL_playground.policy_gradients.policy_estimator import PolicyEstimator
from RL_playground.policy_gradients.value_estimator import ValueEstimator


def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if done:
                break

            state = next_state

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict(transition.state)
            advantage = total_return - baseline_value
            # Update our value estimator
            estimator_value.update(transition.state, total_return)
            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action)
        print("", estimator_value.predict(transition.state), estimator_policy.predict(transition.state), sess.run(estimator_policy.output_layer, {estimator_policy.state:transition.state}))

    return stats


# env = RockPaperScissorsEnv(100, [50, 0, 0])
env = IteratedPDEnv(100)

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(env)
value_estimator = ValueEstimator(env)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, value_estimator, 500, discount_factor=0.9)

plotting.plot_episode_stats(stats, smoothing_window=25)
