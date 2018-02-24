from RL_playground.policy_gradients.lib import plotting
from RL_playground.policy_gradients.rps_env import RockPaperScissorsEnv
from RL_playground.policy_gradients.ipd_env import IteratedPDEnv
from RL_playground.policy_gradients.policy_estimator import PolicyEstimator
from RL_playground.policy_gradients.value_estimator import ValueEstimator

import numpy as np
import itertools
import collections
import tensorflow as tf


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=0.9):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a critic
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

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break

            state = next_state

        print("", estimator_value.predict(state), estimator_policy.predict(state),
              sess.run(estimator_policy.output_layer, {estimator_policy.state: state}))

    return stats

# env = RockPaperScissorsEnv(100, [1, 0, 0])
env = IteratedPDEnv(100)

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(env)
value_estimator = ValueEstimator(env)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = actor_critic(env, policy_estimator, value_estimator, 40)

plotting.plot_episode_stats(stats, smoothing_window=25)