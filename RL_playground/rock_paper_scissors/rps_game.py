import numpy as np
from RL_playground.rock_paper_scissors.rps_agent import RPSAgentNaive, RPSAgentLearner
import tensorflow as tf

# 0 = rock, 1 = paper, 2 = scissors
SCORES = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])


def main(num_episodes=100, rollouts=10):
    agent1 = RPSAgentLearner()
    agent2 = RPSAgentNaive()
    print(agent1.policy_estimator.predict(0))

    discount_factor = 0.9

    for _ in range(num_episodes):
        episode = []

        for _ in range(rollouts):
            a1 = agent1.choose_action(0)
            a2 = agent2.choose_action(0)
            r1 = SCORES[a1][a2]
            r2 = SCORES[a2][a1]
            transition = {'state': 0, 'a1': a1, 'a2': a2, 'r1': r1, 'r2': r2}
            episode.append(transition)

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            total_return = sum(discount_factor ** i * t.r1 for i, t in enumerate(episode[t:]))

            # Calculate baseline/advantage
            baseline_value = agent1.estimate_value(transition.state)
            advantage = total_return - baseline_value

            # Update policy and value estimator
            agent1.update_value_estimator(transition.state, total_return)
            agent1.update_policy_estimator(transition.state, advantage, transition.action)
        print(agent1.policy_estimator.predict(0))


if __name__ == "__main__":
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        main()