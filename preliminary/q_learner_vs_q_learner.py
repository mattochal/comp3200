import matplotlib.pyplot as plt
import numpy as np

from preliminary.IPD.ipd_env import IPDEnv
from preliminary.q_learner import QLearner
from preliminary.Core.exploration import EGreedyExploration


def basic_game(n, g, print_summary=False):
    # n = number of games

    l1 = QLearner(n, g=g, exp_strategy=EGreedyExploration(n, 1, 0.01))
    l2 = QLearner(n, g=g, exp_strategy=EGreedyExploration(n, 1, 0.01))


    l1_action = l1.init_action()
    l2_action = l2.init_action()

    env = IPDEnv()

    for i_episode in range(n):
        # state for l1 and l2 agent
        state_l1 = (l2_action, l1_action)
        state_l2 = (l1_action, l2_action)

        next_action_l1 = l1.choose_action(state_l1)
        next_action_l2 = l2.choose_action(state_l2)

        # Get rewards based on players' actions
        actions = [next_action_l1, next_action_l2]
        reward = env.next_state(actions)

        # Update l1 & l2's belief
        next_state_l1 = (next_action_l2, next_action_l1)
        next_state_l2 = (next_action_l1, next_action_l2)
        l1.update_belief(state_l1, next_action_l1, reward[0], next_state_l1)
        l2.update_belief(state_l2, next_action_l2, reward[1], next_state_l2)

        l1_action = next_action_l1
        l2_action = next_action_l2

    if print_summary:
        print("L1 Belief")
        l1.belief.print_belief()

        print("L2 Belief")
        l2.belief.print_belief()

        print("L1: ", l1.total_reward)
        env.print_action_history(l1.transitions, last_n_moves=50)
        print("L2: ", l2.total_reward)
        env.print_action_history(l2.transitions, last_n_moves=50)

    return l1, l2


def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":
    n = 10000
    a1, a2, = basic_game(n, g=0.90, print_summary=True)

    data = np.zeros(n)
    for i, t in enumerate(a1.transitions):
        (state, action, reward, new_state) = t
        data[i] = reward

    plt.plot(moving_average(data))
    plt.show()





