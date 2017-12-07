from Core.exploration import *
from IPD.ipd_env import IPDEnv
from PGAPP.pgapp_agent import PGAPPLearner, BasicPolicyExploration
from tft_agent import TFTAgent


def basic_game(n, g, print_summary=False):
    # n = number of games

    env = IPDEnv()

    l1 = PGAPPLearner(n, g=g, exp_strategy=BasicPolicyExploration())
    l2 = PGAPPLearner(n, g=g, exp_strategy=BasicPolicyExploration())

    l1_action = l1.init_action()
    l2_action = l2.init_action()

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


if __name__ == "__main__":
    basic_game(50000, g=0.9, print_summary=True)

#
#
# def basic_game(n, g, print_summary=False):
#     # n = number of games
#
#     env = IPDEnv()
#     l1 = PGAPPLearner(n, g=g, exp_strategy=BasicPolicyExploration())
#     l2 = PGAPPLearner(n, g=g, exp_strategy=BasicPolicyExploration())
#
#     # l1 = QLearner(n, g=g, exp_strategy=EGreedyExploration(n, 1, 0.001))
#
#     l2.belief.set_opponent_belief(l1.belief)
#     l2.belief.set_payoffs(env.get_payoff(0), env.get_payoff(1))
#
#     l1.belief.set_opponent_belief(l2.belief)
#     l1.belief.set_payoffs(env.get_payoff(1), env.get_payoff(0))
#
#     l1_action = l1.init_action()
#     l2_action = l2.init_action()
#
#     for i_episode in range(n):
#         # state for l1 and l2 agent
#         state_l1 = (l2_action, l1_action)
#         state_l2 = (l1_action, l2_action)
#
#         next_action_l1 = l1.choose_action(state_l1)
#         next_action_l2 = l2.choose_action(state_l2)
#
#         # Get rewards based on players' actions
#         actions = [next_action_l1, next_action_l2]
#         reward = env.next_state(actions)
#
#         # Update l1 & l2's belief
#         next_state_l1 = (next_action_l2, next_action_l1)
#         next_state_l2 = (next_action_l1, next_action_l2)
#         l1.update_belief(state_l1, next_action_l1, next_state_l1, reward[0])
#         l2.update_belief(state_l2, next_action_l2, next_state_l2, reward[1])
#
#         l1_action = next_action_l1
#         l2_action = next_action_l2
#
#     if print_summary:
#         l1.belief.print_belief()
#         l2.belief.print_belief()
#
#         print("learner: ", l1.total_reward)
#         env.print_action_history(l1.transitions, last_n_moves=50)
#         print("pgapp: ", l2.total_reward)
#         env.print_action_history(l2.transitions, last_n_moves=50)
#
#
# if __name__ == "__main__":
#     basic_game(50000, g=0, print_summary=True)
