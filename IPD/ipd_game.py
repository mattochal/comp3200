from tft_agent import TFTAgent

from IPD.ipd_env import IPDEnv
from LOLA.q_learner import QLearner

# number of game episodes
n = 250

# player type is optimistic TFT
tft = TFTAgent(False)
learner = QLearner()

tft_action = tft.init_action()
# a2_action = a2.init_action()
learner_action = learner.init_action()

env = IPDEnv()

# state is the (opp_prev_act, my_prev_act)

for i_episode in range(n):

    next_tft_action = tft.choose_action(learner_action)
    # next_a2_action = a2.choose_action(a2_action)
    state = (tft_action, learner_action)
    next_learner_action = learner.choose_action(state)

    reward = env.next_state([next_tft_action, next_learner_action])

    tft.update_reward(reward[0])
    # a2.update_reward(reward[1])
    print(state, next_learner_action, reward[1])

    learner.update_belief(state, next_learner_action, reward[1])
    # print(learner.alpha, learner.epsilon)

    tft_action = next_tft_action
    # a2_action = next_a2_action
    learner_action = next_learner_action

    # construct next state for Naive Learner
    # next_state_q, reward_q = env.next_state()

    # construct next state for TFT Agent
    # next_state_tft, reward_tft = env.next_state()

    # construct next state for PAVLOV Agent
    # next_state_pavlov, reward_pavlov = env.next_state()

    # construct next state for a Random Agent
    # next_state_random, reward_random = env.next_state( ["agents"], ["actions for each agent"])


learner.belief.print_belief()

"""
discount factor = 0 for static defecting player
discount factor high for tft player
"""


# print("a1: ", a1.rewards, " sum: ", sum(a1.rewards))
# print("a2: ", a2.rewards, " sum: ", sum(a2.rewards))

print("a1: ", tft.rewards, " sum: ", sum(tft.rewards))
print("a3: ", learner.rewards, " sum: ", sum(learner.rewards))