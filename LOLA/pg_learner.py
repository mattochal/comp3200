from Core.belief import QBelief
from Core.exploration import *


class PGLearner:
    """
        Policy gradient learner
    """

    def __init__(self, n, a=0.2, end_ratio_a=1, g=0.9, exp_strategy=EGreedyExploration(1, 0.05, 10000)):
        # n = number of games
        self.rewards = []

        self.alpha = a
        self.gamma = g  # discount factor
        self.a_rate = 1.0*end_ratio_a**(1.0/n)

        self.belief = QBelief(2, self.gamma)
        self.belief.set_exploration_strategy(exp_strategy)

        self.action_history = []

    def init_action(self):
        next_action = 0
        self.action_history.append(next_action)
        return next_action

    def choose_action(self, prev_opp_action):
        """
        Call this method to choose the correct action
        :param prev_opp_action: previous opponent action
        :return: the action chosen by the agent
        """
        state = prev_opp_action
        next_action = self.belief.get_next_action(state)
        self.action_history.append(next_action)
        return next_action

    def update_belief(self, state, action, new_state, reward):
        """
        Update function for the belief system
        :param new_state: new state
        :param state: the environment state
        :param action: the action chosen by the agent
        :param reward: the reward received
        :return: nothing
        """
        self.belief.update(state, action, new_state, reward, self.alpha)
        self.alpha *= self.a_rate

    def add_reward(self, reward):
        self.rewards.append(reward)

    def print_action_history(self):
        actions = ""
        for a in self.action_history:
            actions += str(a)
        print(actions)