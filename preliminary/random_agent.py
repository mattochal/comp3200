import numpy as np


class RandomAgent:
    """
        Class for a Tit For Tat agent
    """

    def __init__(self):
        self.rewards = []

    def init_action(self):
        return self.choose_action(0)

    def choose_action(self, prev_opp_action):
        """
        Call this method to choose the correct action
        :param prev_opp_action: previous opponent action
        :return: the action chosen by the agent
        """
        return np.random.randint(0, 2)

    def update_reward(self, reward):
        """
        Update function for the belief system
        :param reward: the reward received
        :return: nothing
        """
        self.rewards.append(reward)
