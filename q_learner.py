from Core.agent import Agent
from Core.belief import Belief
from Core.exploration import *


class QBelief(Belief):

    def __init__(self, action_space, discount_factor):
        super(QBelief, self).__init__(action_space)
        self.discount_factor = discount_factor

    def update(self, state, action, reward, new_state, alpha):
        super().update(state, action, reward, new_state, alpha)

        # TD(0) update for Q-learning
        # differs from SARSA by assuming greedy on one step ahead instead of the next action state
        td_target = reward + self.discount_factor * np.max(self.Q[new_state])
        self.Q[state][action] += alpha * (td_target - self.Q[state][action])
        self.state_visits[state][action] += 1


class QLearner(Agent):
    """
        Class for a Tit For Tat agent
    """

    def __init__(self, n, a=0.2, end_ratio_a=1, g=0.9, exp_strategy=EGreedyExploration(1, 0.05, 1000)):
        super(QLearner, self).__init__()
        self.alpha = a
        self.gamma = g  # discount factor
        self.a_rate = 1.0*end_ratio_a**(1.0/n)

        self.belief = QBelief(2, self.gamma)
        self.belief.set_exploration_strategy(exp_strategy)

    def init_action(self):
        next_action = 0
        return next_action

    def choose_action(self, prev_opp_action):
        """
        Call this method to choose the correct action
        :param prev_opp_action: previous opponent action
        :return: the action chosen by the agent
        """
        state = prev_opp_action
        next_action = self.belief.get_next_action(state)
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
        super(QLearner, self).update_belief(state, action, new_state, reward)
        self.belief.update(state, action, new_state, reward, self.alpha)
        self.alpha *= self.a_rate