from preliminary.Agents.agent import Agent
from preliminary.Core.learning_param import LearningParam
from preliminary.q_learner import QBelief
from preliminary.Core.exploration import Exploration

from collections import defaultdict
import numpy as np

"""
    Policy gradient Ascent with approximate policy prediction
    Algorithm proposed by Zhang and Lesser 2010

    ZHANG, C.; LESSER, V.. Multi-Agent Learning with Policy Prediction.
    AAAI Conference on Artificial Intelligence, North America, jul. 2010.
    Available at: <https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/view/1885>. Date accessed: 06 Dec. 2017.
"""


class BasicPolicyExploration(Exploration):
    """
        Exploration method for PGAPP learner where the exploration is based on the e-greedy policy
    """

    def __init__(self):
        super(BasicPolicyExploration).__init__()
        self.current_policy = None
        self.e = LearningParam(init_value=1, end_value_ratio=0.01, n=10000)

    def set_belief(self, belief):
        super().set_belief(belief)
        self.current_policy = defaultdict(lambda: np.ones(belief.action_space) * 0.5)

    def update_policy(self, state, new_policy):
        new_policy = np.clip(new_policy, 0, 1) + self.e.get_value()
        new_policy /= sum(new_policy)  # with projection into a valid space
        self.current_policy[state] = new_policy

    def policy(self, observation):
        return self.current_policy[observation]

    def action(self, observation):
        return np.random.choice(np.arange(self.belief.action_space), p=self.policy(observation))

    def next_action(self, observation):
        return self.action(observation)


class PGAPPBelief(QBelief):
    """
        Belief for Policy gradient Ascent with approximate policy prediction
    """
    def __init__(self, action_space, gamma, eta=1):
        super(PGAPPBelief, self).__init__(action_space, gamma)
        self.gamma = gamma
        self.eta = eta
        self.beta = 0.01
        # equal prob policy
        self.state_policy = defaultdict(lambda: np.ones(action_space)/action_space)

    def update(self, state, action, reward, new_state, alpha):
        # Update Q function
        super(PGAPPBelief, self).update(state, action, reward, new_state, alpha)

        # state_policy = self.exploration.policy(state)
        state_policy = self.state_policy[state]  # State action value function
        state_v = np.dot(state_policy, self.Q[state])

        pd = defaultdict(lambda: np.zeros(self.action_space) * 1.0)  # partial derivative given current state
        d = defaultdict(lambda: np.zeros(self.action_space) * 1.0)  # partial derivative given current state
        gamma = self.gamma
        eta = self.eta

        for a in range(self.action_space):
            if abs(1.0 - state_policy[a]) < 0.01:
                pd[state][a] = self.Q[state][a] - state_v
            else:
                pd[state][a] = (self.Q[state][a] - state_v) / (1.0 - state_policy[a])
            d[state][a] = pd[state][a] - self.beta * abs(pd[state][a]) * state_policy[a]
            state_policy[a] += eta * d[state][a]
        self.exploration.update_policy(state, state_policy)

        new_policy = np.clip(state_policy, 0, 1)
        new_policy /= sum(new_policy)  # with projection into a valid space
        self.state_policy[state] = new_policy


class PGAPPLearner(Agent):
    """
        Agent for Policy gradient Ascent with approximate policy prediction
    """

    def __init__(self, n, a=0.2, end_ratio_a=1, g=0.9, exp_strategy=BasicPolicyExploration()):
        super(PGAPPLearner, self).__init__()
        self.alpha = a
        self.gamma = g  # discount factor
        self.a_rate = 1.0*end_ratio_a**(1.0/n)

        self.belief = PGAPPBelief(2, self.gamma)
        self.belief.set_exploration_strategy(exp_strategy)

    def init_action(self):
        return 0  # cooperate

    def choose_action(self, state):
        """
        Call this method to choose the correct action
        :param state: previous opponent action
        :return: the action chosen by the agent
        """
        return self.belief.get_next_action(state)

    def update_belief(self, state, action, new_state, reward):
        super(PGAPPLearner, self).update_belief(state, action, new_state, reward)
        self.belief.update(state, action, new_state, reward, self.alpha)
        self.alpha *= self.a_rate