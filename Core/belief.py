from collections import defaultdict

import numpy as np


class Belief:
    """
        Agent's Belief represents everything the agent knows about the environment.
        And then draws actions from policy based on the set exploration method.
    """

    def __init__(self, action_space, exploration=None):
        self.policy = defaultdict(lambda: np.zeros(action_space))  # policy function
        self.Q = defaultdict(lambda: np.zeros(action_space))  # State action value function
        self.V = defaultdict(lambda: 0.0)  # State value function
        self.state_visits = defaultdict(lambda: np.zeros(action_space))
        self.action_space = action_space
        self.exploration = None
        if exploration is not None: self.set_exploration_strategy(exploration)

    def get_next_action(self, state):
        """
        Gets chooses the next action according to the exploration method
        :param state:
        :return: action
        """
        pass

    def get_policy(self, state):
        """
        returns the probability distributions over action in the state
        :return: policy under state
        """
        pass

    def update(self, state, action, reward, new_state, alpha):
        """
        Update function for the belief system
        :param state: the normalised environment state
        :param action: the action chosen by the agent
        :param reward: the reward received
        :param new_state: the next state
        :param alpha: learning rate
        :return: nothing
        """
        pass

    def print_belief(self):
        np.set_printoptions(precision=3)
        for s in self.Q:
            print("state: {0} \t value: {1} \t policy: {2} \t visited: {3}".format(s, self.Q[s], self.get_policy(s), self.state_visits[s]))

    def set_exploration_strategy(self, exploration):
        if self.exploration is not None:
            raise Exception("Exploration already set")
        self.exploration = exploration
        exploration.set_belief(self)
        self.get_next_action = lambda o: exploration.next_action(o)
        self.get_policy = lambda o: exploration.policy(o)



class BeliefWithOppQBelief(Belief):

    def __init__(self, action_space, discount_factor):
        super().__init__(action_space)
        self.opponent_belief = None
        self.payoff = None
        self.opponent_payoff = None
        self.V = defaultdict(lambda: 0.0)
        self.gamma = 3
        self.eta = 0.001

    def set_opponent_belief(self, belief):
        self.opponent_belief = belief

    def set_payoffs(self, payoff, opp_payoff):
        self.payoff = payoff
        self.opponent_payoff = opp_payoff

    def update(self, state, action, reward, new_state, alpha):
        super().update(state, action, reward, new_state, alpha)
        state_policy = self.exploration.policy(state)
        state_v = np.dot(state_policy, self.Q[state])

        pd = defaultdict(lambda: np.zeros(self.action_space) * 1.0)  # partial derivative given current state
        d = defaultdict(lambda: np.zeros(self.action_space) * 1.0)  # partial derivative given current state
        gamma = self.gamma
        eta = self.eta

        for a in range(self.action_space):
            if abs(state_policy[a] - 1.0) < 0.01:
                pd[state][a] = self.Q[state][a] - state_v
                # print(pd[state][a])
            else:
                pd[state][a] = (self.Q[state][a] - state_v) / (1.0 - state_policy[a])
            d[state][a] = pd[state][a] - gamma * abs(pd[state][a]) * state_policy[a]
            state_policy[a] += eta * d[state][a]
        self.exploration.update_policy(state, state_policy)
        #
        # for s in self.exploration.current_policy:
        #     print(s, self.exploration.current_policy[s], sum(self.exploration.current_policy[s]))
        # pass


class BeliefWithOppQBeliefLOLA(Belief):

    def __init__(self, action_space, discount_factor):
        super().__init__(action_space, discount_factor)
        self.opponent_belief = None
        self.payoff = None
        self.opponent_payoff = None
        self.V = defaultdict(lambda: 0.0)
        self.gamma = 3
        self.eta = 0.001

    def set_opponent_belief(self, belief):
        self.opponent_belief = belief

    def set_payoffs(self, payoff, opp_payoff):
        self.payoff = payoff
        self.opponent_payoff = opp_payoff

    def update(self, state, action, reward, new_state, alpha):
        super().update(state, action, reward, new_state, alpha)
        state_policy = self.exploration.policy(state)
        state_v = np.dot(state_policy, self.Q[state])

        pd = defaultdict(lambda: np.zeros(self.action_space) * 1.0)  # partial derivative given current state
        d = defaultdict(lambda: np.zeros(self.action_space) * 1.0)  # partial derivative given current state
        gamma = self.gamma
        eta = self.eta

        for a in range(self.action_space):
            if abs(state_policy[a] - 1.0) < 0.01:
                pd[state][a] = self.Q[state][a] - state_v
                # print(pd[state][a])
            else:
                pd[state][a] = (self.Q[state][a] - state_v) / (1.0 - state_policy[a])
            d[state][a] = pd[state][a] - gamma * abs(pd[state][a]) * state_policy[a]
            state_policy[a] += eta * d[state][a]
        self.exploration.update_policy(state, state_policy)
        #
        # for s in self.exploration.current_policy:
        #     print(s, self.exploration.current_policy[s], sum(self.exploration.current_policy[s]))
        # pass


    def set_exploration_strategy(self, exploration):
        exploration.set_belief(self)
        self.exploration = exploration
        self.get_next_action = lambda o: exploration.next_action(o)
        self.get_policy = lambda o: exploration.policy(o)
