import numpy as np
from collections import defaultdict


class Exploration:

    def __init__(self):
        self.belief = None

    def set_belief(self, belief):
        self.belief = belief

    def policy(self, o):
        pass

    def action(self, o):
        pass

    def next_action(self, o):
        pass


class GreedyExploration(Exploration):

    def __init__(self):
        super(GreedyExploration).__init__()

    def policy(self, observation):
        A = np.zeros(self.belief.action_space, dtype=float)
        best_action = np.argmax(self.belief.Q[observation])
        A[best_action] = 1
        return A

    def action(self, observation):
        return np.argmax(self.belief.Q[observation])

    def next_action(self, observation):
        return self.action(observation)


class EGreedyExploration(Exploration):

    def __init__(self, n, init_epsilon=1, end_epsilon_ratio=0.05):
        super(EGreedyExploration).__init__()
        self.end_epsilon_ratio = end_epsilon_ratio
        self.epsilon = init_epsilon
        self.rate = 1.0*end_epsilon_ratio**(1.0/n)
        print(self.rate)

    def policy(self, observation):
        A = np.ones(self.belief.action_space, dtype=float) * self.epsilon / self.belief.action_space
        best_action = np.argmax(self.belief.Q[observation])
        A[best_action] += (1.0 - self.epsilon)
        return A

    def action(self, observation):
        return np.random.choice(np.arange(self.belief.action_space), p=self.policy(observation))

    def next_action(self, observation):
        self.epsilon *= self.rate
        return self.action(observation)


class BoltzmannExploration(Exploration):

    def __init__(self, n, init_t=5, end_t_ratio=0.05, cut_off=None):
        super(BoltzmannExploration).__init__()
        self.end_t_ratio = end_t_ratio
        self.t = init_t
        self.rate = 1.0 * end_t_ratio ** (1.0 / n)
        self.cut_off = cut_off

    def policy(self, observation):
        A = np.exp(self.belief.Q[observation] / self.t, dtype=float)
        A /= A.sum()
        return A

    def action(self, observation):
        return np.random.choice(np.arange(self.belief.action_space), p=self.policy(observation))

    def next_action(self, observation):

        if self.cut_off and self.t < self.cut_off:
            return np.argmax(self.belief.Q[observation])
        else:
            self.t *= self.rate
            return self.action(observation)







