from Core.belief import BeliefWithOppQBeliefLOLA
import numpy as np


# Initialise weights randomly to form probabilities of cooperation in the 5 states
theta = np.array([np.random.random(5), np.random.random(5)])

states = ["s0", "CC", "CD", "DC", "DD"]

policy = lambda theta: lambda t: [theta[t], 1-theta[t]]
policy1 = policy(theta[0])
policy2 = policy(theta[1])

p = lambda s: np.array([theta[0][s] * theta[1][s],
                        theta[0][s] * (1- theta[1][s]),
                        (1- theta[0][s]) * theta[1][s],
                        (1- theta[0][s]) * (1- theta[1][s])]).transpose()

P = lambda: np.matrix([p(s+1) for s in range(4)])

r1 = np.array([-1, -3, 0, -2]).transpose()
r2 = np.array([-1, 0, -3, -2]).transpose()

gamma = 0.95


# V1 = lambda: (np.array([theta[0][0] * theta[1][0],
#                         theta[0][0] * (1 - theta[1][0]),
#                         (1 - theta[0][0]) * theta[1][0],
#                         (1 - theta[0][0]) * (1 - theta[1][0])]).transpose() * (np.identity(4) / (np.identity(4) - gamma*P())) * r1)
# V2 = lambda: (np.array([theta[0][0] * theta[1][0],
#                         theta[0][0] * (1 - theta[1][0]),
#                         (1 - theta[0][0]) * theta[1][0],
#                         (1 - theta[0][0]) * (1 - theta[1][0])]).transpose() * (np.identity(4) / (np.identity(4) - gamma*P())) * r2)


V1 = lambda: (np.array([theta[0][0] * theta[1][0],
                        theta[0][0] * (1 - theta[1][0]),
                        (1 - theta[0][0]) * theta[1][0],
                        (1 - theta[0][0]) * (1 - theta[1][0])]).transpose() * (np.linalg.inv(np.identity(4) - gamma*np.matrix([[theta[0][s] * theta[1][s],
                                             theta[0][s] * (1 - theta[1][s]),
                                             (1 - theta[0][s]) * theta[1][s], (1 - theta[0][s]) * (1 - theta[1][s])] for s in range(1, 5)])) * r1))

p0 = np.array([theta[0][0] * theta[1][0],
              theta[0][0] * (1 - theta[1][0]),
              (1 - theta[0][0]) * theta[1][0],
              (1 - theta[0][0]) * (1 - theta[1][0])])

inv = np.linalg.inv(np.identity(4) - gamma*np.matrix([[theta[0][s] * theta[1][s],
                                                       theta[0][s] * (1 - theta[1][s]),
                                                       (1 - theta[0][s]) * theta[1][s],
                                                       (1 - theta[0][s]) * (1 - theta[1][s])] for s in range(1, 5)]))

theta = np.array([np.random.random(5), np.random.random(5)])
#
# p0 = np.array([theta[0][0] * theta[1][0],
#               theta[0][0] * (1 - theta[1][0]),
#               (1 - theta[0][0]) * theta[1][0],
#               (1 - theta[0][0]) * (1 - theta[1][0])])
#
# inv = np.linalg.inv(np.identity(4) - gamma*np.matrix([[theta[0][s] * theta[1][s],
#                                                        theta[0][s] * (1 - theta[1][s]),
#                                                        (1 - theta[0][s]) * theta[1][s],
#                                                        (1 - theta[0][s]) * (1 - theta[1][s])] for s in range(1, 5)]))
#
# r = np.array([r2]).transpose()
#
#
# V = lambda theta: p0 * inv * r


theta = np.array([np.random.random(5), np.random.random(5)])

V1 = lambda theta: (np.array([theta[0][0] * theta[1][0],
              theta[0][0] * (1 - theta[1][0]),
              (1 - theta[0][0]) * theta[1][0],
              (1 - theta[0][0]) * (1 - theta[1][0])]) * np.linalg.inv(np.identity(4) - gamma*np.matrix([[theta[0][s] * theta[1][s],
                                                       theta[0][s] * (1 - theta[1][s]),
                                                       (1 - theta[0][s]) * theta[1][s],
                                                       (1 - theta[0][s]) * (1 - theta[1][s])] for s in range(1, 5)])) * np.array([r1]).transpose())
V2 = lambda theta: (np.array([theta[0][0] * theta[1][0],
              theta[0][0] * (1 - theta[1][0]),
              (1 - theta[0][0]) * theta[1][0],
              (1 - theta[0][0]) * (1 - theta[1][0])]) * np.linalg.inv(np.identity(4) - gamma*np.matrix([[theta[0][s] * theta[1][s],
                                                       theta[0][s] * (1 - theta[1][s]),
                                                       (1 - theta[0][s]) * theta[1][s],
                                                       (1 - theta[0][s]) * (1 - theta[1][s])] for s in range(1, 5)])) * np.array([r2]).transpose())


class LolaNaiveLearner:

    def __init__(self, n):
        # n = number of games
        self.rewards = []

        self.epsilon = 1
        self.alpha = 0.2

        decrease_a_by = 1
        decrease_e_by = 0.05

        self.a_rate = 1.0*decrease_a_by**(1.0/n)
        self.e_rate = 1.0*decrease_e_by**(1.0/n)

        self.discount = 0.9
        self.belief = BeliefWithOppQBeliefLOLA(2, self.discount)
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
        next_action = self.belief.get_epsilon_greedy_action(state, self.epsilon)
        self.epsilon *= self.e_rate
        self.action_history.append(next_action)
        return next_action

    def update_belief(self, state, action, reward):
        """
        Update function for the belief system
        :param state: the environment state
        :param action: the action chosen by the agent
        :param reward: the reward received
        :return: nothing
        """
        self.belief.update(state, action, reward, self.alpha)
        self.alpha *= self.a_rate

    def add_reward(self, reward):
        self.rewards.append(reward)

    def print_action_history(self):
        actions = ""
        for a in self.action_history:
            actions += str(a)
        print(actions)

