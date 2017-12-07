import numpy as np


class IPDEnv:

    def __init__(self):
        # self.payoff = [[-1, -3], [0, -2]]
        self.payoff = [[0.3, 0.0], [0.5, 0.1]]  # from the Sandholm 95 paper
        # self.payoff = [[1, 0.0], [0.0, 0.0]]

    def next_state(self, action):
        a1_action = action[0]
        a2_action = action[1]

        a1_reward = self.payoff[a1_action][a2_action]
        a2_reward = self.payoff[a2_action][a1_action]

        return [a1_reward, a2_reward]

    def print_action_history(self, transitions, last_n_moves=None, line_width=150):
        if last_n_moves is None:
            last_n_moves = len(transitions)

        actions = ""
        count = 0
        for transition in transitions[-last_n_moves:]:
            a = transition[1]
            actions += "C" if a == 0 else "D"
            count+=1
            if count % line_width == 0:
                actions += "\n"
        print(actions)

    def get_payoff(self, player):
        if player == 0:
            return self.payoff
        else:
            return np.transpose(self.payoff)

