from Core.belief import QBelief


class LolaLearner:

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
        self.belief = QBelief(2, self.discount)
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