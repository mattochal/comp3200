from LOLA.q_learner import QLearner


class NaiveLearnerWithHistory(QLearner):

    def __init__(self, n, h=4):
        super(NaiveLearnerWithHistory, self).__init__(n)
        self.state_history_window = []
        self.state_history_window_size = h

    def get_modified_state(self):
        new_state = []
        for s in self.state_history_window:
            new_state.extend(s)
        return tuple(new_state)

    def add_to_state_history(self, state):
        self.state_history_window.append(state)
        if len(self.state_history_window) > self.state_history_window_size:
            self.state_history_window.pop(0)

    def choose_action(self, prev_opp_action):
        """
        Call this method to choose the correct action
        :param prev_opp_action: previous opponent action
        :return: the action chosen by the agent
        """
        state = self.get_modified_state()
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
        self.add_to_state_history(state)
        state = self.get_modified_state()
        self.belief.update(state, action, reward, self.alpha)
        self.alpha *= self.a_rate