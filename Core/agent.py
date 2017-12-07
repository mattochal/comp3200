

class Agent:
    """
        Abstract class for an agent
    """
    def __init__(self, record_transitions=True, belief=None):
        """
        :param record_transitions: if set to true the agent will store all of the transitions made
        """
        self.record_transitions = record_transitions
        if record_transitions:
            self.transitions = []

        self.total_reward = 0
        self.belief = belief

    def choose_action(self, state):
        """
        Call this method to choose the correct action
        :param state: the environment state
        :return: the action chosen by the agent
        """
        pass

    def set_belief(self, belief):
        """
        Set the belief of the agent, raises an exception if the belief is already set
        :param belief:
        :return: nothing
        """
        if self.belief is not None:
            raise Exception("QBelief already set")
        self.belief = belief

    def update_belief(self, state, action, reward, new_state):
        """
        Update function for the belief system
        :param state: the environment state
        :param action: the action chosen by the agent
        :param reward: the reward received
        :param new_state: the next state after taking the 'action' in the 'state'
        :return: nothing
        """
        self.total_reward += reward
        if self.record_transitions:
            self.transitions.append((state, action, reward, new_state))
