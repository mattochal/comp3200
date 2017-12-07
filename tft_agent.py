from Core.agent import Agent


class TFTAgent(Agent):
    """
        Class for a Tit For Tat agent
    """

    def __init__(self, optimistic):
        super(TFTAgent, self).__init__()
        self.optimistic = optimistic

    def init_action(self):
        return 0 if self.optimistic else 1

    def choose_action(self, state):
        """
        Call this method to choose the correct action
        :param state: it's previous opponent action
        :return: the action chosen by the agent
        """
        return state