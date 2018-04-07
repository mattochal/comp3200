import numpy as np
import copy


class AgentPair:

    def __init__(self, settings, other_settings=None):
        self.other_settings = other_settings
        self.parameters = copy.deepcopy(settings)

    def run(self, seed):
        """
        TO IMPLEMENT by child class
        returns a json dictionary of results
        """
        # np.random.seed(seed=seed)
        return {}

