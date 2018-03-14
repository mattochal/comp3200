import numpy as np
import copy


class AgentPair:

    def __init__(self, pair_setting, game_settings, other_settings=None):
        self.setting = copy.copy(pair_setting)
        self.game_settings = copy.copy(game_settings)
        self.other_settings = other_settings
        self.parameters = {**self.setting, **self.game_settings}

    def run(self, seed):
        """
        TO IMPLEMENT by child class
        returns a json dictionary of results
        """
        # np.random.seed(seed=seed)
        return {}

