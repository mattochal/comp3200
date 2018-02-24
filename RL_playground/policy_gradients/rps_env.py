import Core.functions as cf
import numpy as np

SCORES = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]


class RockPaperScissorsEnv:

    action_space = 3
    observation_space = 1

    def __init__(self, no_games, params=None):
        self.params = params if params is not None else [1, 1, 1]
        self.no_games = no_games
        self.game_count = no_games

    def step(self, action):
        opp_action = np.random.choice(range(RockPaperScissorsEnv.action_space), p=cf.softmax(np.array(self.params)))
        reward = SCORES[action][opp_action]
        next_state = 1
        self.game_count -= 1
        done = (self.game_count == 0)
        return next_state, reward, done, None

    def reset(self):
        self.game_count = self.no_games
        return 1
