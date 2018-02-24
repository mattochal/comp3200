import numpy as np

SCORES = np.array([[0.3, 0.0], [0.5, 0.1]])
SCORES = SCORES - SCORES.mean()
state_number = [[1, 2], [3, 4]]


class IteratedPDEnv():

    # Opponent policy
    TFT = 0  # Tit for tat

    action_space = 2
    observation_space = 5

    def __init__(self, no_games, policy=TFT):
        self.no_games = no_games
        self.game_count = no_games
        self.policy = policy
        self.__init_state = 0
        self.state = self.__init_state
        self.prev_action = 0

    def step(self, action):
        opp_action = self.prev_action
        reward = SCORES[action][opp_action]
        next_state = state_number[action][opp_action]
        self.prev_action = action
        self.game_count -= 1
        done = (self.game_count == 0)
        return next_state, reward, done, 0

    def reset(self):
        self.game_count = self.no_games
        return self.__init_state
