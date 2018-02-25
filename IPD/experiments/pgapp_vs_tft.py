from tft_agent import TFTAgent

from IPD.ipd_env import IPDEnv
from PGAPP.pgapp_agent import PGAPPLearner, BasicPolicyExploration


def basic_game(n, g, print_summary=False):
    # n = number of games

    env = IPDEnv()

    # player type is optimistic TFT
    tft = TFTAgent(True)
    learner = PGAPPLearner(n, g=g, exp_strategy=BasicPolicyExploration())

    tft_action = tft.init_action()
    learner_action = learner.init_action()

    for i_episode in range(n):
        # state for learning agent
        state_learner = (tft_action, learner_action)

        # state for tft agent
        state_tft = learner_action

        next_action_learner = learner.choose_action(state_learner)
        next_action_tft = tft.choose_action(state_tft)

        # Get rewards based on players' actions
        actions = [next_action_learner, next_action_tft]
        reward = env.next_state(actions)

        # Add reward to each agent
        tft.update_belief(state_tft, next_action_tft, reward[1], next_action_learner)

        # Update learner's belief
        new_state_learner = (next_action_tft, next_action_learner)
        learner.update_belief(state_learner, next_action_learner, reward[0], new_state_learner)

        tft_action = next_action_tft
        learner_action = next_action_learner

    if print_summary:
        learner.belief.print_belief()

        print("tft sum: ", tft.total_reward)
        env.print_action_history(tft.transitions, last_n_moves=50)
        print("learner: ", learner.total_reward)
        env.print_action_history(learner.transitions, last_n_moves=50)

    return tft, learner


if __name__ == "__main__":
    basic_game(50000, g=0.9, print_summary=True)

