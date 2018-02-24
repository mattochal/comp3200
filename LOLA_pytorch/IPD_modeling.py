import numpy as np

state_number = [[1, 2], [3, 4]]


# policy parameter estimation given true policy of the agents
def policy_param_estimation_from_rollouts(policy1, policy2, rollout_length=25, num_rollout=25):
    policy1 = policy1.data.cpu().numpy().tolist()  # turn the the policy into a list
    policy2 = policy2.data.cpu().numpy().tolist()

    # Number of times player 1 and 2 cooperated after ith type of game
    p1C = [0, 0, 0, 0, 0]
    p2C = [0, 0, 0, 0, 0]

    # number of different types of games played incl. first move
    games_type_count = [0, 0, 0, 0, 0]

    for _ in range(num_rollout):  # number of IPD games
        s = [0, 0]  # initial state
        s[0] = np.random.choice([0, 1], p=[policy1[0][0], 1 - policy1[0][0]])  # 0 means Cooperate, 1 means Defect
        s[1] = np.random.choice([0, 1], p=[policy2[0][0], 1 - policy2[0][0]])
        games_type_count[0] += 1

        if s[0] == 0:
            p1C[0] += 1  # mark agent 1 cooperates on first move
        if s[1] == 0:
            p2C[0] += 1  # mark agent 2 cooperates on first move

        for i in range(rollout_length):  # roll outs are just number of individual PD games
            a = state_number[s[0]][s[1]]
            games_type_count[a] += 1

            s[0] = np.random.choice([0, 1], p=[policy1[a][0], 1 - policy1[a][0]])  # next move of agent 1
            s[1] = np.random.choice([0, 1], p=[policy2[a][0], 1 - policy2[a][0]])  # next move of agent 2

            # if on the next turn the agents decide to cooperate then add it
            if s[0] == 0:
                p1C[a] += 1
            if s[1] == 0:
                p2C[a] += 1

    # It seems to be just adding random numbers together, why not just generate a bunch of random numbers at the end?
    # BECAUSE: the policy depends on previous state
    # this is estimating the policy from the roll outs as we don't know true policy function
    # BEWARE of division by 0 error

    pm1 = np.asarray(p1C) / np.asarray(games_type_count)
    pm2 = np.asarray(p2C) / np.asarray(games_type_count)

    # # for each nan field substitute with the corresponding policy value
    # print("B:", pm1)
    # print("B:", pm2)

    inds1 = np.where(np.isnan(pm1))
    inds2 = np.where(np.isnan(pm2))

    pm1[inds1] = np.take(policy1, inds1)
    pm2[inds2] = np.take(policy2, inds2)

    # print("A:", pm1)
    # print("A:", pm2)

    with np.errstate(divide='ignore'):
        pm1_y = np.log(np.divide(pm1, 1 - pm1))  # logit to get the a new set of proposed parameters
        pm2_y = np.log(np.divide(pm2, 1 - pm2))

    return pm1_y.reshape((5, 1)), pm2_y.reshape((5, 1))
