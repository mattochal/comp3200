import numpy as np


def __grad(theta, u, s):
    grad = np.zeros(5)
    grad[s] = (1-u) / theta[s][0] - u / (1-theta[s][0])
    return grad


def __future_discounted_R(trajectory, t, a, gamma=0.9):
    reward = 0
    for i, transition in enumerate(trajectory[t:]):
        r = transition[3+a]
        reward += r * gamma**i
    return reward


def create_baseline(x1, x2, r1, r2, gamma):
    # Identity matrix
    I = np.eye(4)

    x1 = np.array(x1)
    x2 = np.array(x2)
    r1 = np.array(r1)
    r2 = np.array(r2)

    # State transition function
    P = np.concatenate((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)), 1)

    # This is just the rearrangement of equation found in D. Silver's L2 p25
    Zinv = np.linalg.inv(I - gamma * P[1:, :])

    # These are the exact value functions of the agent 1 and 2
    V1 = np.matmul(np.matmul(P[0, :], Zinv), r1)
    V2 = np.matmul(np.matmul(P[0, :], Zinv), r2)
    return lambda a: V1*(1-a) + V2*a


# This gets the average reward given the policies of both agents averaged over 100 epochs of 1000 PDs in IPD game
def get_rollouts(policy1, policy2, r1arr=[-1, -3, 0, -2], r2arr=[-1, 0, -3, -2], rollout_length=100, num_rollout=20, gamma=0.8, verbose=False):
    policy1 = policy1.data.cpu().numpy().tolist()
    policy2 = policy2.data.cpu().numpy().tolist()
    exp_reward = np.zeros((2, 5))

    b = create_baseline(policy1, policy2, r1arr, r2arr, gamma)

    for _ in range(num_rollout):
        trajectory = []
        s = [0, 0]  # initialisation only, gets overwritten later
        s[0] = np.random.choice([0, 1], p=[policy1[0][0], 1 - policy1[0][0]])
        s[1] = np.random.choice([0, 1], p=[policy2[0][0], 1 - policy2[0][0]])
        state_number = 0

        state_transition = [state_number]

        if verbose:
            print("Initial states are {}".format(s))

        for i in range(rollout_length):
            if s[0] == 0 and s[1] == 0:
                state_number = 1
                if verbose:
                    print("Both Cooperated")
            elif s[0] == 0 and s[1] == 1:
                state_number = 2
                if verbose:
                    print("Coop/Def")
            elif s[0] == 1 and s[1] == 0:
                state_number = 3
                if verbose:
                    print("Def/Coop")
            else:
                state_number = 4
                if verbose:
                    print("Both Defected!")

            state_transition.extend([s[0], s[1], r1arr[state_number - 1], r2arr[state_number - 1]])
            trajectory.append(state_transition)
            state_transition = [state_number]

            s[0] = np.random.choice([0, 1], p=[policy1[state_number][0], 1 - policy1[state_number][0]])
            s[1] = np.random.choice([0, 1], p=[policy2[state_number][0], 1 - policy2[state_number][0]])

        for t, transition in enumerate(trajectory):
            s = transition[0]
            u1 = transition[1]
            u2 = transition[2]
            exp_reward[0] += __grad(policy1, u1, s) * (__future_discounted_R(trajectory, t, 0) - b(0)) * gamma**t
            exp_reward[1] += __grad(policy2, u2, s) * (__future_discounted_R(trajectory, t, 1) - b(1)) * gamma**t

    R1 = exp_reward[0] / num_rollout
    R2 = exp_reward[1] / num_rollout
    return np.sum(R1), np.sum(R2)