import torch
import numpy as np
from torch.autograd import Variable
# from myLOLA.IPD_modeling import policy_param_estimation_from_rollouts
# from LOLA_pytorch_complete.IPD_rollouts import get_rollouts

dtype = torch.FloatTensor

# Identity matrix
I = Variable(torch.eye(4).type(dtype))

state_number = [[1, 2], [3, 4]]


def __future_discounted_R(episode, t, agent, gamma):
    reward = 0
    for i, transition in enumerate(episode[t:]):
        r = transition[3 + agent]
        reward += r * gamma**i
    return reward


def __future_grad(episode, t, y1, y2):
    grad1 = np.zeros(5)
    grad2 = np.zeros(5)
    i = 0

    for i, transition in enumerate(episode[t:]):
        s = transition[0]
        u1 = transition[1]
        u2 = transition[2]

        grad1[s] += 1 - u1 - torch.sigmoid(y1[s])
        grad2[s] += 1 - u2 - torch.sigmoid(y2[s])

    grad1 /= i + 1
    grad2 /= i + 1

    return np.multiply(grad1, grad2), np.multiply(grad2, grad1)


def __create_baseline(baseline_name, trajectories):
    if baseline_name == "average_reward":
        b = np.zeros(2)
        for episode in trajectories:
            for transition in episode:
                b[0] += transition[3]
                b[1] += transition[4]
        b /= len(trajectories) * len(trajectories[0])

    return b


def exact(y1, y2, r1, r2, gamma):
    x1 = torch.sigmoid(y1)
    x2 = torch.sigmoid(y2)

    # State transition function, where axis = 1
    P = torch.cat((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)), 1)

    # This is just the rearrangement of equations found in D. Silver's L2 p25: Solving the Bellman Equation
    # Ignoring the s0 state
    Zinv = torch.inverse(I - gamma * P[1:, :])

    # These are the exact value functions of the agent 1 and 2
    # as a sum of the expected average reward given the probability of cooperation in state s0
    V1 = torch.matmul(torch.matmul(P[0, :], Zinv), r1)
    V2 = torch.matmul(torch.matmul(P[0, :], Zinv), r2)

    # 1st order gradient of 1st and 2nd agents' value functions w.r.t. both agents' parameters
    # Note: even though we generate the derivative with respect to both agents' policies
    #       later we only take the partial derivative with respect to agent's own policy
    dV1 = torch.autograd.grad(V1, (y1, y2), create_graph=True)
    dV2 = torch.autograd.grad(V2, (y1, y2), create_graph=True)

    return V1, V2, dV1, dV2


def policy_grad(y1, y2, r1, r2, gamma, rollout_length, num_rollout, baseline_name="average_reward"):
    x1 = torch.sigmoid(y1)
    x2 = torch.sigmoid(y2)

    # Get the estimated value function of the agent 1 and 2 from rollouts based on true parameters
    trajectories = get_rollout_trajectories(x1, x2, r1, r2, rollout_length, num_rollout)

    # Create a baseline for variance reduction
    b = __create_baseline(baseline_name, trajectories)

    d1R1=0
    d2R2=0

    d12R2 = 0
    d21R1 = 0

    for r in num_rollout:
        episode = trajectories[r]
        for t, transition in enumerate(episode):
            s = transition[0]
            u1 = transition[1]
            u2 = transition[2]
            r1 = transition[3]
            r2 = transition[4]

            grad1 = np.zeros(5)
            grad2 = np.zeros(5)

            grad1[s] = 1 - u1 - torch.sigmoid(y1[s])
            grad2[s] = 1 - u2 - torch.sigmoid(y2[s])

            d1R1 += grad1 * (gamma ** t) * (__future_discounted_R(episode, t, 0, gamma) - b[0])
            d2R2 += grad2 * (gamma ** t) * (__future_discounted_R(episode, t, 1, gamma) - b[1])

            future_grad = __future_grad(episode, t, y1, y2)
            d12R2 += (gamma ** t) * r2 * future_grad[0]
            d21R1 += (gamma ** t) * r1 * future_grad[1]

    return d1R1, d2R2, d12R2, d21R1


def modelling(y1, y2, r1, r2, gamma, rollout_length, num_rollout, my1, my2):
    x1 = torch.sigmoid(y1)
    x2 = torch.sigmoid(y2)

    # my1, my2 are modelled parameters
    my1, my2 = param_modelling(x1, x2, my1, my2, rollout_length=rollout_length, num_rollout=num_rollout)
    my1 = Variable(torch.from_numpy(my1).float(), requires_grad=True)
    my2 = Variable(torch.from_numpy(my2).float(), requires_grad=True)

    # Sigmoid function is to ensure that the parameters are within 1
    mx1 = torch.sigmoid(my1)
    mx2 = torch.sigmoid(my2)

    # Agent 1 knows own policy, and models agent 2's policy
    P1 = torch.cat((x1 * mx2, x1 * (1 - mx2), (1 - x1) * mx2, (1 - x1) * (1 - mx2)), 1)

    # Agent 2 knows its own policy, and models agent 1's policy
    P2 = torch.cat((mx1 * x2, mx1 * (1 - x2), (1 - mx1) * x2, (1 - mx1) * (1 - x2)), 1)

    # Assumed that the value function is known (based on parameters y)
    Zinv1 = torch.inverse(I - (gamma.expand_as(I) * P1[1:, :]))
    Zinv2 = torch.inverse(I - (gamma.expand_as(I) * P2[1:, :]))

    # Estimated value function
    V1 = torch.matmul(torch.matmul(P1[0, :], Zinv1), r1)
    V2 = torch.matmul(torch.matmul(P2[0, :], Zinv2), r2)

    dV1_from1 = torch.autograd.grad(V1, (y1, my2), create_graph=True)
    dV2_from1 = torch.autograd.grad(V2, (y1, my2), create_graph=True)
    dV1_from2 = torch.autograd.grad(V1, (my1, y2), create_graph=True)
    dV2_from2 = torch.autograd.grad(V2, (my1, y2), create_graph=True)
    return V1, V2, dV1_from1, dV2_from1, dV1_from2, dV2_from2, my1, my2


def full_modelling(y1, y2, r1, r2, gamma, rollout_length, num_rollout, my1, my2):
    # x1 = torch.sigmoid(y1)
    # x2 = torch.sigmoid(y2)
    #
    # # my1, my2 are modelled parameters
    # my1, my2 = param_modelling(x1, x2, my1, my2, rollout_length=rollout_length, num_rollout=num_rollout)
    # my1 = Variable(torch.from_numpy(my1).float(), requires_grad=True)
    # my2 = Variable(torch.from_numpy(my2).float(), requires_grad=True)
    #
    # # Sigmoid function is to ensure that the parameters are within 1
    # mx1 = torch.sigmoid(my1)
    # mx2 = torch.sigmoid(my2)
    #
    # V1, V2 = get_rollouts(mx1, mx2, r1, r2, gamma=gamma)
    #
    # dV1_from1 = torch.autograd.grad(V1, (y1, my2), create_graph=True)
    # dV2_from1 = torch.autograd.grad(V2, (y1, my2), create_graph=True)
    # dV1_from2 = torch.autograd.grad(V1, (my1, y2), create_graph=True)
    # dV2_from2 = torch.autograd.grad(V2, (my1, y2), create_graph=True)
    # return V1, V2, dV1_from1, dV2_from1, dV1_from2, dV2_from2, my1, my2
    return None


# This gets the average reward given the policies of both agents averaged over 100 epochs of 1000 PDs in IPD game
def get_rollout_trajectories(policy1, policy2, r1arr=[-1, -3, 0, -2], r2arr=[-1, 0, -3, -2], rollout_length=100, num_rollout=20):

    # True policies
    policy1 = policy1.data.cpu().numpy().tolist()
    policy2 = policy2.data.cpu().numpy().tolist()

    all_trajectories = []
    for _ in range(num_rollout):
        trajectory = []
        s = [np.random.choice([0, 1], p=[policy1[0][0], 1 - policy1[0][0]]),
             np.random.choice([0, 1], p=[policy2[0][0], 1 - policy2[0][0]])]

        prev_state_number = 0

        for i in range(rollout_length):
            if s[0] == 0 and s[1] == 0:
                state_number = 1

            elif s[0] == 0 and s[1] == 1:
                state_number = 2

            elif s[0] == 1 and s[1] == 0:
                state_number = 3

            else:
                state_number = 4

            state_transition = [prev_state_number, s[0], s[1], r1arr[state_number - 1], r2arr[state_number - 1]]
            trajectory.append(state_transition)
            prev_state_number = state_number

            s[0] = np.random.choice([0, 1], p=[policy1[state_number][0], 1 - policy1[state_number][0]])
            s[1] = np.random.choice([0, 1], p=[policy2[state_number][0], 1 - policy2[state_number][0]])

        all_trajectories.append(trajectory)

    return all_trajectories


# Estimating the policy parameters of agents given the true parameters
# and then estimation given true policy of the agents
# est_x is the estimated policy parameter at the previous step
def param_modelling(x1, x2, est_x1=None, est_x2=None, rollout_length=25, num_rollout=25):

    # Turn the the policies into a list
    policy1 = x1.data.cpu().numpy().tolist()
    policy2 = x2.data.cpu().numpy().tolist()

    est_x1 = policy1 if est_x1 is None else est_x1
    est_x2 = policy2 if est_x2 is None else est_x2

    # Lists keep track of the number of times player 1 and 2 cooperated after ith type of game
    p1C = [0, 0, 0, 0, 0]
    p2C = [0, 0, 0, 0, 0]

    # List keeps track of the number of different types of games played incl. first state
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

    # Ignoring the divisions by 0, estimate the policies
    with np.errstate(divide='ignore'):
        m_x1 = np.asarray(p1C) / np.asarray(games_type_count)
        m_x2 = np.asarray(p2C) / np.asarray(games_type_count)

        # Find the indices of policies that are 'nan' (caused by division by 0)
        inds1 = np.where(np.isnan(m_x1))
        inds2 = np.where(np.isnan(m_x2))

        # All the nan parameters should be defaulted to the previous policy estimation
        m_x1[inds1] = np.take(est_x1.data.cpu().numpy(), inds1)
        m_x2[inds2] = np.take(est_x2.data.cpu().numpy(), inds2)

        # Ignoring the divisions by 0,
        # Take logit - inverse of sigmoid - to estimate the parameters
        m_y1 = np.log(np.divide(m_x1, 1 - m_x1))
        m_y2 = np.log(np.divide(m_x2, 1 - m_x2))

    return m_y1.reshape((5, 1)), m_y2.reshape((5, 1))