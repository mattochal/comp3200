import torch
import numpy as np
from torch.autograd import Variable
from LOLA_pytorch_complete.IPD_modeling import policy_param_estimation_from_rollouts
from LOLA_pytorch_complete.IPD_rollouts import get_rollouts

dtype = torch.FloatTensor

# Identity matrix
I = Variable(torch.eye(4).type(dtype))

state_number = [[1, 2], [3, 4]]


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


def policy_grad(y1, y2, r1, r2, gamma):
    x1 = torch.sigmoid(y1)
    x2 = torch.sigmoid(y2)

    # Get the estimated value function of the agent 1 and 2 from rollouts based on true parameters
    V1, V2 = get_rollouts(x1, x2, r1, r2, gamma=gamma)

    # 1st order gradient of 1st and 2nd agents' value functions w.r.t. both agents' parameters
    # Note: even though we generate the derivative with respect to both agents' policies
    #       later we only take the partial derivative with respect to agent's own policy
    dV1 = torch.autograd.grad(V1, (y1, y2), create_graph=True)
    dV2 = torch.autograd.grad(V2, (y1, y2), create_graph=True)
    return dV1, dV2, V1, V2


def modelling(y1, y2, r1, r2, gamma, rollout_length, num_rollout, my1, my2):
    x1 = torch.sigmoid(y1)
    x2 = torch.sigmoid(y2)

    # my1, my2 are modelled parameters
    my1, my2 = run_rollouts(x1, x2, my1, my2, rollout_length=rollout_length, num_rollout=num_rollout)
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


# Estimating the policy parameters of agents given the true parameters
# and then estimation given true policy of the agents
# est_x is the estimated policy parameter at the previous step
def run_rollouts(x1, x2, est_x1=None, est_x2=None, rollout_length=25, num_rollout=25):

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