import torch
from IPD_game import av_return
from IPD_rollouts import get_rollouts
from torch.autograd import Variable
import numpy as np


def run(n=200, visualise=False, payoff1=[-1, -3, 0, -2], payoff2=[-1, 0, -3, -2], gamma=0.8, delta=0.1, eta=10,
        init_policy1=[0.5, 0.5, 0.5, 0.5, 0.5], init_policy2=[0.5, 0.5, 0.5, 0.5, 0.5],
        rollout_length="not used but needed", num_rollout="not used but needed"):

    dtype = torch.FloatTensor

    for i, p in enumerate(init_policy1):
        if p is None:
            init_policy1[i] = np.random.random()

    for i, p in enumerate(init_policy2):
        if p is None:
            init_policy2[i] = np.random.random()

    init_policy1 = np.array(init_policy1, dtype="f")
    init_policy2 = np.array(init_policy2, dtype="f")
    y1 = np.log(np.divide(init_policy1, 1 - init_policy1)).reshape((5, 1))
    y2 = np.log(np.divide(init_policy2, 1 - init_policy2)).reshape((5, 1))

    y1 = Variable(torch.from_numpy(y1).float(), requires_grad=True)
    y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True)

    # Define rewards
    r1 = Variable(torch.Tensor(payoff1).type(dtype))
    r2 = Variable(torch.Tensor(payoff2).type(dtype))

    # Identity matrix
    I = Variable(torch.eye(4).type(dtype))

    # future reward discount factor
    gamma = Variable(torch.Tensor([0.8]).type(dtype))

    # Term in f_nl update rule
    delta = Variable(torch.Tensor([0.1]).type(dtype))

    # Term in f_lola update rule
    eta = Variable(torch.Tensor([10]).type(dtype))

    for epoch in range(n):
        x1 = torch.sigmoid(y1)
        x2 = torch.sigmoid(y2)

        # State transition function
        P = torch.cat((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)), 1)

        # This is just the rearrangement of equation found in D. Silver's L2 p25
        Zinv = torch.inverse(I - gamma * P[1:, :])

        # These are the exact value functions of the agent 1 and 2
        V1, V2 = get_rollouts(x1, x2)

        # 1st order gradient of 1st and 2nd agents' value functions w.r.t. both agents' parameters
        # Note: even though we generate the derivative with respect to both agents' policies
        #       later we only take the partial derivative with respect to agent's own policy
        dV1 = torch.autograd.grad(V1, (y1, y2), create_graph=True)
        dV2 = torch.autograd.grad(V2, (y1, y2), create_graph=True)

        # 2nd order gradient of 1st agent's value function w.r.t. 2nd agents' parameters
        # The for-loop exists as the gradients can only be calculated from scalar values
        d2V1 = [torch.autograd.grad(dV1[0][i], y2, create_graph=True)[0] for i in range(y1.size(0))]
        d2V1Tensor = torch.cat([d2V1[i] for i in range(y1.size(0))], 1)

        # Same for the other agent
        d2V2 = [torch.autograd.grad(dV2[1][i], y1, create_graph=True)[0] for i in range(y1.size(0))]
        d2V2Tensor = torch.cat([d2V2[i] for i in range(y2.size(0))], 1)

        # Update for Naive Learner agent
        y1.data += (delta * dV1[0] + delta * eta * torch.matmul(d2V2Tensor, dV1[1])).data

        # Update for LOLA agent
        y2.data += (delta * dV2[1] + delta * eta * torch.matmul(d2V1Tensor, dV2[0])).data

        if epoch % 100 == 0 and visualise:
            print("Epoch {}".format(epoch))
            print("x1: {}".format(x1.data.cpu().numpy().tolist()))
            print("V1: {}".format(V1.data[0]))
            print("x2: {}".format(x2.data.cpu().numpy().tolist()))
            print("V2: {}".format(V2.data[0]))
            print("Rewards: {}".format(av_return(x1, x2)))

    # return policy of both agents
    return torch.sigmoid(y1), torch.sigmoid(y2)

if __name__ == "__main__":
    run(visualise=True)

