import torch
from LOLA_pytorch.IPD_game import av_return
from torch.autograd import Variable
import numpy as np


def run(n=200, visualise=False):
    dtype = torch.FloatTensor

    # parameters, theta of agent 1 - LOLA LEARNER
    y1 = Variable(torch.zeros(5, 1).type(dtype), requires_grad=True)

    # parameters, theta of agent 2 - LOLA LEARNER
    y2 = Variable(torch.zeros(5, 1).type(dtype), requires_grad=True)

    # Define rewards
    r1 = Variable(torch.Tensor([-1, -3, 0, -2]).type(dtype))
    r2 = Variable(torch.Tensor([-1, 0, -3, -2]).type(dtype))

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
        V1 = torch.matmul(torch.matmul(P[0, :], Zinv), r1)
        V2 = torch.matmul(torch.matmul(P[0, :], Zinv), r2)

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
        d2V2 = [torch.autograd.grad(dV2[1][i], y1, create_graph=True)[0] for i in range(y2.size(0))]
        d2V2Tensor = torch.cat([d2V2[i] for i in range(y2.size(0))], 1)

        # Update for LOLA agent
        y1.data += (delta * dV1[0] + delta * eta * torch.matmul(d2V2Tensor, dV1[1])).data

        # Update for LOLA agent
        y2.data += (delta * dV2[1] + delta * eta * torch.matmul(d2V1Tensor, dV2[0])).data

        if epoch % 20 == 0 and visualise:
            print('Epoch: ' + str(epoch))
            print("x1: {}".format(x1.data.cpu().numpy().tolist()))
            print("V1: {}".format(V1.data[0]))
            print("x2: {}".format(x2.data.cpu().numpy().tolist()))
            print("V2: {}".format(V2.data[0]))
            print("Rewards: {}".format(av_return(x1, x2)))

    # return policy of both agents
    return torch.sigmoid(y1), torch.sigmoid(y2)

if __name__ == "__main__":
    run(visualise=False)