import torch
from IPD_game import av_return
from IPD_rollouts import get_rollouts
from torch.autograd import Variable


def run(n=3000, visualise=False):
    dtype = torch.FloatTensor

    # parameters, theta of agent 1 - NAIVE LEARNER
    y1 = Variable(torch.rand(5, 1).type(dtype), requires_grad=True)

    # parameters, theta of agent 2 - NAIVE LEARNER
    y2 = Variable(torch.rand(5, 1).type(dtype), requires_grad=True)

    # Define rewards
    r1 = Variable(torch.Tensor([-1, -3, 0, -2]).type(dtype))
    r2 = Variable(torch.Tensor([-1, 0, -3, -2]).type(dtype))

    # Identity matrix
    I = Variable(torch.eye(4).type(dtype))

    # future reward discount factor
    gamma = Variable(torch.Tensor([0.8]).type(dtype))

    # Term in f_nl update rule
    delta = Variable(torch.Tensor([0.1]).type(dtype))

    for epoch in range(n):
        x1 = torch.sigmoid(y1)
        x2 = torch.sigmoid(y2)

        # State transition function
        P = torch.cat((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)), 1)

        # This is just the rearrangement of equation found in D. Silver's L2 p25
        Zinv = torch.inverse(I - gamma * P[1:, :])

        # These are the value function of the agent 1 and 2 estimated from rollouts
        V1, V2 = get_rollouts(x1, x2)
        dV1 = Variable(torch.Tensor([V1]).type(dtype), requires_grad=True)
        dV2 = Variable(torch.Tensor([V2]).type(dtype), requires_grad=True)

        # Update for both agents
        y1.data += (delta * dV1).data
        y2.data += (delta * dV2).data

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