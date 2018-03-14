import torch
from LOLA_pytorch.IPD_game import av_return
from torch.autograd import Variable
import numpy as np
from agent_pair import AgentPair
import random


def run(n=200, visualise=False, payoff1=[-1, -3, 0, -2], payoff2=[-1, 0, -3, -2], gamma=0.8, delta=0.1, eta=10,
        init_policy1=[0.5, 0.5, 0.5, 0.5, 0.5], init_policy2=[0.5, 0.5, 0.5, 0.5, 0.5],
        rollout_length="not used but needed", num_rollout="not used but needed"):

    dtype = torch.FloatTensor
    result = {"epoch": []}

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
    gamma = Variable(torch.Tensor([gamma]).type(dtype))

    # Term in f_nl update rule
    delta = Variable(torch.Tensor([delta]).type(dtype))

    # Term in f_lola update rule
    eta = Variable(torch.Tensor([eta]).type(dtype))

    for epoch in range(n):
        x1 = torch.sigmoid(y1)
        x2 = torch.sigmoid(y2)

        # State transition function, where axis = 1
        P = torch.cat((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)), 1)

        # This is just the rearrangement of equations found in D. Silver's L2 p25
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

        result["epoch"].append({"V1": np.squeeze(V1.data.cpu().numpy()).tolist(),
                                "V2": np.squeeze(V2.data.cpu().numpy()).tolist(),
                                "P1": np.squeeze(x1.data.cpu().numpy()).tolist(),
                                "P2": np.squeeze(x2.data.cpu().numpy()).tolist()})

        if epoch % 20 == 0 and visualise:
            print('Epoch: ' + str(epoch))
            print("x1: {}".format(x1.data.cpu().numpy().tolist()))
            print("V1: {}".format(V1.data[0]))
            print("x2: {}".format(x2.data.cpu().numpy().tolist()))
            print("V2: {}".format(V2.data[0]))
            print("Rewards: {}".format(av_return(x1, x2)))

    # return policy of both agents
    p1, p2 = torch.sigmoid(y1), torch.sigmoid(y2),
    result["P1"] = np.squeeze(p1.data.cpu().numpy()).tolist()
    result["P2"] = np.squeeze(p2.data.cpu().numpy()).tolist()
    return p1, p2, result


class LOLA_VS_LOLA(AgentPair):

    def run(self, seed):
        super(LOLA_VS_LOLA, self).run(seed)
        return run(**self.parameters)


if __name__ == "__main__":
    run(visualise=False)