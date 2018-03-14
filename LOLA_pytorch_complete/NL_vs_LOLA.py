import torch
from LOLA_pytorch.IPD_game import av_return
from torch.autograd import Variable
import numpy as np
from agent_pair import AgentPair
from LOLA_pytorch_complete.update_rules import update
from LOLA_pytorch_complete.IPD_value_fns import exact, policy_grad, modelling


def run(n=200, visualise=False, payoff1=[-1, -3, 0, -2], payoff2=[-1, 0, -3, -2], gamma=0.8, delta=0.1, eta=10, beta=0,
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

    # Term in f_lola update rule
    beta = Variable(torch.Tensor([beta]).type(dtype))

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

        V1, V2, dV1, dV2 = torch.autograd.grad(V2, (y1, y2), create_graph=True)

        # Update for Naive Learner agent
        y1.data += update("LOLA2", 0, (y1, y2), (dV1, dV2), delta, eta, beta)

        # Update for LOLA agent
        y2.data += update("LOLA1", 1, (y1, y2), (dV1, dV2), delta, eta, beta)

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


class NL_VS_LOLA(AgentPair):

    def run(self, seed):
        super(NL_VS_LOLA, self).run(seed)
        return run(**self.parameters)


if __name__ == "__main__":
    run(visualise=True)