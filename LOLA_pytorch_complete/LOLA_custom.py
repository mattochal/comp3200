import torch
from LOLA_pytorch.IPD_game import av_return
from torch.autograd import Variable
import numpy as np
from agent_pair import AgentPair
from LOLA_pytorch_complete.IPD_value_fns import exact, policy_grad, modelling
from LOLA_pytorch_complete.update_rules import update
import random


def run(n=200, visualise=False, record_trajectory=False, payoff1=[-1, -3, 0, -2], payoff2=[-1, 0, -3, -2], gamma=0.8, delta=0.1, eta=10, beta=0,
        init_policy1=[0.5, 0.5, 0.5, 0.5, 0.5], init_policy2=[0.5, 0.5, 0.5, 0.5, 0.5], rollout_length=50, num_rollout=50,
        agents=["LOLA1", "LOLA1"], value_fn="exact"):

    dtype = torch.FloatTensor
    result = {"epoch": []}

    init_policy1 = np.array(init_policy1, dtype="f")
    init_policy2 = np.array(init_policy2, dtype="f")

    # policy parameters
    y1 = np.log(np.divide(init_policy1, 1 - init_policy1)).reshape((5, 1))
    y2 = np.log(np.divide(init_policy2, 1 - init_policy2)).reshape((5, 1))
    y1 = Variable(torch.from_numpy(y1).float(), requires_grad=True)
    y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True)

    # modeled parameters, initially assumed 50% chance of cooperation in all states
    my1 = Variable(torch.Tensor([0, 0, 0, 0, 0]).type(dtype), requires_grad=True)
    my2 = Variable(torch.Tensor([0, 0, 0, 0, 0]).type(dtype), requires_grad=True)

    # Define rewards
    r1 = Variable(torch.Tensor(payoff1).type(dtype))
    r2 = Variable(torch.Tensor(payoff2).type(dtype))

    # future reward discount factor
    gamma = Variable(torch.Tensor([gamma]).type(dtype))

    # Term in f_nl update rule
    delta = Variable(torch.Tensor([delta]).type(dtype))

    # Term in f_lola1 update rule (first order LOLA)
    eta = Variable(torch.Tensor([eta]).type(dtype))

    # Term in f_lola2 update rule (higher order LOLA)
    beta = Variable(torch.Tensor([beta]).type(dtype))

    for epoch in range(n):
        x1 = torch.sigmoid(y1)
        x2 = torch.sigmoid(y2)

        if value_fn == "exact":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)
            y1.data += update(agents[0], 0, (y1, y2), (dV1, dV2), delta, eta, beta)
            y2.data += update(agents[1], 1, (y1, y2), (dV1, dV2), delta, eta, beta)

        elif value_fn == "policy_grad":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)
            _, _, dV1_from2, dV2_from1 = policy_grad(y1, y2, r1, r2, gamma)
            y1.data += update(agents[0], 0, (y1, y2), (dV1, dV2_from1), delta, eta, beta)
            y2.data += update(agents[1], 1, (y1, y2), (dV1_from2, dV2), delta, eta, beta)

        elif value_fn == "modelling":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)
            # dV1_from1, dV2_from1 are gradients of the value functions from perspective of agent 1
            _, _, dV1_from1, dV2_from1, dV1_from2, dV2_from2, my1, my2 = \
                modelling(y1, y2, r1, r2, gamma, rollout_length, num_rollout, my1, my2)

            y1.data += update(agents[0], 0, (y1, my2), (dV1_from1, dV2_from1), delta, eta, beta)
            y2.data += update(agents[1], 1, (my1, y2), (dV1_from2, dV2_from2), delta, eta, beta)

        if record_trajectory:
            result["epoch"].append({"P1": np.squeeze(x1.data.cpu().numpy()).tolist(),
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


class LOLA1_VS_LOLA1(AgentPair):

    def run(self, seed):
        super(LOLA1_VS_LOLA1, self).run(seed)
        self.parameters["agents"] = ["LOLA1", "LOLA1"]
        self.parameters["value_fn"] = "exact"
        self.parameters["record_trajectory"] = True
        return run(**self.parameters)


class LOLA1B_VS_LOLA1B(AgentPair):

    def run(self, seed):
        super(LOLA1B_VS_LOLA1B, self).run(seed)
        self.parameters["agents"] = ["LOLA1B", "LOLA1B"]
        self.parameters["value_fn"] = "exact"
        self.parameters["record_trajectory"] = True
        return run(**self.parameters)


class LOLA1_VS_LOLA1_PG(AgentPair):

    def run(self, seed):
        super(LOLA1_VS_LOLA1_PG, self).run(seed)
        self.parameters["agents"] = ["LOLA1", "LOLA1"]
        self.parameters["value_fn"] = "policy_grad"
        # self.parameters["record_trajectory"] = False
        return run(**self.parameters)


class LOLA1_VS_LOLA1_OM(AgentPair):

    def run(self, seed):
        super(LOLA1_VS_LOLA1_OM, self).run(seed)
        self.parameters["agents"] = ["LOLA1", "LOLA1"]
        self.parameters["value_fn"] = "modelling"
        return run(**self.parameters)

if __name__ == "__main__":
    run(visualise=True)